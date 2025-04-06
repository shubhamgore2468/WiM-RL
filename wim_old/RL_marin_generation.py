import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from wim import WIMInference

@dataclass
class RLConfig:
    """Configuration for the RL-based margin generation."""
    learning_rate: float = 5e-5
    kl_coef: float = 0.1
    discount_factor: float = 0.99
    ppo_epochs: int = 4
    ppo_mini_batch_size: int = 4
    max_grad_norm: float = 0.5
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01

class MarginRewardModel:
    """Model to compute rewards for generated margins."""
    
    def __init__(self, model_id: str, device: str = "cuda"):
        """Initialize the reward model.
        
        Args:
            model_id: The ID of the model to use for reward computation.
            device: The device to use for computation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.device = device
        
    def compute_reward(self, margins: List[str], query: str, classification_results: List[bool]) -> torch.Tensor:
        """Compute rewards for a batch of margins.
        
        The reward combines:
        1. Relevance to the query
        2. Conciseness (penalize overly verbose margins)
        3. Information density
        4. Agreement with classifier (higher reward if classifier agrees)
        
        Args:
            margins: List of generated margins
            query: The original query
            classification_results: Whether each margin was classified as relevant
            
        Returns:
            Tensor of rewards for each margin
        """
        rewards = []
        
        for margin, is_relevant in zip(margins, classification_results):
            with torch.no_grad():
                # Construct prompt to evaluate margin quality
                prompt = f"Query: {query}\nMargin: {margin}\n\nRate the quality of this margin from 0 to 10 based on relevance and information density. A good margin should contain key information relevant to the query."
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                
                # Extract score from the generated text
                generated_text = self.tokenizer.decode(outputs.sequences[0][-5:], skip_special_tokens=True)
                try:
                    # Try to extract a numeric score from the response
                    score = float(''.join(c for c in generated_text if c.isdigit() or c == '.'))
                    # Normalize score to 0-1 range
                    score = min(max(score / 10.0, 0.0), 1.0)
                except:
                    # Default score if parsing fails
                    score = 0.5
                
            # Additional reward components
            length_penalty = min(1.0, 100 / max(10, len(margin.split())))  # Prefer concise margins
            classifier_agreement = 1.0 if is_relevant else 0.2  # Reward if classifier agrees it's relevant
            
            # Combine reward components
            reward = (0.6 * score) + (0.2 * length_penalty) + (0.2 * classifier_agreement)
            rewards.append(reward)
            
        return torch.tensor(rewards, device=self.device)

class RLMarginGenerator:
    """Generate margins using reinforcement learning."""
    
    def __init__(
        self, 
        model_id: str,
        reward_model_id: str = None,
        rl_config: RLConfig = None,
        device: str = "cuda"
    ):
        """Initialize the RL-based margin generator.
        
        Args:
            model_id: The ID of the model to use for generation.
            reward_model_id: The ID of the model to use for reward computation.
                If None, uses the same model as the generator.
            rl_config: Configuration for the RL algorithm.
            device: The device to use for computation.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Initialize the policy model (for generating margins)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        
        # Initialize the reference model (for KL divergence computation)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False

        for param in self.policy_model.parameters():
            param.requires_grad = True
            
        # Initialize the reward model
        if reward_model_id is None:
            reward_model_id = model_id
        self.reward_model = MarginRewardModel(reward_model_id, device)
        
        # Initialize the RL config
        self.rl_config = rl_config if rl_config is not None else RLConfig()
        
        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_model.parameters(),
            lr=self.rl_config.learning_rate
        )
        
        # Initialize the WIM inference
        self.wim = WIMInference(self.policy_model, self.tokenizer)

        # Prime the KV caches with a dummy input to ensure they're properly initialized
        # This is a critical step to fix the "Cache only has 0 layers" error
        # dummy_input = "This is a dummy input to initialize KV cache."
        # self.wim.prefill_text_with_kv_cache(dummy_input, self.wim.wim_kv_cache)
        # self.wim.prefill_text_with_kv_cache(dummy_input, self.wim.classifier_kv_cache)
        
        # # Now clear the caches
        # self.wim.shrink_kv_cache_from_end(0, self.wim.wim_kv_cache)
        # self.wim.shrink_kv_cache_from_end(0, self.wim.classifier_kv_cache)
        
    def _compute_logprobs(self, model, input_ids, attention_mask, labels):
        """Compute log probabilities for a batch of sequences."""
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Get log probs for each token in the sequence
            token_log_probs = torch.gather(
                log_probs[:, :-1, :], 2, labels[:, 1:, None]
            ).squeeze(-1)
            
            # Mask out padding tokens
            mask = (labels[:, 1:] != self.tokenizer.pad_token_id).float()
            token_log_probs = token_log_probs * mask
            
            # Sum log probs over sequence
            seq_log_probs = token_log_probs.sum(dim=1)
            
            return seq_log_probs
            
    def _compute_kl_divergence(self, input_ids, attention_mask, labels):
        """Compute KL divergence between policy and reference model."""
        policy_log_probs = self._compute_logprobs(
            self.policy_model, input_ids, attention_mask, labels
        )
        with torch.no_grad():
            ref_log_probs = self._compute_logprobs(
                self.ref_model, input_ids, attention_mask, labels
            )
            
        kl_div = policy_log_probs - ref_log_probs
        return kl_div
    
    def generate_rl_margin(
        self,
        segment: str,
        query: str,
        extractive_summary_prompt: str,
        classification_prompt: str,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 1.0,
        early_stopping: bool = True,
    ):
        """Generate a margin using the current policy model."""

        # # Ensure KV cache is properly initialized
        # if self.wim.wim_kv_cache.get_seq_length() == 0:
        #     # Initialize with a dummy input if empty
        #     dummy_input = "Initializing KV cache."
        #     self.wim.prefill_text_with_kv_cache(dummy_input, self.wim.wim_kv_cache)
        #     self.wim.shrink_kv_cache_from_end(0, self.wim.wim_kv_cache)
        
        # # Same for classifier KV cache
        # if self.wim.classifier_kv_cache.get_seq_length() == 0:
        #     dummy_input = "Initializing classifier KV cache."
        #     self.wim.prefill_text_with_kv_cache(dummy_input, self.wim.classifier_kv_cache)
        #     self.wim.shrink_kv_cache_from_end(0, self.wim.classifier_kv_cache)
    

        try:
            # Prefill the segment
            prefilled_tokens_before_extractive_summary, _, _ = self.wim.prefill_text_with_kv_cache(
                segment, self.wim.wim_kv_cache
            )
            
            # Prefill the extractive summary prompt
            _, _, extractive_summary_outputs = self.wim.prefill_text_with_kv_cache(
                extractive_summary_prompt.format(query=query), self.wim.wim_kv_cache
            )
            
            # Generate the margin
            margin = self.wim.generate_text_with_kv_cache(
                max_new_tokens=max_new_tokens,
                previous_logits=extractive_summary_outputs["logits"],
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                early_stopping=early_stopping,
                kv_cache=self.wim.wim_kv_cache,
            )
            
            # Shrink the KV cache back to before the extractive summary prompt
            self.wim.shrink_kv_cache_from_end(
                new_size=prefilled_tokens_before_extractive_summary,
                kv_cache=self.wim.wim_kv_cache,
            )
            
            # Classify the margin
            classification_input = classification_prompt.format(query=query, answer=margin)
            _, _, classification_prompt_logits = self.wim.prefill_text_with_kv_cache(
                classification_input, self.wim.classifier_kv_cache
            )
            
            classification_output = self.wim.generate_text_with_kv_cache(
                max_new_tokens=10,
                previous_logits=classification_prompt_logits["logits"],
                do_sample=False,
                top_p=0.9,
                temperature=1.0,
                early_stopping=early_stopping,
                kv_cache=self.wim.classifier_kv_cache,
            )
            
            # Parse the classification output
            is_relevant = self._parse_classifier_output(classification_output)
            
            # Clear the classifier KV cache
            self.wim.shrink_kv_cache_from_end(
                new_size=0, kv_cache=self.wim.classifier_kv_cache
            )
            
            return margin, is_relevant
        
        except Exception as e:
            print(f"Error during margin generation: {e}")
            return "Error generating margin", False
        
    def _parse_classifier_output(self, output: str) -> bool:
        """Parse the classification output to determine if the margin is relevant."""
        output = output.replace("```", "").strip()
        output = output.split("#")[0]
        if output.endswith("YES"):
            return True
        else:
            return False
            
    def train_rl_margin_generator(
        self,
        segments: List[str],
        query: str,
        extractive_summary_prompt: str,
        classification_prompt: str,
        num_episodes: int = 10,
        max_new_tokens: int = 100,
    ):
        """Train the margin generator using PPO."""

        # RL model in training phase
        self.tokenizer.pad_token = self.tokenizer.eos_token

        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Sample a batch of segments
            batch_size = min(self.rl_config.ppo_mini_batch_size, len(segments))
            segment_indices = np.random.choice(len(segments), batch_size, replace=False)
            batch_segments = [segments[i] for i in segment_indices]
            
            # Generate margins using the current policy
            margins = []
            is_relevant_list = []
            
            for segment in tqdm(batch_segments, desc="Generating margins"):
                # Clear KV caches
                self.wim.shrink_kv_cache_from_end(0, self.wim.wim_kv_cache)
                self.wim.shrink_kv_cache_from_end(0, self.wim.classifier_kv_cache)
                
                margin, is_relevant = self.generate_rl_margin(
                    segment=segment,
                    query=query,
                    extractive_summary_prompt=extractive_summary_prompt,
                    classification_prompt=classification_prompt,
                    max_new_tokens=max_new_tokens,
                )
                
                margins.append(margin)
                is_relevant_list.append(is_relevant)
            
            # Compute rewards for the generated margins
            rewards = self.reward_model.compute_reward(margins, query, is_relevant_list)
            
            # Prepare inputs for PPO update
            inputs = []
            for segment, margin in zip(batch_segments, margins):
                # Tokenize the segment + extractive summary prompt + margin
                context = segment + extractive_summary_prompt.format(query=query) + margin
                input_tokens = self.tokenizer(context, return_tensors="pt", padding=True).to(self.device)
                
                # Create labels for computing log probs
                labels = input_tokens.input_ids.clone()
                # Mask out tokens we don't want to compute loss for
                context_without_margin = segment + extractive_summary_prompt.format(query=query)
                context_tokens = len(self.tokenizer(context_without_margin, return_tensors="pt").input_ids[0])
                labels[:, :context_tokens] = -100  # Mask out non-margin tokens
                
                inputs.append({
                    "input_ids": input_tokens.input_ids,
                    "attention_mask": input_tokens.attention_mask,
                    "labels": labels,
                })
            
            # PPO update
            for _ in range(self.rl_config.ppo_epochs):
                for i in range(len(inputs)):
                    self.optimizer.zero_grad()
                    
                    # Compute KL divergence
                    kl_div = self._compute_kl_divergence(
                        inputs[i]["input_ids"],
                        inputs[i]["attention_mask"],
                        inputs[i]["labels"],
                    )
                    
                    # Compute policy loss
                    policy_loss = -rewards[i] + self.rl_config.kl_coef * kl_div
                    
                    # Backward pass
                    policy_loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        self.rl_config.max_grad_norm,
                    )
                    
                    # Update policy model
                    self.optimizer.step()
            
            # Log metrics
            avg_reward = rewards.mean().item()
            avg_kl_div = kl_div.mean().item()
            relevance_rate = sum(is_relevant_list) / len(is_relevant_list)
            
            print(f"Average reward: {avg_reward:.4f}")
            print(f"Average KL divergence: {avg_kl_div:.4f}")
            print(f"Relevance rate: {relevance_rate:.4f}")
    
    def save_model(self, output_dir: str):
        """Save the trained model."""
        self.policy_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)




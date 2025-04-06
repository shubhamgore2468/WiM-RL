from wim import WIMInference
from RL_margin_generation import RLMarginGenerator

import tiktoken
from nltk import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

class WIMRLInference(WIMInference):
    """Extended WIM inference class that uses RL-trained model for margin generation."""
    
    def __init__(self, model, tokenizer, rl_margin_generator=None):
        """Initialize the WIM inference with an optional RL margin generator.
        
        Args:
            model: The base model for generation.
            tokenizer: The tokenizer.
            rl_margin_generator: Optional RL-based margin generator.
        """
        super().__init__(model, tokenizer)
        self.rl_margin_generator = rl_margin_generator
        
    def process_with_rl_margins(
        self,
        context: str,
        query: str,
        system_message: str,
        extractive_summary_prompt: str,
        classification_prompt: str,
        final_answer_prompt: str,
        min_tokens_segment: int = 4096,
        max_new_tokens_extractive_summary: int = 100,
        max_new_tokens_final_answer: int = 100,
        max_new_tokens_classification: int = 10,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 1.0,
        early_stopping: bool = True,
        use_rl_generator: bool = True,
        print_step_summary: bool = False,
    ):
        """Process the context using the WIM approach with RL-enhanced margin generation.
        
        Args:
            context: The full document context.
            query: The user query.
            system_message: The system message prompt.
            extractive_summary_prompt: Template for extractive summary prompt.
            classification_prompt: Template for classification prompt.
            final_answer_prompt: Template for final answer prompt.
            min_tokens_segment: Minimum number of tokens per segment.
            max_new_tokens_extractive_summary: Maximum number of tokens to generate for margin.
            max_new_tokens_final_answer: Maximum number of tokens to generate for final answer.
            max_new_tokens_classification: Maximum number of tokens to generate for classification.
            do_sample: Whether to use sampling for generation.
            top_p: Top-p sampling parameter.
            temperature: Temperature for generation.
            early_stopping: Whether to use early stopping.
            use_rl_generator: Whether to use the RL-based margin generator.
            print_step_summary: Whether to print a summary for each step.
            
        Returns:
            final_answer: The generated answer.
            positive_margins: List of relevant margins used.
        """
        # Clear KV caches
        self.shrink_kv_cache_from_end(0, self.wim_kv_cache)
        self.shrink_kv_cache_from_end(0, self.classifier_kv_cache)
        
        # Segment the context
        segments = self._chunk_text_to_segments(context, min_tokens_segment)
        
        # Prefill the system message
        _, _, _ = self.prefill_text_with_kv_cache(system_message, self.wim_kv_cache)
        
        positive_margins = []
        
        with torch.no_grad():
            for segment_index in range(len(segments)):
                segment = segments[segment_index]
                
                if use_rl_generator and self.rl_margin_generator is not None:
                    print("# Use RL-based margin generator")
                    margin, is_relevant = self.rl_margin_generator.generate_rl_margin(
                        segment=segment,
                        query=query,
                        extractive_summary_prompt=extractive_summary_prompt,
                        classification_prompt=classification_prompt,
                        max_new_tokens=max_new_tokens_extractive_summary,
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        early_stopping=early_stopping,
                    )
                else:
                    print('# Use standard WIM approach')
                    prefilled_tokens_before_extractive_summary, _, _ = self.prefill_text_with_kv_cache(
                        segment, self.wim_kv_cache
                    )
                    
                    formatted_extractive_summary = extractive_summary_prompt.format(query=query)
                    _, _, extractive_summary_outputs = self.prefill_text_with_kv_cache(
                        formatted_extractive_summary, self.wim_kv_cache
                    )
                    
                    margin = self.generate_text_with_kv_cache(
                        max_new_tokens=max_new_tokens_extractive_summary,
                        previous_logits=extractive_summary_outputs["logits"],
                        do_sample=do_sample,
                        top_p=top_p,
                        temperature=temperature,
                        early_stopping=early_stopping,
                        kv_cache=self.wim_kv_cache,
                    )
                    
                    # Shrink KV cache back to before extractive summary
                    self.shrink_kv_cache_from_end(
                        new_size=prefilled_tokens_before_extractive_summary,
                        kv_cache=self.wim_kv_cache,
                    )
                    
                    # Classify the margin
                    classification_input = classification_prompt.format(query=query, answer=margin)
                    _, _, classification_prompt_logits = self.prefill_text_with_kv_cache(
                        classification_input, self.classifier_kv_cache
                    )
                    
                    classification_output = self.generate_text_with_kv_cache(
                        max_new_tokens=max_new_tokens_classification,
                        previous_logits=classification_prompt_logits["logits"],
                        do_sample=False,
                        top_p=top_p,
                        temperature=temperature,
                        early_stopping=early_stopping,
                        kv_cache=self.classifier_kv_cache,
                    )
                    
                    is_relevant = self._parse_classifier_output(classification_output)
                    
                    # Clear the classifier KV cache
                    self.shrink_kv_cache_from_end(
                        new_size=0, kv_cache=self.classifier_kv_cache
                    )
                
                if is_relevant:
                    positive_margins.append(margin)
                
                if print_step_summary:
                    print({
                        "step": segment_index,
                        "prefilled_tokens_so_far": self.wim_kv_cache.get_seq_length(),
                        "margin": margin.strip(),
                        "classification_result": is_relevant,
                    })
            
            # Prefill the concatenated margins and the prompt to ask the final answer
            concatenated_margins = "".join(positive_margins)
            formatted_final_answer = final_answer_prompt.format(
                margins=concatenated_margins, query=query
            )
            
            _, _, final_answer_prefill_outputs = self.prefill_text_with_kv_cache(
                formatted_final_answer, self.wim_kv_cache
            )
            
            # Generate the final answer
            final_answer = self.generate_text_with_kv_cache(
                max_new_tokens=max_new_tokens_final_answer,
                previous_logits=final_answer_prefill_outputs["logits"],
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                early_stopping=early_stopping,
                kv_cache=self.wim_kv_cache,
            )
            
            return final_answer, positive_margins
    
    def _chunk_text_to_segments(self, text, min_tokens_segment=4096):
        """Chunk text into segments of approximately min_tokens_segment tokens."""
        
        
        tokenizer = tiktoken.encoding_for_model("gpt-4-turbo")
        segments = []
        current_segment = ""
        sentences = sent_tokenize(text)
        curr_tokens = 0
        
        for line in sentences:
            tokens = len(tokenizer.encode(line))
            if curr_tokens + tokens > min_tokens_segment:
                segments.append(current_segment)
                current_segment = ""
                curr_tokens = 0
            
            current_segment += line + " "
            curr_tokens += tokens
        
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _parse_classifier_output(self, output: str) -> bool:
        """Parse the classification output to determine if the margin is relevant."""
        output = output.replace("```", "").strip()
        output = output.split("#")[0]
        if output.endswith("YES"):
            return True
        else:
            return False


def run_wim_rl(
    model_id: str,
    model_id_rl: str,
    input_document: str,
    query: str,
    use_rl_generator: bool = True,
    train_rl_generator: bool = False,
    num_episodes: int = 10,
    output_model_dir: str = None,
    # attn_implementation: str = "flash_attention_2",
    attn_implementation:str = 'sdpa',
    dtype: str = "bfloat16",
    min_tokens_segment: int = 4096,
    max_new_tokens_extractive_summary: int = 100,
    max_new_tokens_final_answer: int = 50,
    max_new_tokens_classification: int = 10,
    do_sample: bool = True,
    top_p: float = 0.9,
    temperature: float = 1.0,
    early_stopping: bool = True,
    print_step_summary: bool = False,
    user_header: str = "",
    generation_header: str = "",
):
    """Run WIM with RL-enhanced margin generation.
    
    Args:
        model_id: The ID of the model to use.
        input_document: The input document content.
        query: The user query.
        use_rl_generator: Whether to use the RL-based margin generator.
        train_rl_generator: Whether to train the RL generator.
        num_episodes: Number of episodes for RL training.
        output_model_dir: Directory to save the trained model.
        attn_implementation: Attention implementation to use.
        dtype: Data type for model weights.
        min_tokens_segment: Minimum number of tokens per segment.
        max_new_tokens_extractive_summary: Maximum number of tokens to generate for margin.
        max_new_tokens_final_answer: Maximum number of tokens to generate for final answer.
        max_new_tokens_classification: Maximum number of tokens to generate for classification.
        do_sample: Whether to use sampling for generation.
        top_p: Top-p sampling parameter.
        temperature: Temperature for generation.
        early_stopping: Whether to use early stopping.
        print_step_summary: Whether to print a summary for each step.
        user_header: User header for the model.
        generation_header: Generation header for the model.
        
    Returns:
        final_answer: The generated answer.
    """
    
    
    # Define model dtype
    model_dtype = torch.float32
    if dtype == "float16":
        model_dtype = torch.float16
    elif dtype == "float32":
        model_dtype = torch.float32
    elif dtype == "bfloat16":
        model_dtype = torch.bfloat16
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer_rl = AutoTokenizer.from_pretrained(model_id_rl)
    tokenizer_rl.pad_token = tokenizer_rl.eos_token
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,  # could also try bfloat16
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4"  # best performance for LLaMA-like models
    # )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=model_dtype,
    ).eval()

    model_rl = AutoModelForCausalLM.from_pretrained(
        model_id_rl,
        device_map="auto",
        attn_implementation=attn_implementation,
        torch_dtype=model_dtype,
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load templates
    template_extractive_summary = """
    ```
    Given the above context, extract all information relevant to the query: "{query}". If the context is not relevant to the query, answer "I don't know."
    {generation_header}
    """.strip()
    
    template_classification = """
    {user_header}
    I asked an LLM assistant whether a piece of document is related to the query: "{query}". This is its answer: 
    ```text
    {answer}
    ```
    Should I save it for later? 
    Here are rules:
    - Answer YES if the answer contains information about the query. 
    - Answer NO if the answer says the piece isn't related to the query.

    Provide the answer in the format: <YES/NO>#<Explanation>. 
    Here is are example answers:

    YES#Yes, the information contains an excerpt from a book that is related to the question.
    NO#No, the LLM assistant concluded the information isn't relevant.

    Don't add any other comments, all your remarks should be included in the "Explanation" section.
    {generation_header}
    """.strip()
    
    template_system_message = """
    {user_header}
    ```
    """.strip()
    
    template_final_answer = """
    ```
    {margins}
    {query}
    {generation_header}
    """.strip()
    
    # Replace special tokens
    special_tokens = {
        "{user_header}": "<|start_header_id|>user<|end_header_id|>\n\n",
        "{generation_header}": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    }
    
    for token, replacement in special_tokens.items():
        template_extractive_summary = template_extractive_summary.replace(token, replacement)
        template_classification = template_classification.replace(token, replacement)
        template_system_message = template_system_message.replace(token, replacement)
        template_final_answer = template_final_answer.replace(token, replacement)
    
    # Create WIM inference
    # wim_inference = WIMInference(model, tokenizer)


    # Create WIM RL inference for training RL model
    wim_inference = WIMRLInference(model, tokenizer)
    
    # Create RL margin generator if needed
    rl_margin_generator = None
    if use_rl_generator or train_rl_generator:
        print("Use RL-based margin generator")
        rl_margin_generator = RLMarginGenerator(
            model_id=model_id_rl,
            reward_model_id=model_id_rl,  # Use same model for rewards
            device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available else "cpu",
        )
    else:
        print("Use standard WIM approach")
    # Train RL generator if requested
    if train_rl_generator and rl_margin_generator is not None:
        print("Training RL margin generator...")
        segments = wim_inference._chunk_text_to_segments(input_document, min_tokens_segment)
        print(segments)
        rl_margin_generator.train_rl_margin_generator(
            segments=segments,
            query=query,
            extractive_summary_prompt=template_extractive_summary,
            classification_prompt=template_classification,
            num_episodes=num_episodes,
            max_new_tokens=max_new_tokens_extractive_summary,
        )
        
        # Save the trained model if requested
        if output_model_dir is not None:
            print(f"Saving trained model to {output_model_dir}...")
            rl_margin_generator.save_model(output_model_dir)
    
    # Create WIM RL inference
    wim_rl_inference = WIMRLInference(model, tokenizer, rl_margin_generator)
    
    # Process with WIM RL
    final_answer, positive_margins = wim_rl_inference.process_with_rl_margins(
        context=input_document,
        query=query,
        system_message=template_system_message,
        extractive_summary_prompt=template_extractive_summary,
        classification_prompt=template_classification,
        final_answer_prompt=template_final_answer,
        min_tokens_segment=min_tokens_segment,
        max_new_tokens_extractive_summary=max_new_tokens_extractive_summary,
        max_new_tokens_final_answer = max_new_tokens_final_answer,
        max_new_tokens_classification = max_new_tokens_classification,
        do_sample = do_sample,
        top_p = top_p,
        temperature = temperature,
        early_stopping = early_stopping,
        use_rl_generator = use_rl_generator,
        print_step_summary = True
    )

    return final_answer, positive_margins
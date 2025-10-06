import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache, Cache
from dataclasses import dataclass
from typing import List, Dict, Any, Callable

def _sample_top_p(logits, top_p=0.9):
    logits = logits - torch.max(logits)
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    eps = 1e-10
    probs = torch.clamp(probs, min=eps)
    probs = probs / probs.sum()  
    
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    probs = probs.masked_fill(indices_to_remove, 0.0)
    
    probs = probs / (probs.sum() + eps)
    
    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
        probs = torch.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
        probs = torch.clamp(probs, min=eps)
        probs = probs / probs.sum()  
    
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token

class WIMInference:

    def __init__(
        self, model, tokenizer
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.wim_kv_cache = DynamicCache()
        self.classifier_kv_cache = DynamicCache()

    def _prefill_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_positions: torch.Tensor,
        kv_cache: Cache,
    ):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cache_position=cache_positions,
                use_cache=True,
                past_key_values=kv_cache,
            )
        return outputs

    def shrink_kv_cache_from_end(self, new_size: int, kv_cache: Cache):

        def resize_tensor_list(token_list):
            for layer_idx in range(len(token_list)):
                token_list[layer_idx] = token_list[layer_idx][:, :, :new_size, :]

        resize_tensor_list(kv_cache.key_cache)
        resize_tensor_list(kv_cache.value_cache)
        kv_cache._seen_tokens = new_size

    def generate_text_with_kv_cache(
        self,
        max_new_tokens: int,
        previous_logits: torch.Tensor,
        do_sample: bool,
        top_p: float,
        temperature: float,
        early_stopping: bool,    
        kv_cache: Cache,
    ) -> str:
        generated_tokens = []

        # This is needed to create the cache_position tensor
        next_token_pos = kv_cache.get_seq_length()

        # Use the logits from the prefilling to generate the first token
        logits = previous_logits

        for _ in range(max_new_tokens):
            # Select the last token from the logits
            next_token_logits = logits[:, -1, :]
            if do_sample:
                # Divide the logits by the temperature
                next_token_logits = next_token_logits / temperature
                # Apply the softmax
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                
                # Check for invalid values (moved here after next_token_probs is defined)
                if torch.isnan(next_token_probs).any() or torch.isinf(next_token_probs).any():
                    print("Invalid probabilities detected!")
                    print(f"Min prob: {next_token_probs.min()}, Max prob: {next_token_probs.max()}")
                    print(f"Contains NaN: {torch.isnan(next_token_probs).any()}")
                    print(f"Contains Inf: {torch.isinf(next_token_probs).any()}")
                    
                next_token = _sample_top_p(next_token_logits, top_p)  # Note: passing logits, not probs
            else:
                # Select the token with the highest probability
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            # Remove the batch dimension
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            # Stop if we reached the EOS token
            if next_token.item() == self.tokenizer.eos_token_id and early_stopping:
                break
            # Use the generated token as input for the next step
            generation_input_ids = next_token.unsqueeze(-1)
            kv_cache_seq_len = kv_cache.get_seq_length()
            generation_attention_mask = torch.ones(
                (1, kv_cache_seq_len + 1), device=next_token.device, dtype=torch.long
            )
            generation_cache_position = torch.tensor(
                [next_token_pos], device=next_token.device
            )

            with torch.no_grad():
                # Get the model outputs
                outputs = self.model(
                    input_ids=generation_input_ids,
                    attention_mask=generation_attention_mask,
                    cache_position=generation_cache_position,
                    use_cache=True,
                    past_key_values=kv_cache,
                )
            logits = outputs.logits
            next_token_pos += 1

        generated_tokens = torch.cat(generated_tokens, dim=-1)
        # Decode the generated tokens
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded

    def prefill_text_with_kv_cache(self, text: str, kv_cache: Cache):
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        seq_len = input_ids.size(1)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # If we have a KV-Cache, we need to extend the attention mask to account for tokens already in the KV-Cache
        if kv_cache.get_seq_length() > 0:
            kv_cache_seq_len = kv_cache.get_seq_length()
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        kv_cache_seq_len,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )

        # Generate the cache positions for the tokens to be prefilled
        cache_positions = torch.arange(
            kv_cache.get_seq_length(), kv_cache.get_seq_length() + seq_len
        ).to(self.model.device)
        outputs = self._prefill_tokens(input_ids, attention_mask, cache_positions, kv_cache)
        return kv_cache.get_seq_length(), seq_len, outputs
    

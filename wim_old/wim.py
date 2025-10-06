import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache, Cache
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
# import flash_attn


def _sample_top_p(probs: torch.Tensor, p: float):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
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
        # cache_positions: torch.Tensor,
        kv_cache: Cache,
    ):
        '''
        prefills the model’s KV cache by running the model on a chunk of input tokens
        '''
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # cache_position=cache_positions,
                use_cache=True,
                past_key_values=kv_cache,
            )
        return outputs

    def shrink_kv_cache_from_end(self, new_size: int, kv_cache: Cache):
        '''
        Shrink the KV-Cache from the end to the new size.
        so as to keep only the most recent tokens manage memory
        '''

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

        next_token_pos = kv_cache.get_seq_length()

        logits = previous_logits

        for _ in range(max_new_tokens):
            next_token_logits = logits[:, -1, :]
            if do_sample:
                next_token_logits = next_token_logits / temperature
                next_token_probs = torch.nn.functional.softmax(
                    next_token_logits, dim=-1
                )
                next_token = _sample_top_p(next_token_probs, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            assert next_token.size() == (1, 1)
            next_token = next_token.squeeze(0)
            generated_tokens.append(next_token)
            if next_token.item() == self.tokenizer.eos_token_id and early_stopping:
                break
            generation_input_ids = next_token.unsqueeze(-1)
            kv_cache_seq_len = kv_cache.get_seq_length()
            generation_attention_mask = torch.ones(
                (1, kv_cache_seq_len + 1), device=next_token.device, dtype=torch.long
            )
            generation_cache_position = torch.tensor(
                [next_token_pos], device=next_token.device
            )

            with torch.no_grad():
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
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return decoded

    def prefill_text_with_kv_cache(self, text: str, kv_cache: Cache):
        print(f"Received text: {text}")

        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        print(f"Input IDs shape: {input_ids.shape}")

        seq_len = input_ids.size(1)
        print(f"Sequence length: {seq_len}")
        attention_mask = inputs["attention_mask"].to(self.model.device)
        print(f"Initial attention_mask shape: {attention_mask.shape}")

        if kv_cache.get_seq_length() > 0:
            kv_cache_seq_len = kv_cache.get_seq_length()
            print(f"KV Cache sequence length: {kv_cache_seq_len}")
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
            print(f"Extended attention_mask shape: {attention_mask.shape}")
        else:
            print("KV Cache is empty; not extending attention_mask.")

        start_pos = kv_cache.get_seq_length()
        end_pos = start_pos + seq_len
        cache_positions = torch.arange(start_pos, end_pos).to(self.model.device)
        print(f"Cache positions: {cache_positions}")

        outputs = self._prefill_tokens(input_ids, attention_mask, kv_cache)
        print("Obtained outputs from _prefill_tokens method.")

        print(f"KV Cache current sequence length: {kv_cache.get_seq_length()}")
        print(f"New tokens count (seq_len): {seq_len}")

        return kv_cache.get_seq_length(), seq_len, outputs
    

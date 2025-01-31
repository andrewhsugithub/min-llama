import token
import torch

from utils.find_device import get_available_device
from transformers import AutoTokenizer, set_seed
from llama_model import LlamaForCausalLM
from tokenizer import ChatFormat, Tokenizer


class Llama:
    @staticmethod
    def build(hf_model_id="meta-llama/Llama-3.2-1B-Instruct", seed=1, token=None):
        assert token is not None, "hugging face's access token is required"

        # set seed
        torch.manual_seed(seed)
        set_seed(seed)  # for hf transformers

        # set torch defaults: device, dtype
        device = get_available_device()

        if torch.cuda.is_bf16_supported():
            torch.set_default_dtype(torch.bfloat16)
        else:
            torch.set_default_dtype(torch.float16)

        torch.set_default_device(device)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, token=token)

        # load model
        model = LlamaForCausalLM.load_pretrained_weights(
            model_type=hf_model_id, token=token
        )
        model.to(device)
        model.params.device = device

        return Llama(model, tokenizer, device)

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(self, prompts, temperature=0.6, top_k=50, max_new_tokens=250):
        params = self.model.params
        prompt_tokens = [self.tokenizer(p).input_ids for p in prompts]
        print("input length:", [len(p) for p in prompt_tokens])
        batch_size = len(prompt_tokens)
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        assert max_prompt_len <= params.context_length
        total_len = min(params.context_length, max_prompt_len + max_new_tokens)

        pad_id = -1
        tokens = torch.full(
            (batch_size, total_len), pad_id, dtype=torch.long, device=self.device
        )  # (batch_size, seq_len) of pad_ids
        for i, batch in enumerate(prompt_tokens):
            tokens[i, : len(batch)] = torch.tensor(
                batch, dtype=torch.long, device=self.device
            )

        # Mask: True for valid tokens, False for padding
        input_text_mask = tokens != pad_id  # (batch_size, seq_len)

        # Track which sequences have finished (hit eos).
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=tokens.device
        )  # (batch_size,)
        prev_pos = 0

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            logits = logits[:, -1, :]  # -> (batch_size, vocab_size)

            if top_k is not None:
                top_vals, _ = torch.topk(logits, k=top_k, dim=-1)
                # minimum value in the top_k range, for each batch element
                min_topk = top_vals[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_topk,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits,
                )

            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)
                next_token_ids = torch.multinomial(
                    probs, num_samples=1
                )  # (batch_size, 1)
            else:
                next_token_ids = torch.argmax(
                    logits, dim=-1, keepdim=True
                )  # (batch_size, 1)

            next_token_ids = next_token_ids.reshape(
                -1
            )  # (batch_size,1) -> (batch_size,)
            # only replace token if prompt has already been generated, if True(haven't generate new token), keep original token, else replace with new token
            next_token_ids = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token_ids
            )  # (batch_size,)

            # If a sequence is already finished, we keep forcing it to stay at EOS so it doesn't keep generating tokens.
            next_token_ids[finished] = self.tokenizer.eos_token_id

            tokens[:, cur_pos] = next_token_ids

            # Check which sequences just finished
            newly_finished = next_token_ids == self.tokenizer.eos_token_id
            # falsy finish if prompt has not been generated, if True(haven't generate new token), keep original finish, else replace with new finish
            newly_finished = torch.where(
                input_text_mask[:, cur_pos], finished, newly_finished
            )
            finished = finished | newly_finished

            prev_pos = cur_pos
            # If all sequences in the batch are finished, we can stop early.
            if torch.all(finished):
                break

        # remove pad tokens
        tokens = tokens[:, : cur_pos + 1]

        outputs = []
        for i in range(batch_size):
            text = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            outputs.append(text)

        return outputs

    def clean_text(self, text, header_end="assistant<|end_header_id|>\n\n"):
        return text.split(header_end)[-1]

    def completion(self, prompts, temperature=0.6, top_k=50, max_new_tokens=250):
        assert (
            len(prompts) <= self.model.params.batch_size
        ), f"Batch size is too large! Expected at most {self.model.params.batch_size}, but got {len(prompts)}."

        formatted_prompts = self.tokenizer.apply_chat_template(
            prompts, tokenize=False, add_generation_prompt=True
        )

        responses = self.generate(
            prompts=formatted_prompts,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        )

        return [self.clean_text(r, header_end="assistant\n\n") for r in responses]

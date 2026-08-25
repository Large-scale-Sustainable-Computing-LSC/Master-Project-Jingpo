#!/usr/bin/env python3
"""Compare normal Qwen3.5 decoding with dynamic forward replay.

This script defaults to Qwen/Qwen3.5-4B so it can fit smaller GPUs, but the
same dynamic forward backend is shared with Qwen/Qwen3.5-9B.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from dynamic_forward import dynamic_forward


MODEL_ID = "Qwen/Qwen3.5-4B"
PROMPTS = [
    "What is the capital city of France?",
    "What gas do plants absorb from the atmosphere during photosynthesis?",
    "In one sentence, why does the Moon appear to change shape over a month?",
]


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg != "auto":
        raise ValueError(f"Unsupported dtype: {dtype_arg}")

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def auth_token_arg(cli_token: Optional[str]) -> Any:
    return cli_token or os.environ.get("HF_TOKEN") or True


def load_qwen35_model(model_id: str, token: Any, dtype: torch.dtype, local_files_only: bool):
    from transformers import AutoModelForMultimodalLM

    return AutoModelForMultimodalLM.from_pretrained(
        model_id,
        token=token,
        dtype=dtype,
        local_files_only=local_files_only,
    )


def text_decoder_layers(model: torch.nn.Module):
    decoder = text_decoder_model(model)
    return decoder.layers


def text_decoder_model(model: torch.nn.Module):
    decoder = getattr(model, "model", model)
    if hasattr(decoder, "language_model"):
        return decoder.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(decoder, "layers"):
        return decoder
    raise TypeError("Cannot find Qwen text decoder layers.")


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def decode_new_tokens(tokenizer: AutoTokenizer, token_tensors: List[torch.Tensor]) -> str:
    token_ids = torch.cat(token_tensors, dim=-1).squeeze(0).tolist()
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def format_prompt(tokenizer: AutoTokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extend_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    next_mask = torch.ones(
        (attention_mask.shape[0], 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat([attention_mask, next_mask], dim=-1)


@torch.inference_mode()
def run_baseline_one(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    formatted_prompt = format_prompt(tokenizer, prompt)
    encoded = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_len = input_ids.shape[-1]
    generated_new_token_ids: List[torch.Tensor] = []

    past_key_values = None
    step_input_ids = input_ids
    step_attention_mask = attention_mask
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

    for step in range(max_new_tokens):
        position_ids = cache_position.unsqueeze(0)
        outputs = model(
            input_ids=step_input_ids,
            attention_mask=step_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

        next_token = greedy_next_token(outputs.logits)
        generated_new_token_ids.append(next_token.detach().cpu())

        past_key_values = outputs.past_key_values
        step_input_ids = next_token
        step_attention_mask = extend_attention_mask(step_attention_mask)
        cache_position = torch.tensor([prompt_len + step], device=device, dtype=torch.long)

    return {
        "prompt": prompt,
        "formatted_prompt": formatted_prompt,
        "prompt_input_ids": input_ids.detach().cpu().squeeze(0),
        "prompt_attention_mask": attention_mask.detach().cpu().squeeze(0),
        "prompt_len": prompt_len,
        "baseline_new_token_ids": torch.cat(generated_new_token_ids, dim=-1).squeeze(0).tolist(),
        "baseline_text": decode_new_tokens(tokenizer, generated_new_token_ids),
    }


@torch.inference_mode()
def run_split_one(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    record: Dict[str, Any],
    *,
    k: int,
    max_new_tokens: int,
    device: torch.device,
) -> Dict[str, Any]:
    decoder = text_decoder_model(model)
    prefix_cache = None
    suffix_cache = None
    step_input_ids = record["prompt_input_ids"].to(device).unsqueeze(0)
    step_attention_mask = record["prompt_attention_mask"].to(device).unsqueeze(0)
    prompt_len = int(record["prompt_len"])
    cache_position = torch.arange(prompt_len, device=device, dtype=torch.long)

    generated_new_token_ids: List[torch.Tensor] = []
    mismatches: List[Dict[str, int]] = []

    for step in range(max_new_tokens):
        position_ids = cache_position.unsqueeze(0)
        input_embeds = decoder.embed_tokens(step_input_ids)

        prefix_outputs = dynamic_forward(
            model,
            input_embeds,
            start_layer=0,
            end_layer=k,
            attention_mask=step_attention_mask,
            position_ids=position_ids,
            past_key_values=prefix_cache,
            use_cache=True,
            cache_position=cache_position,
            apply_final_norm=False,
        )
        prefix_cache = prefix_outputs.past_key_values

        suffix_outputs = dynamic_forward(
            model,
            prefix_outputs.last_hidden_state,
            start_layer=k,
            attention_mask=step_attention_mask,
            position_ids=position_ids,
            past_key_values=suffix_cache,
            use_cache=True,
            cache_position=cache_position,
        )
        suffix_cache = suffix_outputs.past_key_values

        logits = model.lm_head(suffix_outputs.last_hidden_state[:, -1:, :])
        next_token = greedy_next_token(logits)
        actual = int(next_token.item())
        expected = int(record["baseline_new_token_ids"][step])
        if actual != expected:
            mismatches.append({"step": step, "baseline": expected, "split": actual})

        generated_new_token_ids.append(next_token.detach().cpu())
        step_input_ids = next_token
        step_attention_mask = extend_attention_mask(step_attention_mask)
        cache_position = torch.tensor([prompt_len + step], device=device, dtype=torch.long)

    return {
        "split_new_token_ids": torch.cat(generated_new_token_ids, dim=-1).squeeze(0).tolist(),
        "split_text": decode_new_tokens(tokenizer, generated_new_token_ids),
        "mismatches": mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=2, help="Split layer. k=2 means resume from layer2.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--hf-token", default=None, help="Optional HF token. Defaults to HF_TOKEN/login token.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    token = auth_token_arg(args.hf_token)

    eprint(f"Loading tokenizer/model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=token,
        local_files_only=args.local_files_only,
    )
    model = load_qwen35_model(args.model_id, token, dtype, args.local_files_only)
    model.to(device)
    model.eval()

    num_layers = len(text_decoder_layers(model))
    if not 0 <= args.k < num_layers:
        raise ValueError(f"--k must be in [0, {num_layers - 1}], got {args.k}")

    records = []
    for index, prompt in enumerate(PROMPTS):
        eprint(f"Baseline example {index}:")
        record = run_baseline_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        records.append(record)

    split_results = []
    for index, record in enumerate(records):
        eprint(f"Split-forward example {index}, cut layer {args.k}")
        result = run_split_one(
            model,
            tokenizer,
            record,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        if result["mismatches"]:
            first = result["mismatches"][0]
            eprint(
                "Token mismatch: "
                f"example={index} step={first['step']} "
                f"baseline={first['baseline']} split={first['split']}"
            )
        split_results.append(result)

    for index, record in enumerate(records):
        print(f"[baseline {index}] {record['baseline_text']}")
    for index, result in enumerate(split_results):
        print(f"[split k={args.k} {index}] {result['split_text']}")

    if all(not result["mismatches"] for result in split_results):
            eprint("All matched.")


if __name__ == "__main__":
    main()

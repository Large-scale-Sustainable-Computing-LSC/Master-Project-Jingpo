import argparse
import os
import time

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM


MODEL_P = "8B"
MODEL_PATH = f"../../../Llama-3.2-{MODEL_P}"
MODEL_PATH = f"meta-llama/Llama-3.2-{MODEL_P}"
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_MAX_INPUT = 240
DEFAULT_LAYER = 2
DEFAULT_GROUP = 2
DEFAULT_DELAY_SECONDS = 0.4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-input", type=int, default=DEFAULT_MAX_INPUT)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP)
    parser.add_argument("--delay-seconds", type=float, default=DEFAULT_DELAY_SECONDS)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def validate_args(args):
    if args.group_size < 2 or args.group_size > 5:
        raise ValueError(f"--group-size must be in [2, 5], got {args.group_size}")
    if args.delay_seconds < 0:
        raise ValueError(f"--delay-seconds must be >= 0, got {args.delay_seconds}")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_model_dtype(device: torch.device):
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def pick_default_data_dir(model_p: str):
    local_base = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    return os.path.join(local_base, f"hs_test_{model_p}")


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def read_hidden(run_id: int, step: int, layer_idx: int, device: torch.device, data_dir: str):
    hs_path = f"{data_dir}/{run_id}/hs_s{int(step):03d}.safetensors"
    hs_dict = load_file(hs_path)

    if "hs" in hs_dict:
        with safe_open(hs_path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        saved_layer = meta.get("save_layer_idx")
        if saved_layer is not None and int(saved_layer) != int(layer_idx):
            raise ValueError(
                f"saved hidden layer is {saved_layer}, but bench is configured with --layer {layer_idx}"
            )
        return hs_dict["hs"].to(device)

    return hs_dict[f"hs_layer_{int(layer_idx):03d}"].to(device)


def ensure_data_dir(data_dir: str, max_input: int):
    missing = []
    for run_id in range(max_input):
        if not os.path.exists(f"{data_dir}/{run_id}/masks.safetensors"):
            missing.append(run_id)
    if missing:
        raise FileNotFoundError(
            f"missing saved hidden-state inputs in {data_dir}: {missing}"
        )


def load_sample_state(run_id: int, device: torch.device, data_dir: str):
    data_m = load_file(f"{data_dir}/{run_id}/masks.safetensors")
    return {
        "run_id": run_id,
        "step": 0,
        "past_key_values": None,
        "attention_mask": data_m["attention_mask"].to(device),
        "position_ids": data_m["position_ids"].to(device),
        "generated_text": data_m["input_ids"].to(device),
        "finished": False,
        # The next time this task is allowed to start its next token.
        # First token is exempt from the synthetic delay, so start as immediately ready.
        "next_ready_time": 0.0,
    }


def wait_until(ts: float):
    now = time.perf_counter()
    if ts > now:
        time.sleep(ts - now)


def decode_one_token(state: dict, model_full, device: torch.device, layer: int, data_dir: str):
    start_hidden = read_hidden(
        state["run_id"], state["step"], layer, device, data_dir
    )

    outputs = model_full.model(
        inputs_embeds=start_hidden,
        attention_mask=state["attention_mask"],
        position_ids=state["position_ids"],
        past_key_values=state["past_key_values"],
        use_cache=True,
    )

    logits = model_full.lm_head(outputs.last_hidden_state)
    next_token = logits[:, -1:, :].argmax(dim=-1).detach()

    state["past_key_values"] = outputs.past_key_values
    state["generated_text"] = torch.cat([state["generated_text"], next_token], dim=-1)
    state["attention_mask"] = torch.cat(
        [state["attention_mask"], torch.ones_like(next_token)], dim=-1
    )
    current_length = state["generated_text"].shape[1]
    state["position_ids"] = torch.tensor([[current_length - 1]], device=device)

    state["step"] += 1
    if state["step"] >= args.max_new_tokens:
        state["finished"] = True


args = parse_args()
validate_args(args)
if args.data_dir is None:
    args.data_dir = pick_default_data_dir(MODEL_P)

device = pick_device()
model_dtype = pick_model_dtype(device)
ensure_data_dir(args.data_dir, args.max_input)

print("Using device:", device)
print("Using dtype:", model_dtype)
print("Using hidden dir:", args.data_dir)
print("mode: int_f_n")
print("group_size:", args.group_size)
print("delay_seconds:", args.delay_seconds)

config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model_full = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=model_dtype,
).to(device)
model_full.model.layers = model_full.model.layers[args.layer:]
model_full.eval()

global_start = time.perf_counter()

with torch.no_grad():
    for base_idx in range(0, args.max_input, args.group_size):
        run_ids = list(range(base_idx, min(base_idx + args.group_size, args.max_input)))
        states = [load_sample_state(run_id, device, args.data_dir) for run_id in run_ids]

        while any(not state["finished"] for state in states):
            made_progress = False

            for state in states:
                if state["finished"]:
                    continue

                # A task can only start its next token after its own cooldown expires.
                wait_until(state["next_ready_time"])

                token_end_before = time.perf_counter()
                decode_one_token(state, model_full, device, args.layer, args.data_dir)
                sync_device(device)
                token_end_after = time.perf_counter()

                # First token is exempt from delay; after that, every token completion
                # imposes the same cooldown before this same task may compute its next token.
                if not state["finished"]:
                    state["next_ready_time"] = token_end_after + args.delay_seconds
                else:
                    state["next_ready_time"] = token_end_after

                made_progress = True

            if not made_progress:
                next_ready = min(
                    state["next_ready_time"] for state in states if not state["finished"]
                )
                wait_until(next_ready)

total_time_seconds = time.perf_counter() - global_start

print(f"total_time_seconds={total_time_seconds:.6f}")

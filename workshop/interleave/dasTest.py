from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer, LlamaModel
from datasets import load_dataset, get_dataset_config_names
import torch
import os
import time
from safetensors.torch import save_file

import threading  
import psutil     
MODEL_P = "3B"
MODEL_PATH = f"../../../Llama-3.2-{MODEL_P}"
MODEL_PATH = f"meta-llama/Llama-3.2-{MODEL_P}"
MAX_NEW_TOKENS = 32
SAVE_HIDDEN_LAYERS = [1, 2, 3, 4, 5, 6]

TEST_SET = f"meta-llama/Llama-3.2-{MODEL_P}-evals"
DATA_NAME = f"Llama-3.2-{MODEL_P}-evals__drop__details"
DATA_TO_START = 0
MAX_QUESTIONS = 240
STEP_SIZE = 13
RANDOM_DATA = False


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


def pick_hs_base_dir(model_p: str):
    local_base = (
        os.environ.get("SLURM_TMPDIR")
        or os.environ.get("TMPDIR")
        or "/tmp"
    )
    return os.path.join(local_base, f"hs_test_{model_p}")

def monitor_resources(filename):
    """资源监控函数，每秒记录一次数据"""
    with open(filename, 'w') as f:
        f.write("timestamp,memory_usage_mb,gpu_memory_mb\n")  # 表头
        while True:
            # 获取内存使用（单位：MB）
            memory_mb = psutil.virtual_memory().used / (1024 ** 2)
            # 获取GPU显存使用（单位：MB）
            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            else:
                gpu_mem_mb = 0
            # 写入数据
            timestamp = time.time()
            f.write(f"{timestamp},{memory_mb:.2f},{gpu_mem_mb:.2f}\n")
            f.flush()  # 确保实时写入
            time.sleep(1)

def save_hidden_states_safetensors(
    outputs,
    out_dir: str,
    question_idx: int,
    step: int,
    is_prefill: bool,
    save_layer_indices,
    #cast_dtype: torch.dtype | None = torch.float16,  # 想更省空间用 fp16；想保真设 None（不cast）
) -> str:
    """
    保存多个指定层的 hidden state。
    文件内 keys:
      hs_layer_001, hs_layer_002, ...
    metadata:
      question_idx, step, is_prefill, save_layer_indices
    """
    os.makedirs(out_dir, exist_ok=True)

    hs = outputs.hidden_states
    if hs is None:
        raise ValueError("outputs.hidden_states is None. 你需要在 forward 时加 output_hidden_states=True")

    tensor_dict = {}
    normalized_indices = []
    for layer_idx in save_layer_indices:
        if layer_idx < 0 or layer_idx >= len(hs):
            raise ValueError(
                f"layer_idx={layer_idx} is out of range for hidden_states of length {len(hs)}"
            )
        x = hs[layer_idx].detach().to("cpu").contiguous()
        tensor_dict[f"hs_layer_{layer_idx:03d}"] = x
        normalized_indices.append(int(layer_idx))

    meta = {
        "question_idx": str(int(question_idx)),
        "step": str(int(step)),
        "is_prefill": "1" if is_prefill else "0",
        "save_layer_indices": ",".join(str(i) for i in normalized_indices),
    }

    fname = os.path.join(
        out_dir,
        f"hs_s{int(step):03d}.safetensors"
    )
    save_file(tensor_dict, fname, metadata=meta)
    return fname


data = load_dataset(TEST_SET,
        name = DATA_NAME,
        split="latest"
)

if DATA_TO_START >= len(data):
    print(f"Invalid DATA_TO_START {DATA_TO_START}, will start from 0.")
    DATA_TO_START = 0

# 提取所有问题

device = pick_device()
model_dtype = pick_model_dtype(device)
hs_base_dir = pick_hs_base_dir(MODEL_P)

print("Using device:", device)
print("Using dtype:", model_dtype)
print("Using hidden dir:", hs_base_dir)

monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(f"hidden_cap.csv",),  
        daemon=True  # 设为守护线程，主线程退出时自动终止
    )

#monitor_thread.start()


config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

model_full = LlamaForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=model_dtype,
).to(device)
#model_full.model.layers = model_full.model.layers[:20]
model_full.eval()
model_full.to(device)


#model_full.model.norm = torch.nn.Identity()  # 跳过norm层


model_full.eval()

#que_num_count = 0
#while 0:

questions = data["input_final_prompts"]
#questions = data["input_question"]

inf_start_time = time.time()
input_counter = 0

with torch.no_grad():   #禁用梯度计算
    for que_num_count in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):

        #gen_que = data[que_num_count]

        question  = questions[que_num_count]

        print(f"\n\nInferencing {que_num_count}:", end = "")

        past_key_values_full = None

        prefill_start_time = time.time()
        inputs = tokenizer(question, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)              # save
        position_ids = attention_mask.long().cumsum(dim=1) - 1         # save


        to_save = {
            "input_ids": input_ids.detach().to("cpu"),
            "attention_mask": attention_mask.detach().to("cpu"),
            "position_ids": position_ids.detach().to("cpu"),
        }

        sample_dir = os.path.join(hs_base_dir, str(input_counter))
        os.makedirs(sample_dir, exist_ok=True)
        save_path_m = os.path.join(sample_dir, "masks.safetensors")

        save_file(to_save, save_path_m)

        generated_text = input_ids

        for count in range(MAX_NEW_TOKENS):

            #print(count)
            is_prefill = (past_key_values_full is None)

            outputs = model_full.model(
                input_ids=generated_text[:, -1:] if past_key_values_full else generated_text,

                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_full,
                use_cache=True,
                output_hidden_states=True
            )
    
            #torch.mps.synchronize()

            _ = save_hidden_states_safetensors(
                outputs,
                out_dir=sample_dir,
                question_idx=input_counter,
                step=count,
                is_prefill=is_prefill,
                save_layer_indices=SAVE_HIDDEN_LAYERS,
                #cast_dtype=torch.float16,   # 或 None
            )

            last_hidden_state = outputs.last_hidden_state
            

            logits = model_full.lm_head(last_hidden_state)
            past_key_values_full = outputs.past_key_values

            #last_token_hidden = last_hidden_state[:, -1:, :]
            #last_token_logits = model_full.lm_head(last_token_hidden)

            # pick next token
            next_token = logits[:, -1:, :].argmax(dim=-1).detach()    #分离张量
            generated_text = torch.cat([generated_text, next_token], dim=-1)


            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            #print(attention_mask)
            current_length = generated_text.shape[1]
            position_ids = torch.tensor([[current_length - 1]], device = device)
            

        input_counter += 1
        output_string = tokenizer.decode(generated_text[0][input_ids.size(1):])
        print(output_string)


inf_end_time = time.time()
inf_time_total = inf_end_time - inf_start_time 


print("  ========  Finished (fullSim)  ========  ")
print(f"Data used: {DATA_NAME}")
print(f"Device used: {device}")
print(f"From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.\n")

print(f"Time used: \n{inf_time_total} seconds.\n")

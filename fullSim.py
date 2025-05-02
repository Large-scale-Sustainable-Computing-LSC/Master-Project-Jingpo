from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
import torch
import gc
import time

MODEL_PATH = "./Llama-3.2-1B"
LAYER_P1 = 8
LAYER_P2 = 8
QUESTION = "Where is the capital of USA?"
#QUESTION = "macOS with m1 chip and 16G memory VS linux with AMD 5700XT GPU (8G VRAM), which one could be better for LLM deployment?"
MAX_NEW_TOKENS = 32

DATA_NAME = "Llama-3.2-1B-evals__drop__details"
DATA_TO_START = 3
MAX_QUESTIONS = 128
STEP_SIZE = 37
RANDOM_DATA = False



data = load_dataset("meta-llama/Llama-3.2-1B-evals",
        #name="Llama-3.2-1B-evals__agieval_english__details",
        name = DATA_NAME,
        split="latest"
)

if DATA_TO_START >= len(data):
    print(f"Invalid DATA_TO_START {DATA_TO_START}, will start from 0.")
    DATA_TO_START = 0

# 提取所有问题

try:
    device = torch.device("mps")
except Exception as e:
    print(e)

print("Using device:", device)


config = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

model_full = LlamaForCausalLM.from_pretrained(MODEL_PATH).to(device)


#model_full.model.norm = torch.nn.Identity()  # 跳过norm层


model_full.eval()

#que_num_count = 0
#while 0:

inf_start_time = time.time()

prefill_time_total = 0
first_token_total = 0
following_token_total = 0

with torch.no_grad():   #禁用梯度计算
    for que_num_count in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):

        gen_que = data[que_num_count]
        
        question  = gen_que["input_question"]
        #print(question)

        print(f"Inferencing {que_num_count}")

        if 'past_key_values_full' in locals():   #清理缓存
            del past_key_values_full
        torch.cuda.empty_cache()
        gc.collect()
        past_key_values_full = None

        prefill_start_time = time.time()
        inputs = tokenizer(question, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        position_ids = attention_mask.long().cumsum(dim=1) - 1

        generated_text = input_ids
        
        prefill_end_time = time.time()
        prefill_time = prefill_end_time - prefill_start_time
        prefill_time_total += prefill_time

        for count in range(MAX_NEW_TOKENS):
            
            start_token_time = time.time()

            #print(count)
            outputs = model_full(
                input_ids=generated_text[:, -1:] if past_key_values_full else generated_text,

                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_full,
                use_cache=True,
            )

            end_token_time = time.time()
            single_token_time = end_token_time - start_token_time
            if count == 0:
                first_token_total += single_token_time
            else:
                following_token_total += single_token_time
        

            logits = outputs.logits
            past_key_values_full = outputs.past_key_values

            # pick next token
            next_token = logits[:, -1:, :].argmax(dim=-1).detach()    #分离张量
            generated_text = torch.cat([generated_text, next_token], dim=-1)


            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            #print(attention_mask)
            current_length = generated_text.shape[1]
            position_ids = torch.tensor([[current_length - 1]], device = device)


        output_string = tokenizer.decode(generated_text[0][input_ids.size(1):])
        print("Output with `.forward()`:\n" + output_string)

inf_end_time = time.time()
inf_time_total = inf_end_time - inf_start_time


firstSpeed0 = first_token_total / MAX_QUESTIONS
prefillSpeed0 = prefill_time_total / MAX_QUESTIONS

overallSpeed = (MAX_QUESTIONS * MAX_NEW_TOKENS) / inf_time_total
prefillSpeed = MAX_QUESTIONS / prefill_time_total
firstSpeed = MAX_QUESTIONS / first_token_total
followingSpeed = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_total

print("  ========  Finished (fullSim)  ========  ")
print(f"Data used: {DATA_NAME}")
print(f"Device used: {device}")
print(f"From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.")
print(f"Time used: {inf_time_total} seconds.")
print(f"Overall speed: {overallSpeed} tokens/second.")
print(f"Prefill speed: {prefillSpeed} questions/second.     -- {prefillSpeed0} seconds/question")
print(f"First token: {firstSpeed} tokens/second.        -- {firstSpeed0} seconds/token")
print(f"Following tokens: {followingSpeed} tokens/second.")


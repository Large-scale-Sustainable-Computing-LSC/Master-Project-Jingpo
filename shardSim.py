from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer, LlamaModel
from datasets import load_dataset
from io import BytesIO
import torch
import gc
import time

import threading  
import psutil     

MODEL_PATH = "./Llama-3.2-1B"
LAYER_TO_SPLIT = 12
MAX_NEW_TOKENS = 32

TEST_SET = "meta-llama/Llama-3.2-1B-evals"
DATA_NAME = "Llama-3.2-1B-evals__drop__details"
DATA_TO_START = 0
MAX_QUESTIONS = 16
STEP_SIZE = 39
RANDOM_DATA = False


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




data = load_dataset(TEST_SET,
        name = DATA_NAME,
        split="latest"
)

if DATA_TO_START >= len(data):
    print(f"Invalid DATA_TO_START {DATA_TO_START}, will start from 0.")
    DATA_TO_START = 0



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print("Using device:", device)

monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(f"cuda_shard_3B.csv",),  
        daemon=True  # 设为守护线程，主线程退出时自动终止
    )

#monitor_thread.start()


config_all = AutoConfig.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

tokenizer.pad_token = tokenizer.eos_token

# eos_token_id = tokenizer.eos_token_id


start_time = time.time()

""" model_01 = LlamaForCausalLM.from_pretrained(MODEL_PATH)
model_01.model.layers = model_01.model.layers[ :LAYER_TO_SPLIT]
model_01.model.norm = torch.nn.Identity()  # 跳过norm层
model_01.eval()
model_01.to(device)
print(f"model_01 to device, {len(model_01.model.layers)}") """


model_01 = LlamaModel.from_pretrained(MODEL_PATH)
model_01.layers = model_01.layers[ :LAYER_TO_SPLIT]
model_01.norm = torch.nn.Identity()  # 跳过norm层
model_01.eval()
model_01.to(device)
print(f"model_01 to device, {len(model_01.layers)}")
#print(model_01)

model_02 = LlamaForCausalLM.from_pretrained(MODEL_PATH)
model_02.model.layers = model_02.model.layers[LAYER_TO_SPLIT: ]
model_02.eval()
model_02.to(device)
print(f"model_02 to device, {len(model_02.model.layers)}")
#print(model_02)

end_time_load = time.time()
read_time = end_time_load - start_time
print(f"Models Loaded. [{read_time}s]")


#past_key_values_01 = None
#past_key_values_02 = None

inf_start_time = time.time()

prefill_time_total = 0
first_token_total = 0
first_token_01 = 0
first_token_02 = 0
following_token_total = 0
following_token_01 = 0
following_token_02 = 0
process_time_total = 0

questions = data["input_final_prompts"]

past_key_values_01 = None
past_key_values_02 = None

with torch.no_grad(): 

    all_ser = 0
    all_time = 0
    for que_num_count in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):
        
        question  = questions[que_num_count]

        print(f"\n\nInferencing {que_num_count}", end="")

        prefill_start_time = time.time()

        inputs = tokenizer(question, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        position_ids = attention_mask.long().cumsum(dim=1) - 1

        generated_text = input_ids

        prefill_end_time = time.time()
        prefill_time = prefill_end_time - prefill_start_time
        prefill_time_total += prefill_time

        """ if 'past_key_values_01' in locals():   #清理缓存
            del past_key_values_01
        if 'past_key_values_02' in locals():  
            del past_key_values_02
        torch.cuda.empty_cache()
        gc.collect() """

        past_key_values_01 = None
        past_key_values_02 = None

        rawCount = 0
        serCount = 0
        timeThis = 0

        for count in range(MAX_NEW_TOKENS):

            start_token_time = time.time()

            outputs_01 = model_01(
                input_ids=generated_text[:, -1:] if past_key_values_01 else generated_text,

                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_01,
                use_cache=True,
                output_hidden_states=True
            )

            torch.mps.synchronize()
            checkPoint_01 = time.time()

            last_hidden_state = outputs_01.hidden_states[-1]
            #last_hidden_state = outputs_01.last_hidden_state
            past_key_values_01 = outputs_01.past_key_values

            def serialize_tensor(tensor):
                """将PyTorch张量序列化为gRPC消息"""
                tensor = tensor.detach().cpu()
                buffer = BytesIO()
                torch.save(tensor, buffer)
                return tensor.to(device)

            def measure_serialized_size(tensor):
                # 和 gRPC 中 serialize_tensor 完全一致的序列化方式
                buffer = BytesIO()
                torch.save(tensor.detach().cpu(), buffer)
                tensor_bytes = buffer.getvalue()
                
                raw_bytes = tensor.numel() * tensor.element_size()
                serialized_bytes = len(tensor_bytes)
                
                time0 = time.time()
                serialize_tensor(tensor)
                timeUsed = time.time() - time0
                #print("shape:", tensor.shape)
                #print("dtype:", tensor.dtype)
                #print("raw tensor bytes:", raw_bytes)
                #print("serialized bytes:", serialized_bytes)
                
                return raw_bytes, serialized_bytes, timeUsed
            
            
            # measure_serialized_size(last_hidden_state)
            rawThis,serThis,timeThis = measure_serialized_size(last_hidden_state)
            if count == 0:
                rawCount += rawThis
                serCount += serThis

            

            outputs_02 = model_02(
                inputs_embeds=last_hidden_state,

                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values_02,
                use_cache=True,
            )


            torch.mps.synchronize()
            end_token_time = time.time()
            single_token_time = end_token_time - start_token_time
            single_token_01 = checkPoint_01 - start_token_time
            single_token_02 = end_token_time - checkPoint_01

            if count == 0:
                first_token_total += single_token_time
                first_token_01 += single_token_01
                first_token_02 += single_token_02
            else:
                following_token_total += single_token_time
                following_token_01 += single_token_01
                following_token_02 += single_token_02


            start_pro_time = time.time()

            logits = outputs_02.logits
            past_key_values_02 = outputs_02.past_key_values

            # pick next token
            next_token = logits[:, -1:, :].argmax(dim=-1).detach()



            generated_text = torch.cat([generated_text, next_token], dim=-1)



            """ rawThis,serThis = measure_serialized_size(generated_text)
            if count != 0:
                rawCount += rawThis
                serCount += serThis
 """
    

            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            #print(attention_mask)

            position_ids = torch.tensor([[generated_text.shape[1] - 1]], device=device)

            end_pro_time = time.time()
            pro_time = end_pro_time - start_pro_time
            process_time_total += pro_time

        print("raw: ", rawCount / 1)
        print("ser: ", serCount / 1)
        all_ser += serCount
        all_time += timeThis
        output_string = tokenizer.decode(generated_text[0][input_ids.size(1):])
        print(output_string)
        #print("defult kv")
        #print(past_key_values_01[0][0].device) 
    print("avg_ser", all_ser / MAX_QUESTIONS)
    print("avg_de_time", all_time / MAX_QUESTIONS)

""" inf_end_time = time.time()
inf_time_total = inf_end_time - inf_start_time

firstSpeed0 = first_token_total / MAX_QUESTIONS
prefillSpeed0 = prefill_time_total / MAX_QUESTIONS

overallSpeed = (MAX_QUESTIONS * MAX_NEW_TOKENS) / inf_time_total
prefillSpeed = MAX_QUESTIONS / prefill_time_total
firstSpeed = MAX_QUESTIONS / first_token_total
followingSpeed = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_total
proSpeed = (MAX_QUESTIONS * MAX_NEW_TOKENS) / process_time_total
first_01 = MAX_QUESTIONS / first_token_01
first_02 = MAX_QUESTIONS / first_token_02
following_01 = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_01
following_02 = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_02

print("  ========  Finished (shardSim)  ========  ")
print(f"Data used: {DATA_NAME}")
print(f"Device used: {device}")
print(f"Layer on shard01: 0 - {LAYER_TO_SPLIT -1}")
print(f"Layer on shard02: {LAYER_TO_SPLIT} - 15")
print(f"From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.\n")

print(f"Time used: \n{inf_time_total} seconds.\n")

print(f"Overall speed: \n{overallSpeed} tokens/second.\n{1/overallSpeed} seconds/token.\n")
print(f"Prefill speed: \n{prefillSpeed} questions/second.\n{1/prefillSpeed} seconds/question\n")
print(f"First token: \n{firstSpeed} tokens/second.\n{1/firstSpeed} seconds/token.\n")
print(f"First token (shard01): \n{first_01} tokens/second.\n{1/first_01} seconds/token.\n")
print(f"First token (shard02): \n{first_02} tokens/second.\n{1/first_02} seconds/token.\n")
print(f"Following tokens: \n{followingSpeed} tokens/second.\n{1/followingSpeed} seconds/token.\n")
print(f"Following tokens (shard01): \n{following_01} tokens/second.\n{1/following_01} seconds/token.\n")
print(f"Following tokens (shard02): \n{following_02} tokens/second.\n{1/following_02} seconds/token.\n")
print(f"Processing Speed: \n{proSpeed} tokens/second.\n{1/proSpeed} seconds/token.\n") """
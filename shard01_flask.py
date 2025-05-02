from flask import Flask, request
import requests
from threading import Thread
import time
from datasets import load_dataset
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
import torch, gc

app = Flask(__name__)

MY_IP = '192.168.2.23'
MY_PORT = 5000
#TARGET_IP = '192.168.2.13'
TARGET_IP = '145.100.135.35'
TARGET_PORT = 5087

MODEL_PATH = "./Llama-3.2-1B"
LAYER_TO_SPLIT = 4      # 16 in total

#QUESTION = "Where is the capital of USA?"
#QUESTION = "macOS with m1 chip and 16G memory VS linux with AMD 5700XT GPU (8G VRAM), which one could be better for LLM deployment?"
QUESTION = "When was the first iPhone released?"


MAX_NEW_TOKENS = 32


DATA_NAME = "Llama-3.2-1B-evals__drop__details"
DATA_TO_START =88
MAX_QUESTIONS = 16
STEP_SIZE = 13
RANDOM_DATA = False


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print("Using device:", device)



def init_connection(tar, att, pos, gen):
    try:
        url = f'http://{TARGET_IP}:{TARGET_PORT}/{tar}'
        data = {
            'layer':LAYER_TO_SPLIT,
            'att': att.tolist(),         
            #'att_shape': list(att.shape),   
            'pos': pos.tolist(),      
            #'pos_shape': list(pos.shape),   
            'gen': gen.tolist(),         
            #'gen_shape': list(gen.shape),   
        }
        print("sending...")
        response = requests.post(url, json=data)
        #print(response.status_code)
        return response.status_code
    except Exception as e:
        print("!!! init fail !!!")
        print(e)
        return -1
    
    
def post_tensor(tar, hid, att, pos, gen):
    url = f'http://{TARGET_IP}:{TARGET_PORT}/{tar}'
    data = {
        'hid': hid.tolist(),  # 转换为Python列表
        #'hid_shape': list(hid.shape),  # 保存原始形状
        'att': att.tolist(),         
        #'att_shape': list(att.shape),   
        'pos': pos.tolist(),      
        #'pos_shape': list(pos.shape),   
        'gen': gen.tolist(),         
        #'gen_shape': list(gen.shape),   
        #'dtype': str(tensor_data.dtype)
    }
    try:
        response = requests.post(url, json=data)
        resp = response.json()
        return resp
    except Exception as e:
        print("!!! post_tensor fail !!!")
        print(e)
        return -1
        
    

def run(model, tokenizer, question):

    prefill_start_time = time.time()
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # to share:
    attention_mask = inputs.attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    generated_text = input_ids
    

    # local: 
    if 'past_key_values' in locals():   #清理缓存
        del past_key_values
    torch.cuda.empty_cache()
    gc.collect()

    past_key_values = None
    print("Start connecting")


    response = init_connection(tar="init", att=attention_mask, pos=position_ids, gen=generated_text)
    print(response)

    prefill_end_time = time.time()
    prefill_time = prefill_end_time - prefill_start_time
    first_token_time = 0
    first_token_time_01 = 0
    following_token_time = 0
    following_token_time_01 = 0

    if response == 200:
        with torch.no_grad():
            for count in range(MAX_NEW_TOKENS):
                
                #print(f"Counter: {count}")

                start_token_time = time.time()
                start_token_time_01 = time.time()
                outputs = model(
                    input_ids=generated_text[:, -1:] if past_key_values else generated_text,

                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                end_token_time_01 = time.time()

                last_hidden_state = outputs.hidden_states[-1]
                past_key_values = outputs.past_key_values

                res_tensor = post_tensor(tar="hidden", hid=last_hidden_state, att=attention_mask, pos=position_ids, gen=generated_text)
                
                end_token_time = time.time()
                single_token_time = end_token_time - start_token_time
                single_token_time_01 = end_token_time_01 - start_token_time_01

                if count == 0:
                    first_token_time += single_token_time
                    first_token_time_01 += single_token_time_01
                else: 
                    following_token_time += single_token_time
                    following_token_time_01 += single_token_time_01
                

                #next_token = torch.tensor(res_tensor['tok'],device = device).reshape(res_tensor['tok_shape'])
                #attention_mask = torch.tensor(res_tensor['att'],device = device).reshape(res_tensor['att_shape'])
                #position_ids = torch.tensor(res_tensor['pos'],device = device).reshape(res_tensor['pos_shape'])
                #generated_text = torch.tensor(res_tensor['gen'],device = device).reshape(res_tensor['gen_shape'])

                next_token = torch.tensor(res_tensor['tok'],device = device)
                attention_mask = torch.tensor(res_tensor['att'],device = device)
                position_ids = torch.tensor(res_tensor['pos'],device = device)
                generated_text = torch.tensor(res_tensor['gen'],device = device)



                if next_token == -1: 
                    print("!!! next_token error !!!")
                    break
                
                #generated_text = torch.cat([generated_text, next_token], dim=-1)

                #attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

                #position_ids = torch.tensor([[generated_text.shape[1] - 1]])


        output_string = tokenizer.decode(generated_text[0][input_ids.size(1):])
        print("Output with `.forward()`:\n" + output_string)
        return(prefill_time, first_token_time, following_token_time, first_token_time_01, following_token_time_01)




if __name__ == '__main__':

    data = load_dataset("meta-llama/Llama-3.2-1B-evals",
        name = DATA_NAME,
        split="latest"
    )

    if DATA_TO_START >= len(data):
        print(f"Invalid DATA_TO_START {DATA_TO_START}, will start from 0.")
        DATA_TO_START = 0

    questions  = data["input_question"]

    model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
    model.model.layers = model.model.layers[ :LAYER_TO_SPLIT]
    model.model.norm = torch.nn.Identity()  # skip norm
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token

    prefill_time_total = 0
    first_token_total = 0
    following_token_total = 0
    first_token_total_01 = 0
    following_token_total_01 = 0
    
    inf_start_time = time.time()

    for que_num_count in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):
        print(f"Inferencing {que_num_count}")
        prefill_time, first_token_time, following_token_time, first_token_time_01, following_token_time_01 = run(model, tokenizer, questions[que_num_count])
        prefill_time_total += prefill_time
        first_token_total += first_token_time
        following_token_total += following_token_time
        first_token_total_01 += first_token_time_01
        following_token_total_01 += following_token_time_01


    inf_end_time = time.time()
    inf_time_total = inf_end_time - inf_start_time

    firstSpeed0 = first_token_total / MAX_QUESTIONS
    firstSpeed0_01 = first_token_total_01 / MAX_QUESTIONS
    prefillSpeed0 = prefill_time_total / MAX_QUESTIONS


    overallSpeed = (MAX_QUESTIONS * MAX_NEW_TOKENS) / inf_time_total
    prefillSpeed = MAX_QUESTIONS / prefill_time_total
    firstSpeed = MAX_QUESTIONS / first_token_total
    firstSpeed_01 = MAX_QUESTIONS / first_token_total_01
    followingSpeed = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_total
    followingSpeed_01 = MAX_QUESTIONS * (MAX_NEW_TOKENS - 1) / following_token_total_01

    print("  ========  Finished (shard)  ========  ")
    print(f"Data used: {DATA_NAME}")
    print(f"Layer to split: {LAYER_TO_SPLIT}")
    print(f"From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.")
    print(f"Time used: {inf_time_total} seconds.")
    print(f"Overall speed: {overallSpeed} tokens/second.")
    print(f"Prefill speed: {prefillSpeed} questions/second.     -- {prefillSpeed0} seconds/question")
    print(f"First token: {firstSpeed} tokens/second.        -- {firstSpeed0} seconds/token")
    print(f"Following tokens: {followingSpeed} tokens/second.")

    print(f"First token shard01: {firstSpeed_01} tokens/second.        -- {firstSpeed0_01} seconds/token")
    print(f"Following tokens shard01: {followingSpeed_01} tokens/second.")




    

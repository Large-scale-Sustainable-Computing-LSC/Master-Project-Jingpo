from flask import Flask, request, jsonify
import requests
import time

from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
import torch,gc

app = Flask(__name__)

MY_IP = '0.0.0.0'   # 145.100.135.35
MY_PORT = 5087
#TARGET_IP = '192.168.2.23'
#TARGET_PORT = 5000

MODEL_PATH = "./Llama-3.2-1B"
LAYER_TO_SPLIT = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

past_key_values = None
attention_mask = None
position_ids = None
generated_text = None

model = ()

first_token_count = 0
first_total_count = 0
following_token_count = 0
following_total_count = 0
init_counter = 0




@app.route("/test",methods=["POST"])
def test_con():
    try:
        data = request.get_json()
        if data['z']:
            print(data['z'])
            return '',200
        return '', 200
    except Exception as e:
        print(f"init fail: {e}")
        return '',400

@app.route("/init",methods=["POST"])
def init_connection():
    global attention_mask, position_ids, generated_text, past_key_values, model, init_counter, LAYER_TO_SPLIT
    try:
        data = request.get_json()
        layer = data['layer']
        attention_mask = torch.tensor(data['att'],device = device)
        position_ids = torch.tensor(data['pos'],device = device)
        generated_text = torch.tensor(data['gen'],device = device)
        
        if layer != LAYER_TO_SPLIT:
            model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
            model.model.layers = model.model.layers[layer: ]
            model.eval()
            model.to(device)
            LAYER_TO_SPLIT = layer

        if 'past_key_values' in locals():   #清理缓存
            del past_key_values
        torch.cuda.empty_cache()
        gc.collect()

        past_key_values = None
        #past_key_values = None
        init_counter += 1
        return '',200
    except Exception as e:
        print(f"init fail: {e}")
        return '',400
    
@app.route("/hidden",methods=["POST"])
def get_hidden():
    global attention_mask, position_ids, generated_text, past_key_values, first_token_count, following_token_count, init_counter, first_total_count, following_total_count
    try:
        start_time_T = time.time()
        data = request.get_json()
        hidden_state = torch.tensor(data['hid'],device = device)
        attention_mask = torch.tensor(data['att'],device = device)
        position_ids = torch.tensor(data['pos'],device = device)
        generated_text = torch.tensor(data['gen'],device = device)

        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(
                inputs_embeds=hidden_state,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        end_time = time.time()

        logits = outputs.logits
        past_key_values = outputs.past_key_values

        next_token = logits[:, -1:, :].argmax(dim=-1).detach()

        generated_text = torch.cat([generated_text, next_token], dim=-1)

        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        position_ids = torch.tensor([[generated_text.shape[1] - 1]],device = device)

        res = {
            'tok': next_token.tolist(),
            #'tok_shape': list(next_token.shape),
            'att': attention_mask.tolist(),
            #'att_shape': list(attention_mask.shape),
            'pos': position_ids.tolist(),
            #'pos_shape': list(position_ids.shape),
            'gen': generated_text.tolist(),
            #'gen_shape': list(generated_text.shape),
            #'dtype': str(next_token.dtype)
        }
        end_time_T = time.time()
        time_used = end_time - start_time
        time_used_T = end_time_T - start_time_T
        if init_counter > 0:
            first_token_count += time_used
            first_total_count += time_used_T
            init_counter = 0
        else:
            following_token_count += time_used
            following_total_count += time_used_T

        return jsonify(res), 200
    
    except Exception as e:
        print(f"fail to run model: {e}")
        return '',400

if __name__ == '__main__':
    print("Using device:", device)
    app.run(host = MY_IP, port = MY_PORT)

    print(f"first token total: {first_total_count}")
    print(f"following tokens total: {following_total_count}")
    print(f"first token: {first_token_count}")
    print(f"following tokens: {following_token_count}")
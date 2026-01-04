import grpc
import torch
import gc
from torch import nn
import time
import llama_shard_pb2
import llama_shard_pb2_grpc
from datasets import load_dataset
from io import BytesIO 
from transformers import LlamaForCausalLM, AutoTokenizer , LlamaModel

import threading  
import psutil     
import time       

# 配置参数
MODEL_PATH = "../Llama-3.2-3B"
LAYER_TO_SPLIT = 12
DEVICE = torch.device("mps")  # 根据设备类型调整
TARGET_IP = "145.100.135.35"
MAX_NEW_TOKENS = 32

TEST_SET  = "meta-llama/Llama-3.2-3B-evals"
DATA_NAME = "Llama-3.2-3B-evals__drop__details"
DATA_TO_START = 0
MAX_QUESTIONS = 128
STEP_SIZE = 26

def monitor_resources(filename):
    """资源监控函数，每秒记录一次数据"""
    with open(filename, 'w') as f:
        f.write("timestamp,memory_usage_mb\n")  
        try:
            while True:
                # 获取内存使用（单位：MB）
                memory_mb = psutil.virtual_memory().used / (1024 ** 2)
                # 写入数据
                timestamp = time.time()
                f.write(f"{timestamp},{memory_mb:.2f}\n")
                f.flush()  # 确保实时写入
                time.sleep(1)
        finally:
            f.close()

class ShardClient:
    def __init__(self, target_ip):
        options = [
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
            ('grpc.max_send_message_length', 100 * 1024 * 1024)
        ]
        self.channel = grpc.insecure_channel(f"{target_ip}:5087", options = options)
        self.stub = llama_shard_pb2_grpc.ShardServiceStub(self.channel)
        
    def _serialize_tensor(self, tensor):
        """将PyTorch张量序列化为gRPC消息"""
        tensor = tensor.detach().cpu()
        buffer = BytesIO()
        torch.save(tensor, buffer)

        #byte_data = buffer.getvalue()
        #print(f"Serialized Tensor Size: {len(byte_data)} bytes")

        return llama_shard_pb2.TensorData(
            tensor_bytes=buffer.getvalue(),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype)
        )
    
    def _deserialize_tensor(self, tensor_data, device):
        """反序列化gRPC消息为PyTorch张量"""
        tensor = torch.load(BytesIO(tensor_data.tensor_bytes))
        return tensor.to(device)
    
    def init_connection(self, layer):
        """初始化分片连接"""
        request = llama_shard_pb2.InitRequest(
            layer=layer,
            #att=self._serialize_tensor(att),
            #pos=self._serialize_tensor(pos),
            #gen=self._serialize_tensor(gen)
            )
        #self.stub.InitConnection(request)
        response = self.stub.InitConnection(request)  # 捕获响应
        return response

    
    def process_hidden(self, hid, att, pos, gen, device):
        """处理隐藏层请求"""
        #print("hid, att, pos, gen: ")
        request = llama_shard_pb2.HiddenRequest(
            hid=self._serialize_tensor(hid),
            att=self._serialize_tensor(att),
            pos=self._serialize_tensor(pos),
            gen=self._serialize_tensor(gen))
        
        tr_start = time.time()
        response = self.stub.ProcessHidden(request)
        tr_time = time.time() - tr_start

        return {
            #'tok': self._deserialize_tensor(response.tok, device),
            'att': self._deserialize_tensor(response.att, device),
            'pos': self._deserialize_tensor(response.pos, device),
            'gen': self._deserialize_tensor(response.gen, device),
            'time_less':response.time_less,
            'time_more':response.time_more,
            'tr_time':tr_time,
        }
    
    def test_tensor(self, tensor, device):
        tensor_pb = self._serialize_tensor(tensor)
        request = llama_shard_pb2.TestTensor(tensor=tensor_pb)  # 使用正确的消息类
        response = self.stub.TestTensorMethod(request)
        return self._deserialize_tensor(response.tensor, device)

# 初始化gRPC客户端
client = ShardClient(TARGET_IP)


def run(model, tokenizer, question):
    #prefill_start = time.time()
    
    # 初始文本处理
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    generated_text = input_ids.clone()
    
    #size_bytes = generated_text.storage().nbytes()
    #print(f"generated_text: {size_bytes} bytes")

    # 清理缓存
    """ if 'past_key_values' in locals():
        del past_key_values
    torch.mps.empty_cache()
    gc.collect() """
    
    past_key_values = None

    # 初始化分片连接
    init_res = client.init_connection(
        layer=LAYER_TO_SPLIT,
        #att=attention_mask,
        #pos=position_ids,
        #gen=generated_text,
        #device=DEVICE
    )
    print(f"init_res: {init_res}")
    
    #prefill_time = time.time() - prefill_start
    timing_metrics = {
        'first_token': 0,
        'following_tokens': 0,
        'shard02_tr': 0,
        'shard02_de': 0,
        'shard02_in': 0,
        'shard01_first': 0,
        'shard01_following': 0,
        'shard02_tr_1': 0,
        'shard02_de_1': 0,
        'shard02_in_1': 0,
        'shard02_tr_f': 0,
        'shard02_de_f': 0,
        'shard02_in_f': 0,
        #'prefill': prefill_time
    }



    # 生成循环
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            token_start = time.time()
            
            # 本地前向计算
            #print(past_key_values == None)
            outputs = model(
                input_ids=generated_text[:, -1:] if past_key_values else generated_text,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            
            # 获取隐藏层并发送给分片
            last_hidden = outputs.hidden_states[-1]
            #print(last_hidden)
            past_key_values = outputs.past_key_values

            torch.mps.synchronize()
            
            s01_time = time.time() - token_start
            if timing_metrics['shard01_first'] == 0:
                timing_metrics['shard01_first'] += s01_time
            else:
                timing_metrics['shard01_following'] += s01_time

            # 调用gRPC服务
            res = client.process_hidden(
                hid=last_hidden,
                att=attention_mask,
                pos=position_ids,
                gen=generated_text,
                device=DEVICE
            )
            
            # 更新生成状态
            #next_token = res['tok']
            generated_text = res['gen']
            attention_mask = res['att']
            position_ids = res['pos']

            tr_time = res['tr_time']
            time_less = res['time_less']
            time_more = res['time_more']

            #tr_cost = tr_time - time_more
            #de_cost = time_more - time_less

            timing_metrics['shard02_tr'] += tr_time
            timing_metrics['shard02_de'] += time_more
            timing_metrics['shard02_in'] += time_less

            
            # 记录时间
            token_time = time.time() - token_start
            if timing_metrics['first_token'] == 0:
                timing_metrics['first_token'] = token_time
                timing_metrics['shard02_tr_1'] += tr_time
                timing_metrics['shard02_de_1'] += time_more
                timing_metrics['shard02_in_1'] += time_less
                
            else:
                timing_metrics['following_tokens'] += token_time
                timing_metrics['shard02_tr_f'] += tr_time
                timing_metrics['shard02_de_f'] += time_more
                timing_metrics['shard02_in_f'] += time_less
    
    # 解码结果
    #print(generated_text)
    output_text = tokenizer.decode(
        generated_text[0][inputs.input_ids.shape[1]:],
        #skip_special_tokens=True
    )
    return output_text, timing_metrics

if __name__ == '__main__':

    #rand_tensor = torch.rand(2, 3)
    #print(f"rand_tensor: {rand_tensor}")
    #res_tensor = client.test_tensor(rand_tensor,DEVICE)
    #print(f"res_tensor: {res_tensor}")

    # 启动资源监控线程
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(f"../finalRes/log_3B_{LAYER_TO_SPLIT}/shard01_mps.csv", ),  # 保存到当前目录的 resource_log.csv
        daemon=True  # 设为守护线程，主线程退出时自动终止
    )
    monitor_thread.start()

    print("Loading model ...")


    # 初始化模型
    model = LlamaModel.from_pretrained(MODEL_PATH)
    model.layers = model.layers[:LAYER_TO_SPLIT]
    model.norm = nn.Identity()  # 跳过归一化层
    model.eval()
    model.to(DEVICE)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_dataset(TEST_SET, 
        name = DATA_NAME,
        split = "latest"
    )

    client.init_connection(
        layer=LAYER_TO_SPLIT,
    )
    
    first_total = 0
    following_total = 0

    tr_all = 0
    de_all = 0
    in_all = 0
    tr_1 = 0
    de_1 = 0
    in_1 = 0
    tr_f = 0
    de_f = 0
    in_f = 0
    s01_first = 0
    s01_follow = 0

    start_time_0 = time.time()
    # 执行推理

    outputFile = f"../finalRes/log_3B_{LAYER_TO_SPLIT}/output.txt"
    with open(outputFile, 'w') as outf:
        outf.write("output\n")

        for idx in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):  # 示例运行4个问题
            question = dataset["input_final_prompts"][idx]
            print(f"\nProcessing question {idx}: ...")
            
            response, timing = run(model, tokenizer, question)

            outf.write(f"[{idx}]: {response}\n\n\n")
            print(f"Response: {response}")

            first_total += timing['first_token']
            following_total += timing['following_tokens']
            tr_all += timing['shard02_tr']
            de_all += timing['shard02_de']
            in_all += timing['shard02_in']
            s01_first += timing['shard01_first']
            s01_follow += timing['shard01_following']
            tr_1 += timing['shard02_tr_1']
            de_1 += timing['shard02_de_1']
            in_1 += timing['shard02_in_1']
            tr_f += timing['shard02_tr_f']
            de_f += timing['shard02_de_f']
            in_f += timing['shard02_in_f']
    


    total_time_0 = time.time() - start_time_0
    overallSpeed = MAX_NEW_TOKENS * MAX_QUESTIONS / total_time_0
    toFirstToken = first_total / MAX_QUESTIONS
    toFollowing = following_total / ((MAX_NEW_TOKENS - 1) * MAX_QUESTIONS)
    averageDelay = (first_total + following_total) / (MAX_QUESTIONS * MAX_NEW_TOKENS)

    tr_la = (tr_all - de_all) / (MAX_NEW_TOKENS * MAX_QUESTIONS)
    de_la = (de_all - in_all) / (MAX_NEW_TOKENS * MAX_QUESTIONS)
    in_la = in_all / (MAX_NEW_TOKENS * MAX_QUESTIONS)
    tr_la_1 = (tr_1 - de_1) / (MAX_QUESTIONS)
    de_la_1 = (de_1 - in_1) / (MAX_QUESTIONS)
    in_la_1 = in_1 / (MAX_QUESTIONS)
    tr_la_f = (tr_f - de_f) / ((MAX_NEW_TOKENS - 1) * MAX_QUESTIONS)
    de_la_f = (de_f - in_f) / ((MAX_NEW_TOKENS - 1) * MAX_QUESTIONS)
    in_la_f = in_f / ((MAX_NEW_TOKENS - 1) * MAX_QUESTIONS)

    s01_la_first = s01_first / MAX_QUESTIONS
    s01_la_follow = s01_follow / ((MAX_NEW_TOKENS-1) * MAX_QUESTIONS)
    s01_la_all = s01_first+s01_follow / (MAX_NEW_TOKENS * MAX_QUESTIONS)


    
    result_content = f"""========  Finished (shard)  ========
Data used: {DATA_NAME}
Layer to split: {LAYER_TO_SPLIT}
Max generation length: {MAX_NEW_TOKENS} tokens.
From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.

Overall Speed: 
{overallSpeed} tokens/sec
{averageDelay} s

- First token: 
{1/toFirstToken} tokens/sec
{toFirstToken} s

- Following tokens: 
{1/toFollowing} tokens/sec
{toFollowing} s

- 01 first token
{1/s01_first} tokens/sec
{s01_la_first} s

- 01 following token
{1/s01_la_follow} tokens/sec
{s01_la_follow} s

- 01 all
{1/s01_la_all} tokens/sec
{s01_la_all} s

- 01->02->01 latency: {tr_la} s
- en/de latency: {de_la} s
- 02 inference latency: {in_la} s
                        {1/in_la} tokens/sec

- 01->02->01 latency 1st: {tr_la_1} s
- en/de latency 1st: {de_la_1} s
- 02 inference latency 1st: {in_la_1} s
                        {1/in_la_1} tokens/sec

- 01->02->01 latency following: {tr_la_f} s
- en/de latency following: {de_la_f} s
- 02 inference latency following: {in_la_f} s
                        {1/in_la_f} tokens/sec"""
    
    

    print(result_content)

"""     save_path =f"log_{LAYER_TO_SPLIT}/results.txt"

    # 将结果写入文件
    with open(save_path, 'w', encoding='utf-8') as f_res:
        f_res.write(result_content)
        f_res.close() """
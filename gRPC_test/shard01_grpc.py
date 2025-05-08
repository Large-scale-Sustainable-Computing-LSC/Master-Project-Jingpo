import grpc
import torch
import gc
from torch import nn
import time
import llama_shard_pb2
import llama_shard_pb2_grpc
from datasets import load_dataset
from io import BytesIO 
from transformers import LlamaForCausalLM, AutoTokenizer

import threading  
import psutil     
import time       

# 配置参数
MODEL_PATH = "../Llama-3.2-3B"
LAYER_TO_SPLIT = 4
DEVICE = torch.device("mps")  # 根据设备类型调整
TARGET_IP = "145.100.135.35"
MAX_NEW_TOKENS = 32

DATA_NAME = "Llama-3.2-3B-evals__drop__details"
DATA_TO_START = 7
MAX_QUESTIONS = 4
STEP_SIZE = 13

def monitor_resources(filename):
    """资源监控函数，每秒记录一次数据"""
    with open(filename, 'w') as f:
        f.write("timestamp,memory_usage_mb\n")  
        while True:
            # 获取内存使用（单位：MB）
            memory_mb = psutil.virtual_memory().used / (1024 ** 2)
            # 写入数据
            timestamp = time.time()
            f.write(f"{timestamp},{memory_mb:.2f}\n")
            f.flush()  # 确保实时写入
            time.sleep(1)

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
        
        response = self.stub.ProcessHidden(request)
        
        return {
            #'tok': self._deserialize_tensor(response.tok, device),
            'att': self._deserialize_tensor(response.att, device),
            'pos': self._deserialize_tensor(response.pos, device),
            'gen': self._deserialize_tensor(response.gen, device)
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
    
    # 清理缓存
    if 'past_key_values' in locals():
        del past_key_values
    torch.mps.empty_cache()
    gc.collect()
    
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
            
            # 记录时间
            token_time = time.time() - token_start
            if timing_metrics['first_token'] == 0:
                timing_metrics['first_token'] = token_time
            else:
                timing_metrics['following_tokens'] += token_time
    
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
        args=(f"resource_log_mps_{LAYER_TO_SPLIT}.csv", ),  # 保存到当前目录的 resource_log.csv
        daemon=True  # 设为守护线程，主线程退出时自动终止
    )
    #monitor_thread.start()

    print("Loading model ...")


    # 初始化模型
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH)
    model.model.layers = model.model.layers[:LAYER_TO_SPLIT]
    model.model.norm = nn.Identity()  # 跳过归一化层
    model.eval()
    model.to(DEVICE)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_dataset("meta-llama/Llama-3.2-3B-evals", 
        name = DATA_NAME,
        split = "latest"
    )

    client.init_connection(
        layer=LAYER_TO_SPLIT,
    )
    
    first_total = 0
    following_total = 0

    start_time_0 = time.time()
    # 执行推理
    for idx in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS)*STEP_SIZE, STEP_SIZE):  # 示例运行4个问题
        question = dataset["input_final_prompts"][idx]
        print(f"\nProcessing question {idx}: ...")
        
        response, timing = run(model, tokenizer, question)
        
        print(f"Response: {response}")
        first_total += timing['first_token']
        following_total += timing['following_tokens']
    
    total_time_0 = time.time() - start_time_0
    overallSpeed = MAX_NEW_TOKENS * MAX_QUESTIONS / total_time_0
    toFirstToken = first_total / MAX_QUESTIONS
    toFollowing = following_total / ((MAX_NEW_TOKENS - 1) * MAX_QUESTIONS)


    print("  ========  Finished (shard)  ========  ")
    print(f"Data used: {DATA_NAME}")
    print(f"Layer to split: {LAYER_TO_SPLIT}")
    print(f"From {DATA_TO_START} with step size {STEP_SIZE}, {MAX_QUESTIONS} questions in total.")
    print(f"Overall Speed: {overallSpeed:.4f} tokens/sec  |  to first token: {toFirstToken:.4f}s  |  "
              f"to following tokens: {toFollowing:.4f}s")
from concurrent import futures
import grpc
import torch
import gc
import llama_shard_pb2
import llama_shard_pb2_grpc
from io import BytesIO 
from transformers import LlamaForCausalLM

import threading  
import psutil     
import time       

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


class ShardServicer(llama_shard_pb2_grpc.ShardServiceServicer):
    def __init__(self):
        self.model = None
        self.past_key_values = None
        self.layer_to_split = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def InitConnection(self, request, context):
        if request.layer != self.layer_to_split:
            self.model = LlamaForCausalLM.from_pretrained("../Llama-3.2-3B")
            self.model.model.layers = self.model.model.layers[request.layer:]
            self.model.eval()
            #print(self.model)
            self.model.to(self.device)
            self.layer_to_split = request.layer
        
        if hasattr(self, 'past_key_values'):   #清理缓存
            del self.past_key_values
        torch.cuda.empty_cache()
        gc.collect()
        self.past_key_values = None
            
        return llama_shard_pb2.Init_res(res = 200)
    
    def ProcessHidden(self, request, context):
        def deserialize_tensor(tensor_data):
            tensor = torch.load(BytesIO(tensor_data.tensor_bytes))
            return tensor.to(self.device)
            
        hidden = deserialize_tensor(request.hid)
        att = deserialize_tensor(request.att)
        pos = deserialize_tensor(request.pos)
        gen = deserialize_tensor(request.gen)
        #gen = None
        
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=hidden,
                attention_mask=att,
                position_ids=pos,
                past_key_values=self.past_key_values,
                use_cache=True
            )
        logits = outputs.logits
        self.past_key_values = outputs.past_key_values

        next_token = logits[:, -1:, :].argmax(dim=-1).detach()
        #print(next_token)
        gen = torch.cat([gen, next_token], dim=-1)
        att = torch.cat([att, torch.ones_like(next_token)], dim=-1)
        pos = torch.tensor([[gen.shape[1] - 1]], device=self.device)
        
        def serialize_tensor(tensor):
            buffer = BytesIO()
            torch.save(tensor.cpu(), buffer)
            return llama_shard_pb2.TensorData(
                tensor_bytes=buffer.getvalue(),
                shape=list(tensor.shape),
                dtype=str(tensor.dtype)
            )
            
        return llama_shard_pb2.HiddenResponse(
            #tok=serialize_tensor(next_token),
            att=serialize_tensor(att),
            pos=serialize_tensor(pos),
            gen=serialize_tensor(gen)
        )
    
    def TestTensorMethod(self, request, context):
        def deserialize_tensor(tensor_data):
            tensor = torch.load(BytesIO(tensor_data.tensor_bytes))
            return tensor.to(self.device)
        tensor =  deserialize_tensor(request.tensor)

        def serialize_tensor(tensor):
            buffer = BytesIO()
            torch.save(tensor.cpu(), buffer)
            return llama_shard_pb2.TensorData(
                tensor_bytes=buffer.getvalue(),
                shape=list(tensor.shape),
                dtype=str(tensor.dtype)
            )

        return llama_shard_pb2.TestTensor(tensor = serialize_tensor(tensor))


def serve():

    # 启动资源监控线程
    monitor_thread = threading.Thread(
        target=monitor_resources,
        args=(f"resource_log_cuda.csv",),  # 保存到当前目录的 resource_log.csv
        daemon=True  # 设为守护线程，主线程退出时自动终止
    )
    #monitor_thread.start()

    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_send_message_length', 100 * 1024 * 1024)
    ]

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options = options)
    llama_shard_pb2_grpc.add_ShardServiceServicer_to_server(
        ShardServicer(), server)
    server.add_insecure_port('[::]:5087')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
from concurrent import futures
from io import BytesIO
import time

import grpc
import torch
from transformers import LlamaForCausalLM

import llama_ready_pb2
import llama_ready_pb2_grpc


DEFAULT_DTYPE = torch.bfloat16
SERVER_HOST = "[::]"
SERVER_PORT = 5087
MODEL_SIZE_B = 8
PRELOADED_MODEL_ID = f"meta-llama/Meta-Llama-3.1-{MODEL_SIZE_B}B"
PRELOADED_LAYER_SPLIT = 1
PRELOADED_MAX_NEW_TOKENS = 32
PRELOADED_DO_SAMPLE = False
PRELOADED_TEMPERATURE = 1.0
PRELOADED_TOP_P = 1.0
PRELOADED_TOP_K = 0


def safe_torch_load(buffer, map_location):
    try:
        return torch.load(buffer, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(buffer, map_location=map_location)


def deserialize_tensor(tensor_data, device):
    tensor = safe_torch_load(BytesIO(tensor_data.tensor_bytes), map_location="cpu")
    return tensor.to(device)


def serialize_tensor(tensor):
    buffer = BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return llama_ready_pb2.TensorData(
        tensor_bytes=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def parse_torch_dtype(dtype_name):
    if not dtype_name:
        return DEFAULT_DTYPE

    normalized = dtype_name.replace("torch.", "")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, normalized)


def sample_next_token(logits, do_sample, temperature, top_p, top_k):
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=True")

    scores = logits / temperature

    if top_k > 0:
        top_k = min(top_k, scores.shape[-1])
        values, _ = torch.topk(scores, top_k)
        cutoff = values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if 0.0 < top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
        scores = torch.full_like(scores, float("-inf"))
        scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def build_runtime_config_dict(
    model_id,
    layer_split,
    torch_dtype,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    top_k,
):
    return {
        "model_id": model_id,
        "layer_split": layer_split,
        "torch_dtype": str(torch_dtype),
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


class ShardServicer(llama_ready_pb2_grpc.ShardServiceServicer):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.current_config = build_runtime_config_dict(
            model_id=PRELOADED_MODEL_ID,
            layer_split=PRELOADED_LAYER_SPLIT,
            torch_dtype=DEFAULT_DTYPE,
            max_new_tokens=PRELOADED_MAX_NEW_TOKENS,
            do_sample=PRELOADED_DO_SAMPLE,
            temperature=PRELOADED_TEMPERATURE,
            top_p=PRELOADED_TOP_P,
            top_k=PRELOADED_TOP_K,
        )
        self.past_key_values = None
        self.current_length = 0
        self._preload_model()

    def _preload_model(self):
        dtype = parse_torch_dtype(self.current_config["torch_dtype"])
        model = LlamaForCausalLM.from_pretrained(
            self.current_config["model_id"],
            torch_dtype=dtype,
        )
        model.model.layers = model.model.layers[self.current_config["layer_split"] :]
        model.eval()
        model.to(self.device)
        self.model = model

    def _requested_config(self, config):
        return build_runtime_config_dict(
            model_id=config.model_id,
            layer_split=config.layer_split,
            torch_dtype=config.torch_dtype or DEFAULT_DTYPE,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
        )

    def InitConnection(self, request, context):
        try:
            requested = self._requested_config(request.config)
            if requested != self.current_config:
                message = (
                    "Requested runtime config does not match preloaded server config. "
                    f"requested={requested}, preloaded={self.current_config}"
                )
                context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
                context.set_details(message)
                return llama_ready_pb2.InitResponse(status=412, message=message)

            self.past_key_values = None
            self.current_length = 0
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return llama_ready_pb2.InitResponse(status=200, message="ok")
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return llama_ready_pb2.InitResponse(status=500, message=str(exc))

    def PrefillStep(self, request, context):
        start = time.time()
        hidden = deserialize_tensor(request.hidden, self.device)
        attention_mask = deserialize_tensor(request.attention_mask, self.device)
        position_ids = deserialize_tensor(request.position_ids, self.device)

        infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        self.past_key_values = outputs.past_key_values

        next_token = sample_next_token(
            logits=logits,
            do_sample=self.current_config["do_sample"],
            temperature=self.current_config["temperature"],
            top_p=self.current_config["top_p"],
            top_k=self.current_config["top_k"],
        )
        infer_time = time.time() - infer_start

        self.current_length = attention_mask.shape[1] + 1

        return llama_ready_pb2.PrefillResponse(
            next_token=serialize_tensor(next_token),
            server_infer_s=infer_time,
            server_total_s=time.time() - start,
        )

    def DecodeStep(self, request, context):
        start = time.time()
        hidden = deserialize_tensor(request.hidden, self.device)

        attention_mask = torch.ones((1, self.current_length), device=self.device, dtype=torch.long)
        position_ids = torch.tensor([[self.current_length - 1]], device=self.device, dtype=torch.long)

        infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
            )
        logits = outputs.logits[:, -1, :]
        self.past_key_values = outputs.past_key_values

        next_token = sample_next_token(
            logits=logits,
            do_sample=self.current_config["do_sample"],
            temperature=self.current_config["temperature"],
            top_p=self.current_config["top_p"],
            top_k=self.current_config["top_k"],
        )
        infer_time = time.time() - infer_start

        self.current_length += 1

        return llama_ready_pb2.DecodeResponse(
            next_token=serialize_tensor(next_token),
            server_infer_s=infer_time,
            server_total_s=time.time() - start,
        )


def serve():
    options = [
        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ]

    servicer = ShardServicer()
    print(
        "Preloaded model config: "
        f"model_id={servicer.current_config['model_id']}, "
        f"layer_split={servicer.current_config['layer_split']}, "
        f"torch_dtype={servicer.current_config['torch_dtype']}"
    )

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=options)
    llama_ready_pb2_grpc.add_ShardServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{SERVER_HOST}:{SERVER_PORT}")
    server.start()
    print(f"Shard02 ready server listening on {SERVER_HOST}:{SERVER_PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()

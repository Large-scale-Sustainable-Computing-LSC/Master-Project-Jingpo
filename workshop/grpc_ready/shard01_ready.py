import csv
import time
from io import BytesIO

import grpc
import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer, LlamaModel

import llama_ready_pb2
import llama_ready_pb2_grpc


MODEL_SIZE_B = 8
EDGE_MODEL_PATH = "../jxu210_llama31_8b_bf16_f6"
SERVER_MODEL_ID = f"meta-llama/Meta-Llama-3.1-{MODEL_SIZE_B}B"
TEST_SET = f"meta-llama/Llama-3.1-{MODEL_SIZE_B}B-evals"
DATA_NAME = f"Llama-3.1-{MODEL_SIZE_B}B-evals__drop__details"

LAYER_TO_SPLIT = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 5087

MAX_NEW_TOKENS = 32
MAX_QUESTIONS = 256
DATA_TO_START = 0
STEP_SIZE = 26

DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
METRICS_CSV = f"ec_metrics_{LAYER_TO_SPLIT}.csv"


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def serialize_tensor(tensor):
    tensor = tensor.detach().cpu()
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return llama_ready_pb2.TensorData(
        tensor_bytes=buffer.getvalue(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
    )


def safe_torch_load(buffer, map_location):
    try:
        return torch.load(buffer, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(buffer, map_location=map_location)


def deserialize_tensor(tensor_data, device):
    tensor = safe_torch_load(BytesIO(tensor_data.tensor_bytes), map_location="cpu")
    return tensor.to(device)


def normalize_question_text(question):
    if isinstance(question, str):
        return question
    if isinstance(question, list):
        return "\n".join(str(item) for item in question)
    return str(question)


class ShardClient:
    def __init__(self, host, port):
        options = [
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = llama_ready_pb2_grpc.ShardServiceStub(self.channel)

    def init_connection(self):
        request = llama_ready_pb2.InitRequest(
            config=llama_ready_pb2.RuntimeConfig(
                model_id=SERVER_MODEL_ID,
                layer_split=LAYER_TO_SPLIT,
                torch_dtype=str(DTYPE),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
            )
        )
        return self.stub.InitConnection(request)

    def prefill(self, hidden, attention_mask, position_ids):
        request = llama_ready_pb2.PrefillRequest(
            hidden=serialize_tensor(hidden),
            attention_mask=serialize_tensor(attention_mask),
            position_ids=serialize_tensor(position_ids),
        )
        rpc_start = time.time()
        response = self.stub.PrefillStep(request)
        rpc_total = time.time() - rpc_start
        return {
            "next_token": deserialize_tensor(response.next_token, DEVICE),
            "server_infer_s": response.server_infer_s,
            "server_total_s": response.server_total_s,
            "rpc_total_s": rpc_total,
        }

    def decode(self, hidden):
        request = llama_ready_pb2.DecodeRequest(hidden=serialize_tensor(hidden))
        rpc_start = time.time()
        response = self.stub.DecodeStep(request)
        rpc_total = time.time() - rpc_start
        return {
            "next_token": deserialize_tensor(response.next_token, DEVICE),
            "server_infer_s": response.server_infer_s,
            "server_total_s": response.server_total_s,
            "rpc_total_s": rpc_total,
        }


def build_metrics_row(question_idx, question, input_token_count, output_text, token_metrics, total_latency_s):
    question_text = normalize_question_text(question)
    row = {
        "question_idx": question_idx,
        "input_token_count": input_token_count,
        "generated_token_count": len(token_metrics),
        "total_latency_s": total_latency_s,
        "output_text": output_text.replace("\n", "\\n"),
        "question_preview": question_text[:200].replace("\n", "\\n"),
    }

    if token_metrics:
        first = token_metrics[0]
        row["first_token_total_s"] = first["total_s"]
        row["first_token_edge_s"] = first["edge_infer_s"]
        row["first_token_cloud_s"] = first["cloud_infer_s"]
        row["first_token_transfer_s"] = first["transfer_s"]
        row["first_token_cloud_overhead_s"] = first["cloud_overhead_s"]
        row["first_token_edge_overhead_s"] = first["edge_overhead_s"]
    else:
        row["first_token_total_s"] = 0.0
        row["first_token_edge_s"] = 0.0
        row["first_token_cloud_s"] = 0.0
        row["first_token_transfer_s"] = 0.0
        row["first_token_cloud_overhead_s"] = 0.0
        row["first_token_edge_overhead_s"] = 0.0

    for follow_idx in range(1, MAX_NEW_TOKENS):
        token_number = f"{follow_idx:02d}"
        if follow_idx < len(token_metrics):
            metric = token_metrics[follow_idx]
            row[f"follow_token_{token_number}_total_s"] = metric["total_s"]
            row[f"follow_token_{token_number}_edge_s"] = metric["edge_infer_s"]
            row[f"follow_token_{token_number}_cloud_s"] = metric["cloud_infer_s"]
            row[f"follow_token_{token_number}_transfer_s"] = metric["transfer_s"]
            row[f"follow_token_{token_number}_cloud_overhead_s"] = metric["cloud_overhead_s"]
            row[f"follow_token_{token_number}_edge_overhead_s"] = metric["edge_overhead_s"]
        else:
            row[f"follow_token_{token_number}_total_s"] = ""
            row[f"follow_token_{token_number}_edge_s"] = ""
            row[f"follow_token_{token_number}_cloud_s"] = ""
            row[f"follow_token_{token_number}_transfer_s"] = ""
            row[f"follow_token_{token_number}_cloud_overhead_s"] = ""
            row[f"follow_token_{token_number}_edge_overhead_s"] = ""

    return row


def get_metrics_fieldnames():
    fieldnames = [
        "question_idx",
        "input_token_count",
        "generated_token_count",
        "total_latency_s",
        "first_token_total_s",
        "first_token_edge_s",
        "first_token_cloud_s",
        "first_token_transfer_s",
        "first_token_cloud_overhead_s",
        "first_token_edge_overhead_s",
    ]
    for follow_idx in range(1, MAX_NEW_TOKENS):
        token_number = f"{follow_idx:02d}"
        fieldnames.extend(
            [
                f"follow_token_{token_number}_total_s",
                f"follow_token_{token_number}_edge_s",
                f"follow_token_{token_number}_cloud_s",
                f"follow_token_{token_number}_transfer_s",
                f"follow_token_{token_number}_cloud_overhead_s",
                f"follow_token_{token_number}_edge_overhead_s",
            ]
        )
    fieldnames.extend(["question_preview", "output_text"])
    return fieldnames


def run_question(model, tokenizer, client, question):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attention_mask = inputs.attention_mask.to(DEVICE)
    position_ids = attention_mask.long().cumsum(dim=1) - 1
    generated_text = input_ids.clone()
    past_key_values = None

    client.init_connection()
    token_metrics = []

    with torch.no_grad():
        token_start = time.time()
        edge_start = time.time()
        outputs = model(
            input_ids=generated_text,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1]
        sync_device()
        edge_infer_s = time.time() - edge_start

        prefill_res = client.prefill(last_hidden, attention_mask, position_ids)
        next_token = prefill_res["next_token"]
        generated_text = torch.cat([generated_text, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        position_ids = torch.tensor([[generated_text.shape[1] - 1]], device=DEVICE, dtype=torch.long)
        total_s = time.time() - token_start
        cloud_overhead_s = prefill_res["server_total_s"] - prefill_res["server_infer_s"]
        transfer_s = prefill_res["rpc_total_s"] - prefill_res["server_total_s"]
        edge_overhead_s = total_s - edge_infer_s - prefill_res["rpc_total_s"]
        token_metrics.append(
            {
                "token_index": 0,
                "total_s": total_s,
                "edge_infer_s": edge_infer_s,
                "cloud_infer_s": prefill_res["server_infer_s"],
                "transfer_s": transfer_s,
                "cloud_overhead_s": cloud_overhead_s,
                "edge_overhead_s": edge_overhead_s,
            }
        )

        for _ in range(MAX_NEW_TOKENS - 1):
            token_start = time.time()
            edge_start = time.time()
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1]
            sync_device()
            edge_infer_s = time.time() - edge_start

            decode_res = client.decode(last_hidden)
            next_token = decode_res["next_token"]

            generated_text = torch.cat([generated_text, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            position_ids = torch.tensor([[generated_text.shape[1] - 1]], device=DEVICE, dtype=torch.long)
            total_s = time.time() - token_start
            cloud_overhead_s = decode_res["server_total_s"] - decode_res["server_infer_s"]
            transfer_s = decode_res["rpc_total_s"] - decode_res["server_total_s"]
            edge_overhead_s = total_s - edge_infer_s - decode_res["rpc_total_s"]
            token_metrics.append(
                {
                    "token_index": len(token_metrics),
                    "total_s": total_s,
                    "edge_infer_s": edge_infer_s,
                    "cloud_infer_s": decode_res["server_infer_s"],
                    "transfer_s": transfer_s,
                    "cloud_overhead_s": cloud_overhead_s,
                    "edge_overhead_s": edge_overhead_s,
                }
            )

    output_text = tokenizer.decode(generated_text[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text, token_metrics, int(input_ids.shape[1])


def main():
    print("Loading edge model ...")
    print(f"Using device: {DEVICE}")

    model = LlamaModel.from_pretrained(EDGE_MODEL_PATH, torch_dtype=DTYPE)
    available_layers = len(model.layers)
    if LAYER_TO_SPLIT <= 0 or LAYER_TO_SPLIT > available_layers:
        raise ValueError(
            f"LAYER_TO_SPLIT must be in [1, {available_layers}] for edge model {EDGE_MODEL_PATH}, "
            f"got {LAYER_TO_SPLIT}"
        )
    model.layers = model.layers[:LAYER_TO_SPLIT]
    model.norm = nn.Identity()
    model.eval()
    model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(EDGE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(TEST_SET, name=DATA_NAME, split="latest")
    client = ShardClient(TARGET_HOST, TARGET_PORT)

    rows = []
    fieldnames = get_metrics_fieldnames()

    start = time.time()
    for idx in range(DATA_TO_START, DATA_TO_START + (MAX_QUESTIONS * STEP_SIZE), STEP_SIZE):
        question = normalize_question_text(dataset["input_final_prompts"][idx])
        print(f"\nProcessing question {idx} ...")
        question_start = time.time()
        response, token_metrics, input_token_count = run_question(model, tokenizer, client, question)
        total_latency_s = time.time() - question_start
        print(f"Response: {response}")

        rows.append(
            build_metrics_row(
                question_idx=idx,
                question=question,
                input_token_count=input_token_count,
                output_text=response,
                token_metrics=token_metrics,
                total_latency_s=total_latency_s,
            )
        )

    with open(f"./orin_met/{METRICS_CSV}", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_time = time.time() - start
    total_generated_tokens = sum(row["generated_token_count"] for row in rows)
    first_values = [row["first_token_total_s"] for row in rows]
    following_values = []
    for row in rows:
        for follow_idx in range(1, MAX_NEW_TOKENS):
            value = row[f"follow_token_{follow_idx:02d}_total_s"]
            if value != "":
                following_values.append(float(value))

    overall_tps = total_generated_tokens / total_time if total_time > 0 else 0.0
    avg_first = sum(first_values) / len(first_values) if first_values else 0.0
    avg_following = sum(following_values) / len(following_values) if following_values else 0.0

    print("\n======== Finished (grpc_ready) ========")
    print(f"Edge model path: {EDGE_MODEL_PATH}")
    print(f"Server model id: {SERVER_MODEL_ID}")
    print(f"Split layer: {LAYER_TO_SPLIT}")
    print(f"Questions: {len(rows)}")
    print(f"Overall speed: {overall_tps:.4f} tokens/sec")
    print(f"First token latency: {avg_first:.4f} s")
    print(f"Following token latency: {avg_following:.4f} s")
    print(f"Metrics CSV: {METRICS_CSV}")


if __name__ == "__main__":
    main()

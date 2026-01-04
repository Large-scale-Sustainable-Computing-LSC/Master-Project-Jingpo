This repository contains the prototype code used for the **single edge–cloud evaluation** in my master’s thesis on collaborative LLM inference with model sharding. It is a research implementation rather than a fully generalized framework, but it already supports **end-to-end collaboration between two devices** (one edge node and one cloud node) to run sharded inference experiments.

## Abstract

Large language models (LLMs) are increasingly accessed from edge devices such as phones and laptops, yet most deployments still follow either a cloud-only or an edge-only pattern, each with clear limitations in terms of latency, privacy, and hardware utilization. This thesis investigates an alternative design based on edge–cloud collaborative inference, which uses model sharding to split a decoder-only Transformer model across heterogeneous edge and cloud devices and executes, and executes the resulting shards using pipeline parallelism. The framework includes optimized sharding layouts that remove redundant computation, a configuration module for per-request partitioning, and a cloud-side scheduling interface intended to fill pipeline bubbles and handle heterogeneous edge configurations. We design and implement a prototype edge–cloud system that shards a pre-trained LLM across an edge device and a cloud GPU server, connected over a residential wide-area network. Through extensive experiments, we quantify the structural overhead of sharding, the impact of cut-layer choices on latency, memory footprint, and communication cost, and the conditions under which collaboration outperforms edge-only and cloud-only baselines. Our results show that optimized sharding can keep additional overhead negligible on a server-grade GPU, while system-level effects dominate on weaker edge hardware. For models that already fit comfortably on the edge device, collaboration yields little additional latency benefit; however, for larger models that exceed edge memory capacity, our scheme enables practical inference with only moderate extra end-to-end latency while preserving privacy by keeping user input tokens and generated tokens on the edge and offloading only intermediate activations to the cloud. These findings provide concrete design guidelines for future edge–cloud LLM deployments on heterogeneous hardware.

## Dependencies

| Name         | Version | Purpose                                                      |
| ------------ | ------: | ------------------------------------------------------------ |
| torch        |   2.6.0 | Core tensor computation and inference runtime (PyTorch).     |
| transformers |  4.51.1 | Hugging Face model definitions and inference utilities.      |
| accelerate   |   1.6.0 | Utilities for device placement and distributed/accelerated inference. |
| tokenizers   |  0.21.1 | Fast tokenization backend used by Transformers.              |
| safetensors  |   0.5.3 | Safe and efficient model weight format.                      |
| grpcio       |  1.71.0 | gRPC communication for two-device (edge–cloud) collaboration. |
| protobuf     |  5.29.4 | Protocol Buffers serialization (dependency of gRPC).         |
| Flask        |   3.1.0 | Lightweight HTTP service / control API (if enabled).         |
| requests     |  2.32.3 | HTTP client utilities.                                       |
| numpy        |   2.1.3 | Numerical computing utilities.                               |
| pandas       |   2.2.3 | Experiment result processing and analysis.                   |
| datasets     |   3.5.1 | Dataset loading and preprocessing utilities.                 |



## Repository Contents

### `gRPC_test/`
Main prototype code for **two-node (edge–cloud) collaborative inference**.

- `shard01_grpc.py`: gRPC **client** (edge-side).
- `shard02_grpc.py`: gRPC **server** (cloud-side).

**Notes / limitations**

- Experiments were conducted with **LLaMA 3.2 1B** and **3B**.
- Model **download and switching are manual** (not automated in the codebase).
- Compatibility with other **LLaMA 3.2** variants has not been verified.
- The implementation is **likely not portable to other model families**, due to special handling of **normalization layers**.

### `test_results/`
Collected experimental outputs.

- `test_results.txt`: aggregated **latency statistics** (summary).
- Other subfolders/files: detailed outputs per **layer partition setting**, including **intermediate logs** and **memory usage** traces.

### `shardSim.py`
Local simulation script for the thesis **shard-opt** study.

- Provides an intuitive, local view of how **shard structure optimizations** affect the partitioning layout (as discussed in the thesis).
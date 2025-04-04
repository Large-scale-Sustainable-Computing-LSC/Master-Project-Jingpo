## Literature Study

### Hardware-Software Co-design

1. **~~Domain-Specific Accelerators~~** 

​	TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for 	Embeddings	[https://arxiv.org/abs/2304.01433]



2. **Compiler & IR Optimization**

   TVM, XLA, nGraph, Glow...

   

   TVM: An Automated End-to-End Optimizing Compiler for Deep Learning	

   [https://arxiv.org/abs/1802.04799]

​	***Target-independent XLA optimization using Reinforcement Learning***	[notes](#XLA optimization using Reinforcement Learning)

​	[https://arxiv.org/abs/2308.14364]

​	Operator Fusion in XLA: Analysis and Evaluation

​	[https://arxiv.org/abs/2301.13062]



3. **Algorithm-Hardware Co-Optimization**

​	**Quantization**

​	*TensorRT, PyTorch Quantization, GPTQ*

​	***SpinQuant: LLM quantization with learned rotations***	[notes](#SpinQuant: LLM quantization with learned rotations)

​	[https://arxiv.org/abs/2405.16406]



​	**Sparsity**

​	*DeepSpeed-Inference, Tenstorrent SDK*

​	***Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention***	[notes](#Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention)

​	[https://arxiv.org/abs/2502.11089]

​	

​	**Dynamic Computation**

​	*vLLM, NVIDIA Triton, Dynamic Route*







### Elastic Resource Management

1. **Dynamic Scaling Strategies** 

​	Tool: 		Kubernetes HPA、AWS Auto Scaling



2. **Load Prediction & Scheduling**



3. **Multi-Tenancy Resource Isolation**

   Tool: 		Kubernetes ResourceQuota、NVIDIA MIG

​	Application:    AWS SageMaker



​	Gandiva: Introspective Cluster Scheduling for Deep Learning

​	[https://www.usenix.org/conference/osdi18/presentation/xiao]



4. **Cloud-Edge-Device Coordination**

   Offloading,  Edge Caching, Federated Learning

   Tools:		TensorFlow Federated、AWS IoT Greengrass

   Application:     Llama Edge, TinyBERT





## XLA optimization using Reinforcement Learning

Pass sequence optimization. (Operator Fusion, Constant Folding, Dead Code Elimination...)

Redundant operations reduced by 10-20%. Memory usage reduced by 5-15%. 

ResNet-50 inference speed increased by 8-12%. 

Transformer training speed increased by 6-10% (depending on sequence length and batch size).

**Target-Independent RL Framework**: Focuses on optimizing pass order in the hardware-agnostic stage (HLO level) of XLA compilation, avoiding hardware-specific backend optimizations.

**State Representation via Graph Neural Network (GNN):** Encodes HLO graphs into vectors using GNNs, capturing topology, operator types, and tensor shapes.

**Limitations:** [1] Static reward metrics may not fully align with real hardware performance. [2] High computational cost due to RL training requiring extensive compilation and evaluation.

[Back](#Hardware-Software Co-design)





## SpinQuant: LLM quantization with learned rotations

https://github.com/facebookresearch/SpinQuant

With 4-bit quantization of weight, activation, and **KV-cache**, SpinQuant narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-2 7B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points. For LLaMA-3 8B models that are hard to quantize, SpinQuant reduces the gap to full precision by up to 45.1% relative to QuaRot.

[Back](#Hardware-Software Co-design)



## Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

NSA

- A dynamically sparse attention mechanism for long-context modeling.

- Transform raw long sequences into a more information-dense and computationally efficient representation. (3 Branches: Compression, Selection, Sliding Window)
- Hardware-aligned optimizations.

Results

- Up to 11.6× speedup in decoding on 64k-length sequences.
- Up to 9.0× speedup in forward propagation and 6.0× in backward propagation.

[Back](#Hardware-Software Co-design)

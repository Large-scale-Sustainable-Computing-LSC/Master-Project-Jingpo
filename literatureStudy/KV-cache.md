Question:

1. ~~Many LLM service providers save users' history. Is there any connection between history records and kv cache?~~ 
2. ~~How can we optimize the kv cache to save VRAM? Is it possible to transfer parts of the kv cache to main memory or SSD? Will the optimized high-parameter model still perform better than the low-parameter model?~~



### Keep the Cost Down: A Review on Methods to Optimize LLM' s KV-Cache Consumption

https://arxiv.org/abs/2407.18003

| **Category**          | **Key Techniques**                                           | **Advantages**                                            | **Challenges**                                       |
| --------------------- | ------------------------------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------- |
| **Quantization**      | Compress Key/Value vectors to low-precision (e.g., INT8/4) representations. | Directly reduces memory usage by 50%-75%.                 | Potential quality degradation due to precision loss. |
| **Pruning**           | Remove unimportant Key/Value vectors based on importance scores (e.g., attention weights, gradients). | Preserves critical information while reducing cache size. | Requires robust and efficient scoring mechanisms.    |
| **Sparsification**    | Retain only a subset of Key/Value vectors (e.g., every k-th token). | Simple implementation, linear memory reduction.           | May disrupt local context dependencies.              |
| ***Dynamic Caching*** | Adjust caching strategy dynamically based on context (e.g., cache only user queries and recent responses). | Task-adaptive memory savings.                             | Requires intelligent cache update rules.             |
| **Hybrid Methods**    | Combine multiple techniques (e.g., quantization + pruning).  | Balances trade-offs for better overall performance.       | Increased implementation complexity.                 |



### LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

https://arxiv.org/abs/2503.19950

**Core Idea**
LogQuant introduces a log-distributed 2-bit quantization method for KV cache in LLMs, prioritizing the preservation of critical features through:

- **Log-based importance scoring**: Amplifies mid-to-large magnitude values via logarithmic transformation to identify semantically significant tokens.
- **Dynamic Bucketing**: the KVs are divided into four different groups, and the groups with higher importance have finer quantization granularity.
- **Error compensation**: Propagates quantization residuals to subsequent layers to mitigate cumulative errors.

**Results**

- **Memory reduction**: Achieves 87.5% KV cache compression (e.g., 16GB → 2GB) with 2-bit precision.
- **Accuracy preservation**: Limits perplexity increase to <1% vs. FP16 baselines on LLaMA/GPT-3, outperforming existing 2-bit methods by 2–4% in perplexity.
- **Long-context robustness**: Reduces accuracy drop by >50% on 32k-token sequences compared to prior methods like H2O.

**Future Directions**

- **Hybrid precision**: Combining 2-bit LogQuant with higher-precision layers for critical tokens.

- **Sparsity integration**: Joint optimization with token sparsity to further compress memory.

  

### Unifying KV Cache Compression for Large Language Models with LeanKV

https://arxiv.org/abs/2412.03131

**Hetero-KV Quantization**

**Key:** responsible for calculating attention weights (determining the importance of different tokens), sensitive to quantization noise, and requires higher precision (such as 8-bit).

*Group-wise Quantization, reduce quantization error.*

**Value:** directly involved in output calculations, has a high tolerance for quantization noise, and can use lower precision (such as 4-bit).

*Dynamic Range Quantization, adapting to different input distributions.*

...



### ZeroMerge: Parameter-Free KV Cache Compression for Memory-Efficient Long-Context LLMs

https://arxiv.org/abs/2503.10714



### InfLLM: Training-Free Long-Context Extrapolation for LLMs with an Efficient Context Memory

https://arxiv.org/html/2402.04617v2





Offloading...
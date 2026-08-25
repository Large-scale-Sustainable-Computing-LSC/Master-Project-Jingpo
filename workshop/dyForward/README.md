# Dynamic Forward

## Supported Models

- `meta-llama/Llama-3.1-8B`
- `meta-llama/Llama-3.2-3B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`

## Supported Parameters

```python
dynamic_forward(
    model,
    hidden_states,
    start_layer,
    model_type="auto",
    end_layer=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=True,
    cache_position=None,
    apply_final_norm=True,
)
```

`model_type="auto"` detects the backend from `model.config.model_type`. Use
`"llama"` or `"qwen3_5"` to specify it manually.

Pass the normal 2D `attention_mask`; `dynamic_forward` builds the causal mask
internally.

## Example

```python
from dynamic_forward import dynamic_forward

input_embeds = model.model.embed_tokens(input_ids)

prefix_outputs = dynamic_forward(
    model,
    input_embeds,
    start_layer=0,
    end_layer=k,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=prefix_cache,
    cache_position=cache_position,
    apply_final_norm=False,
)
prefix_cache = prefix_outputs.past_key_values

suffix_outputs = dynamic_forward(
    model,
    prefix_outputs.last_hidden_state,
    start_layer=k,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=suffix_cache,
    cache_position=cache_position,
)
suffix_cache = suffix_outputs.past_key_values

logits = model.lm_head(suffix_outputs.last_hidden_state[:, -1:, :])
```

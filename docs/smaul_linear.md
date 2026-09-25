# smaul_linear.py

The model core: `LinearConfig` and `SmaulLinear`, a linear-attention model with SwiGLU
FFN/MoE blocks and tiled E4M3 FP8 weights (see [rqt.md](rqt.md) for the quantization
scheme and [cpu.md](cpu.md) for execution).

## Loading a checkpoint

```python
from smaul_linear import SmaulLinear
model = SmaulLinear.from_pretrained("./runs/linear")
```

## Config

`LinearConfig` is a dataclass saved as `config.json`:

| Field | Default | Meaning |
|---|---|---|
| `vocab_size` | `32000` | vocabulary size |
| `d_model` | `512` | model width; must be divisible by `n_heads` |
| `n_layer` | `8` | block count |
| `n_heads` | `8` | linear-attention head count |
| `ffn_mult` | `2.5` | FFN hidden width multiplier |
| `eps` | `1e-6` | RMSNorm epsilon |
| `tile` | `64` | FP8 quantization tile width |
| `is_moe` | `False` | use `SwiFFN_MoE` instead of `SwiFFN` |
| `num_experts` | `1` | expert count (MoE) |
| `num_experts_per_tok` | `1` | active experts per token (MoE) |

## Architecture

- **`LinearAttention`** -- Q/K/V/O projections are `FP8Linear`. Queries and keys pass
  through an `elu + 1` feature map with normalized keys; a recurrent state `(S, z)`
  accumulates `k^T v` outer products per timestep, so memory is `O(D^2)` per head
  instead of `O(T^2)` in sequence length.
- **`SwiFFN`** -- gated feed-forward (`silu(gate(x)) * up(x)` through `down`), all
  three projections `FP8Linear`.
- **`SwiFFN_MoE`** -- one `SwiFFN` per expert plus a softmax router; the top-k experts
  per token are renormalized and only tokens routed to an expert are computed by it.
- **`Block`** -- RMSNorms around attention and FFN with residual adds. While training
  with gradients enabled, attention and FFN run under gradient checkpointing.
- **`SmaulLinear`** -- bf16 embeddings, `n_layer` blocks, final norm + head. Forward
  returns `(logits, loss)`; with labels, loss is cross-entropy with `ignore_index=-100`
  (the `dataset.py` padding/mask convention).

## Persistence

`save_pretrained(dir)` writes `model.safetensors` + `config.json`;
`from_pretrained(dir)` loads them back with `strict=True`.

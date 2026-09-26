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
| `precision` | `fp8` | `fp8` (tiled E4M3) or `fp32` (plain) linear weights |

`precision` is validated on construction and persisted in `config.json`, so
checkpoints self-describe. In `fp32` mode every projection is a plain FP32 linear with
the same float-compute/input-dtype behavior as `FP8Linear`; the attention core,
Lion, inference, and export paths are shared.

## Architecture

- **`LinearAttention`** -- Q/K/V/O projections are `FP8Linear`. Queries and keys pass
  through an `elu + 1` feature map; the recurrent core (`_LinearAttnFn`) normalizes
  keys per step, then accumulates FP32 state `(S, z)` causally with no softmax and no
  QK^T materialization. The core runs a native AVX1 kernel when available
  (`kernel/attn_cpu.cpp`, same Ivy Bridge-safe flags as the FP8 kernels) and falls back to
  the pure-torch `_attn_reference` otherwise; gradients flow through a native
  two-pass backward with the same fallback.
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

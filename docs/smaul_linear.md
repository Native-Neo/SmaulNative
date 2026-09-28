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
| `architecture` | `plain` | `plain` (dense) or `rawr` (graph-sparse FFN + head). The train/infer CLIs default to `rawr`; the code default stays `plain` so library callers and legacy checkpoints are unaffected. |
| `embedding_storage` | `ram` | `ram` (`nn.Embedding`) or `mmap` (file-backed) |
| `rawr_sparsity` | `0.5` | Rawr: fraction of hidden/head connections omitted. Really a column-count knob — see `train.md` |
| `rawr_min_degree` | `4` | Rawr: connectivity floor per token (>= 1) |
| `rawr_graph_hash` / `rawr_edge_count` | `""` / `0` | recorded on save, verified on load; a mismatched graph is refused |

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
- **`RawrFFN` / `SparseLinear`** (`architecture="rawr"`) -- the FFN and LM head project
  through graph-derived column sets from `rawr_graph.hidden_cols`, stored as an
  `[out_f, K]` `values` parameter with the indices in a non-persistent `cols` buffer (rebuilt
  from `rawr_graph.json` on load). The forward and `d/dx` are dispatched as CSR sparse
  products (`_SparseLinearFn`), so only the nonzeros are visited; `d/dvalues` goes through
  `backend.sparse_grad_v`, which fuses the gathered-activation reduction into one pass.
  `RawrFFN.gate`/`up` share one `cols` tensor, since they have identical shapes.
  **The attention projections are not sparsified** — see the `compute_profile` note below.
- **`Block`** -- RMSNorms around attention and FFN with residual adds. While training
  with gradients enabled the **FFN** runs under gradient checkpointing; attention
  deliberately does not. Checkpointing it re-runs the q/k/v/o projections in backward
  (cost ~rows*d^2) to avoid holding Q, K, V, Y and DEN, which is only 5*rows*d*4 bytes —
  a scale-invariant bad trade at every model size (measured 1124.6 -> 899.1 ms per block
  step, 1.25x, for 5 MiB/layer).
- **`SmaulLinear`** -- bf16 embeddings, `n_layer` blocks, final norm + head. Forward
  returns `(logits, loss)`; with labels, loss is cross-entropy with `ignore_index=-100`
  (the `dataset.py` padding/mask convention).

## What the model actually executes

`SmaulLinear.compute_profile()` reports the per-token multiply-accumulates actually
performed, by walking the built model. It is the counterpart to `RawrGraph.stats()`, and it
exists because the two disagree sharply: the graph's `sparsity` describes the `vocab x vocab`
token-edge space, not the model. At the 32M preset (V8000, d512, 8 layers, ffn 2.5) the graph
reports 99.91% while **97.7% of the arithmetic the model performs is dense**, because the
graph only drives the FFN and the LM head — the attention projections are dense `FP8Linear`
and dominate. `train.py` prints both at startup and `rawr_graph.print_stats` labels the graph
figure "NOT the model" for the same reason.

## Incremental decoding

`SmaulLinear.prefill(idx)` / `SmaulLinear.step(idx, states)` decode one token at a time from
the carried linear-attention state, replacing a re-forward of the whole prefix per generated
token (2.4x-21.5x for generation; see `inference.md`). `inference.py` drives them; training
never calls them, and `Block.forward` remains the training path.

## Persistence

`save_pretrained(dir)` writes `model.safetensors` + `config.json` (plus `rawr_graph.json` and
`embeddings.dat` for the rawr/mmap combinations); `from_pretrained(dir)` loads them back with
`strict=True`.

# rwkv_x_core.py

The model core: `RWKVXConfig`, `RWKVXModel`, and `config_for_target_params`. It implements a CPU-capable
RWKV-7 + MOBA hybrid.

## Loading a checkpoint

```python
from rwkv_x_core import RWKVXModel
model = RWKVXModel.from_pretrained("./RWKV-X-256M")
```

## Architecture

- **`RWKV_Tmix_x070`** -- RWKV-7 TimeMix with value-residual, gated decay, and a recurrent state
  `(B,H,N,N)`. Training uses a finite BPTT window; inference can carry state across chunks.
- **`RWKV_CMix_x070`** -- dense Channel-Mix FFN using `relu(key(x))^2`, with `C -> 4C -> C` projections.
- **`MOBABlock`** -- sparse-attention blocks interleaved with RWKV blocks. CPU uses full causal
  scaled-dot-product attention, so its cost is O(T²) for sequence length T.
- **Interleaving** -- MOBA blocks are distributed among the RWKV blocks.
- **Weights** -- embeddings and the output head are separate, matching the upstream RWKV layout.

## Config & sizing

- **`RWKVXConfig`** is a dataclass saved in `config.json`.
- **`config_for_target_params(target, vocab_size, n_embd, n_moba_layer, head_size)`** searches for a
  layer count that approaches the requested parameter count at the selected width.
- `n_embd` must be divisible by `head_size`.
- MOBA configuration must leave at least one RWKV layer; the automatic sizing helper clamps the MOBA
  count accordingly.
- Native CPU WKV supports head sizes up to 128.

## Persistence & interop

- **`save_pretrained()` / `from_pretrained()`** use an HF-style directory containing `config.json` and
  `model.safetensors`, with the tokenizer and upstream-compatible weights exported alongside them.
- **`upstream_compatible_state_dict()`** maps keys to upstream RWKV-X naming for compatible inference
  workflows.

## CPU execution

When `train.py --cpu` is used, `cpu.configure()` installs the native C++ WKV implementation. The native
kernel handles forward and backward passes in float32 and parallelizes independent batch/head work.

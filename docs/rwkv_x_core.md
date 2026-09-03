# rwkv_x_core.py

The model itself -- `RWKVXConfig`, `RWKVXModel`, `config_for_target_params`. Not run directly;
imported by everything else. A CPU-pure-PyTorch RWKV-7 + MOBA hybrid (no CUDA kernel).

## Loading a checkpoint for your own script

```python
from rwkv_x_core import RWKVXModel
model = RWKVXModel.from_pretrained("./RWKV-X-256M")   # reads config.json + model.safetensors
```

## Architecture

- **`RWKV_Tmix_x070`** (`rwkv_x_core.py:76`) -- RWKV-7 TimeMix with value-residual, gated decay
  (`w_eff = exp(-0.606531 * sigmoid(w0 + g))`, verified equal to the real kernel's
  `exp(-exp(w_raw))`), and an O(1)-per-token recurrent state `(B,H,N,N)` for
  streaming/unlimited-context inference (training uses a finite BPTT window). On CPU the
  recurrence is a plain per-timestep loop (`rwkv_x_core.py:209`).
- **`RWKV_CMix_x070`** (`rwkv_x_core.py:232`) -- the dense Channel-Mix FFN: `relu(key(x))^2`
  through a `C -> 4C` expand then `4C -> C` contract. `key` (bias-free) is what QAT targets.
- **`MOBABlock`** (`rwkv_x_core.py:312`) -- sparse-attention blocks interleaved among RWKV blocks.
  On CPU, attention always falls back to full causal `scaled_dot_product_attention` (upstream's
  block-sparse path needs a GPU-only varlen flash-attn kernel), so it's correct but O(T^2) for
  long sequences.
- **Interleaving** (`rwkv_x_core.py:367`) -- MOBA blocks are spread evenly among RWKV blocks the
  same way upstream's `RWKVHybrid` does: `self._order` is a list of `("rwkv", i) | ("moba", i)`
  tuples the forward loop walks through.
- **Tied weights**: embeddings and the output head are *not* tied (separate `head` Linear,
  `rwkv_x_core.py:43`), matching upstream RWKV.

## Config & sizing

- **`RWKVXConfig`** is a plain dataclass saved/loaded as `config.json`
  (`rwkv_x_core.py:17`). `is_moe`/`num_experts`/`num_experts_per_tok` mark a `merge_moe.py`
  upcycled checkpoint.
- **`config_for_target_params(target, vocab_size, n_embd, n_moba_layer, head_size)`**
  (`rwkv_x_core.py:56`) searches `n_layer` (4..80) to get as close to `target_params` as possible
  at a fixed width, using `approx_param_count()` (`rwkv_x_core.py:41`). It requires
  `n_embd % head_size == 0`. `train.py` calls it with its fixed 256M target; you can call it
  directly with any target.

## Persistence & interop

- **`save_pretrained()`/`from_pretrained()`** (`rwkv_x_core.py:460`) give the HF-style dir:
  `config.json` + `model.safetensors`, plus a bundled `tokenizer.json` and an
  `rwkvx_upstream_compatible.pth`.
- **`upstream_compatible_state_dict()`** (`rwkv_x_core.py:440`) re-prefixes keys to upstream
  RWKV-X naming (`rwkv.emb.weight`, `rwkv.blocks.N.*`, `moba.*`), so the `.pth` can be loaded by
  the `rwkv-x` pip package's inference path on a CUDA box without re-exporting. The pure-RWKV
  blocks renumber as `blocks.N` (moba blocks are separate).

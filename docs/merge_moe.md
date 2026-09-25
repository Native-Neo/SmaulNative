# merge_moe.py

Merges one base SmaulLinear checkpoint plus one or more branch checkpoints (same
architecture, e.g. trained on different domains) into one SwiGLU-MoE model: each
branch's block FFN becomes one expert; attention, embeddings, norms, and head come
from the base; the router is freshly initialized.

## Run it

```bash
python merge_moe.py --base ./runs/base --branches ./runs/branch1 ./runs/branch2 \
    --out ./runs/moe --top_k 1
```

| Flag | Default | What it does |
|---|---|---|
| `--base` | *(required)* | checkpoint dir providing everything except the FFN experts |
| `--branches` | *(required)* | one or more checkpoint dirs; each becomes one expert (space-separated) |
| `--out` | *(required)* | output dir for the merged MoE model |
| `--top_k` | `1` | experts activated per token |

## Requirements

- Base and branch checkpoints must share `vocab_size`, `d_model`, `n_layer`, and
  `n_heads` -- a mismatch raises `ValueError`. Tokenizers are *not* unioned, so the
  vocabularies must already agree.
- `top_k` must be `>= 1` and `<=` the number of experts.
- Every checkpoint dir needs `config.json` + `model.safetensors` (any `train.py` run
  produces both).

## How it works

1. Builds a `LinearConfig` from the base with `is_moe=True`,
   `num_experts = len(branches)`, `num_experts_per_tok = top_k`.
2. Copies every shared tensor straight from the base, skipping `.ffn.` and `.gate.`
   keys; a missing key or shape mismatch raises `ValueError`.
3. Fills each expert from the corresponding branch's `.ffn.` tensors
   (`.ffn.` -> `.ffn.experts.{e}.`); the router gate keeps its random init.
4. Writes `config.json` + `model.safetensors` plus a `merge_config.json` recording the
   base, branches, expert count, and top_k.

The result loads like any other checkpoint:

```python
from smaul_linear import SmaulLinear
model = SmaulLinear.from_pretrained("./runs/moe")
```

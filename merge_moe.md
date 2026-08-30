# merge_moe.py

Combines a base checkpoint with one or more branch checkpoints (same architecture, e.g. SFT'd on
different domains) into one **Channel-Mix MoE** model: each branch's FFN becomes an expert,
everything else is shared from the base. Also union-merges their tokenizers. This is this
project's own MoE-upcycling extension -- *not* part of upstream RWKV-X.

## Run it

```bash
python merge_moe.py --base ./RWKV-X-256M --branches ./sft_branch1 ./sft_branch2 --out ./RWKV-X-MoE --top_k 1
```

| Flag | Default | What it does |
|---|---|---|
| `--base` | *(required)* | checkpoint dir providing everything except the FFN experts |
| `--branches` | *(required)* | one or more checkpoint dirs; each becomes one expert (space-separated) |
| `--out` | *(required)* | output dir for the merged MoE model |
| `--top_k` | `1` | experts activated per token |

## Requirements

- All base + branch checkpoints must share `n_embd`, `n_layer`, `n_moba_layer`, and `head_size` --
  vocab size may differ (tokenizers get unioned, embeddings/head auto-resized). A mismatch raises
  a clear error (`assert_compatible`, `merge_moe.py:25`).
- Every checkpoint dir needs a bundled `tokenizer.json` (any `train.py` run does this
  automatically, `merge_moe.py:47`).
- Result loads via `RWKVXModel.from_pretrained("./RWKV-X-MoE")` like any other checkpoint.

## How it works

- **Tokenizer union** (`merge_tokenizers`, `merge_moe.py:56`): the merged vocab keeps the base's
  ids, then appends each branch's new tokens/merges (`next_id` continues from base's max).
  Duplicate tokens/merges are skipped; a token string that maps to a *different* id across
  branches is a conflict where **base's id wins** (with a `[WARN]`, `merge_moe.py:110`).
- **Model merge** (`merge`, `merge_moe.py:136`):
  1. Base config, plus `is_moe: True`, `num_experts = len(branches)`, and
     `num_experts_per_tok = min(top_k, num_experts)` (`merge_moe.py:153`).
  2. Every non-FFN tensor is copied straight from base (`merge_moe.py:162`); `emb.weight` and
     `head.weight` are resized if the vocab grew, with new rows drawn from a small-normal
     distribution matched to existing stats (`resize_vocab_matrix`, `merge_moe.py:119`).
  3. Each expert's `ffn.key`/`ffn.value` are filled from the corresponding branch's FFN tensors
     (`merge_moe.py:173`); the router gate keeps the model's own random init.
- Writes `config.json` + `model.safetensors` + a unioned `tokenizer.json`, plus a
  `merge_config.json` metadata file recording the base, branches, expert count, and tokenizer-merge
  stats (`merge_moe.py:196`).

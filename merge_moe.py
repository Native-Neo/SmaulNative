#!/usr/bin/env python3
# merge_moe.py -- merges N RWKV-X checkpoints into one Channel-Mix MoE model (each branch's FFN
# becomes an expert; everything else shared from base) and unions their tokenizers (base ids kept,
# no duplicate tokens/merges, embeddings resized if vocab grows). Own MoE extension, not upstream RWKV-X.
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from safetensors.torch import load_file, save_file

from rwkv_x_core import RWKVXConfig, RWKVXModel


# Model checkpoint loading

def load_checkpoint(d: Path):
    cfg = RWKVXConfig.load(d / "config.json")
    sd = load_file(str(d / "model.safetensors"))
    return cfg, sd


def assert_compatible(base_cfg: RWKVXConfig, branch_cfg: RWKVXConfig, branch_path: Path):
    # vocab_size is intentionally excluded here: tokenizer merging below may legitimately grow it.
    # is_moe/num_experts are intentionally excluded too -- both dense and MoE branches are
    # supported (see branch_expert_state_dicts below); they just contribute different expert counts.
    for field in ("n_embd", "n_layer", "n_moba_layer", "head_size"):
        bv, ov = getattr(base_cfg, field), getattr(branch_cfg, field)
        if bv != ov:
            raise ValueError(
                f"{branch_path}: {field}={ov} does not match base {field}={bv}. "
                "All branches must share the exact same architecture to merge."
            )


def cmix_prefixes(cfg: RWKVXConfig) -> List[str]:
    """Every ffn.* prefix across both rwkv_blocks and moba_blocks -- both use RWKV_CMix_x070
    (dense) or RWKV_CMix_MoE (already-merged), depending on cfg.is_moe."""
    prefixes = [f"rwkv_blocks.{i}.ffn" for i in range(cfg.n_layer - cfg.n_moba_layer)]
    prefixes += [f"moba_blocks.{i}.ffn" for i in range(cfg.n_moba_layer)]
    return prefixes


def branch_expert_state_dicts(cfg: RWKVXConfig, sd: Dict[str, torch.Tensor], prefix: str
                               ) -> List[Dict[str, torch.Tensor]]:
    """Returns this branch's contribution at `prefix` as a list of one-expert-worth state dicts
    (relative to a single RWKV_CMix_x070, i.e. keys like 'key.weight'/'value.weight').
    - Dense branch (cfg.is_moe == False): one expert -- the branch's own ffn.{key,value}.weight.
    - MoE branch (cfg.is_moe == True): cfg.num_experts experts -- each of its existing
      ffn.experts.{i}.{key,value}.weight, gate dropped (a fresh router is always used for
      the new merge; averaging/reusing old gates across a different expert count isn't
      well-defined)."""
    src_prefix = f"{prefix}."
    if not cfg.is_moe:
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith(src_prefix):
                out[k[len(src_prefix):]] = v
        return [out] if out else []

    experts: List[Dict[str, torch.Tensor]] = [dict() for _ in range(cfg.num_experts)]
    experts_prefix = src_prefix + "experts."
    for k, v in sd.items():
        if not k.startswith(experts_prefix):
            continue  # skips ffn.gate.weight too -- old router intentionally not carried over
        rest = k[len(experts_prefix):]
        e_id_str, _, suffix = rest.partition(".")
        if not e_id_str.isdigit():
            continue
        e_id = int(e_id_str)
        if 0 <= e_id < cfg.num_experts:
            experts[e_id][suffix] = v
    return [e for e in experts if e]


# Tokenizer merge -- union vocab/merges, base ids preserved, no duplicates

def _load_tokenizer_json(d: Path) -> dict:
    path = d / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{d} has no tokenizer.json bundled with it. "
            "Re-save the checkpoint with the updated train.py (which bundles the tokenizer "
            "automatically), or copy tokenizer.json into this checkpoint dir by hand."
        )
    return json.loads(path.read_text())


def merge_tokenizers(base_dir: Path, branch_dirs: List[Path]) -> Tuple[dict, int, dict]:
    """Returns (merged_tokenizer_json_dict, merged_vocab_size, stats)."""
    base_tok = _load_tokenizer_json(base_dir)
    base_vocab: Dict[str, int] = base_tok["model"]["vocab"]
    base_merges: List = base_tok["model"]["merges"]

    merged_vocab: Dict[str, int] = dict(base_vocab)
    merged_merges: List = list(base_merges)
    merges_seen = {tuple(m) if isinstance(m, list) else m for m in merged_merges}
    next_id = max(merged_vocab.values(), default=-1) + 1

    added_tokens_total = 0
    added_merges_total = 0
    conflicting_branches = []

    for bd in branch_dirs:
        if bd.resolve() == base_dir.resolve():
            continue  # base merged with itself is trivially a no-op, skip explicitly
        branch_tok = _load_tokenizer_json(bd)
        branch_vocab: Dict[str, int] = branch_tok["model"]["vocab"]
        branch_merges: List = branch_tok["model"]["merges"]

        # conflict check: same token string mapped to a different id than base already has --
        # cannot safely union without breaking one side's ids, so we keep base's id and just warn
        # (matches the conservative "base wins" policy from the original tokenizer-merge design).
        for token, tid in branch_vocab.items():
            if token in merged_vocab:
                if merged_vocab[token] != tid:
                    conflicting_branches.append((str(bd), token, merged_vocab[token], tid))
                continue  # already present (from base or an earlier branch) -- no duplicate added
            merged_vocab[token] = next_id
            next_id += 1
            added_tokens_total += 1

        for m in branch_merges:
            key = tuple(m) if isinstance(m, list) else m
            if key in merges_seen:
                continue
            merges_seen.add(key)
            merged_merges.append(m)
            added_merges_total += 1

    merged_tok = dict(base_tok)
    merged_tok["model"] = dict(base_tok["model"])
    merged_tok["model"]["vocab"] = merged_vocab
    merged_tok["model"]["merges"] = merged_merges

    stats = {
        "base_vocab_size": len(base_vocab),
        "merged_vocab_size": len(merged_vocab),
        "added_tokens": added_tokens_total,
        "added_merges": added_merges_total,
        "id_conflicts": conflicting_branches,  # token strings where branch id != base id (base id kept)
    }
    if conflicting_branches:
        print(f"[WARN] {len(conflicting_branches)} token(s) had conflicting ids across branches; "
              f"base's id was kept for each. Examples: {conflicting_branches[:5]}")

    return merged_tok, len(merged_vocab), stats


# Embedding/head resize for a grown vocab

def resize_vocab_matrix(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    current = tensor.shape[0]
    if target_size <= current:
        return tensor
    out = torch.empty((target_size, *tensor.shape[1:]), dtype=tensor.dtype)
    out[:current].copy_(tensor)
    src = tensor.float()
    mean, std = src.mean(), src.std(unbiased=False)
    if not torch.isfinite(std) or std.item() <= 1e-12:
        std = torch.tensor(0.02)
    extra = torch.normal(mean=float(mean), std=float(std), size=(target_size - current, *tensor.shape[1:]))
    out[current:].copy_(extra.to(tensor.dtype))
    return out


# Model merge

def merge(base_dir: Path, branch_dirs: List[Path], out_dir: Path, top_k: int = 1):
    base_cfg, base_sd = load_checkpoint(base_dir)
    branches = []
    for bd in branch_dirs:
        cfg, sd = load_checkpoint(bd)
        assert_compatible(base_cfg, cfg, bd)
        branches.append((bd, cfg, sd))

    # Each branch contributes one expert per FFN if dense, or cfg.num_experts if it's itself
    # an already-merged MoE checkpoint -- so the new expert count is a sum, not just len(branches).
    per_branch_expert_counts = [cfg.num_experts if cfg.is_moe else 1 for (_, cfg, _) in branches]
    num_experts = sum(per_branch_expert_counts)
    branch_summary = ", ".join(f"{bd.name}:{n}" for (bd, _, _), n in zip(branches, per_branch_expert_counts))
    print(f"[MERGE] base={base_dir} (is_moe={base_cfg.is_moe}), {len(branches)} branch(es) "
          f"contributing {num_experts} total expert(s) ({branch_summary}), top_k={top_k}")

    print("[MERGE] merging tokenizers...")
    merged_tok_json, merged_vocab_size, tok_stats = merge_tokenizers(base_dir, branch_dirs)
    print(f"[MERGE] tokenizer: base_vocab={tok_stats['base_vocab_size']} "
          f"-> merged_vocab={tok_stats['merged_vocab_size']} "
          f"(+{tok_stats['added_tokens']} tokens, +{tok_stats['added_merges']} merge rules)")

    moe_cfg = RWKVXConfig(**{**base_cfg.__dict__, "is_moe": True, "vocab_size": merged_vocab_size,
                              "num_experts": num_experts, "num_experts_per_tok": min(top_k, num_experts)})
    moe_model = RWKVXModel(moe_cfg)
    out_sd = moe_model.state_dict()  # start from a fresh init, then overwrite with real weights

    prefixes = cmix_prefixes(base_cfg)

    # 1) copy every non-ffn tensor straight from base, resizing emb/head if vocab grew
    ffn_marker = ".ffn."
    for k, v in base_sd.items():
        if ffn_marker in k:
            continue
        if k in ("emb.weight", "head.weight") and v.shape[0] != merged_vocab_size:
            v = resize_vocab_matrix(v, merged_vocab_size)
        if k in out_sd and out_sd[k].shape == v.shape:
            out_sd[k] = v
        else:
            print(f"[WARN] shape/key mismatch for shared tensor {k}, keeping fresh init")

    # 2) fill experts from each branch, in order -- a dense branch supplies 1 expert, an MoE
    # branch supplies all of its existing experts (its own router is dropped; a fresh router
    # is always used post-merge since the expert count/identity has changed)
    for prefix in prefixes:
        e_id = 0
        for bd, cfg, sd in branches:
            branch_experts = branch_expert_state_dicts(cfg, sd, prefix)
            expected = cfg.num_experts if cfg.is_moe else 1
            if not branch_experts:
                print(f"[WARN] branch {bd} has no {prefix}.* tensors; {expected} expert slot(s) "
                      f"at {prefix} keep random init")
                e_id += expected
                continue
            for expert_sd in branch_experts:
                dst_prefix = f"{prefix}.experts.{e_id}."
                found_any = False
                for suffix, v in expert_sd.items():
                    dst_key = dst_prefix + suffix
                    if dst_key in out_sd and out_sd[dst_key].shape == v.shape:
                        out_sd[dst_key] = v
                        found_any = True
                if not found_any:
                    print(f"[WARN] branch {bd} expert at {prefix} had no matching tensors; "
                          f"expert {e_id} keeps random init")
                e_id += 1
        # router: leave at RWKVXModel's own random init (already in out_sd from moe_model construction)

    moe_model.load_state_dict(out_sd, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    moe_model.save_pretrained(out_dir)
    (out_dir / "tokenizer.json").write_text(json.dumps(merged_tok_json))

    meta = {
        "engine": "rwkv-x godfather merge_moe.py",
        "base_model": str(base_dir),
        "base_was_moe": base_cfg.is_moe,
        "branches": [str(b) for b in branch_dirs],
        "branch_expert_counts": {str(bd): n for (bd, _, _), n in zip(branches, per_branch_expert_counts)},
        "num_experts": num_experts,
        "top_k": moe_cfg.num_experts_per_tok,
        "tokenizer_merge": tok_stats,
        "note": "Custom MoE-upcycled Channel-Mix, requires rwkv_x_core.RWKVXModel to load "
                "(not compatible with upstream rwkv-x pip package). Routers from any MoE "
                "branch/base are not carried over -- this merge always initializes a fresh "
                "router over the new expert set and expects further training/fine-tuning.",
    }
    (out_dir / "merge_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] merged model -> {out_dir} ({moe_model.num_parameters()/1e6:.1f}M params, "
          f"vocab_size={merged_vocab_size}, num_experts={num_experts})")


def main():
    p = argparse.ArgumentParser(description="Merge N RWKV-X checkpoints (+ their tokenizers) into a Channel-Mix MoE model")
    p.add_argument("--base", required=True, type=str, help="base checkpoint dir (provides everything except FFN experts)")
    p.add_argument("--branches", required=True, nargs="+", type=str,
                    help="one or more checkpoint dirs, each becomes one expert (or, if a branch "
                         "is itself an MoE checkpoint, all of its existing experts)")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--top_k", type=int, default=1, help="experts activated per token")
    args = p.parse_args()

    merge(Path(args.base), [Path(b) for b in args.branches], Path(args.out), top_k=args.top_k)


if __name__ == "__main__":
    main()

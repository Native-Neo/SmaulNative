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
    for field in ("n_embd", "n_layer", "n_moba_layer", "head_size"):
        bv, ov = getattr(base_cfg, field), getattr(branch_cfg, field)
        if bv != ov:
            raise ValueError(
                f"{branch_path}: {field}={ov} does not match base {field}={bv}. "
                "All branches must share the exact same architecture to merge."
            )


def cmix_prefixes(cfg: RWKVXConfig) -> List[str]:
    prefixes = [f"rwkv_blocks.{i}.ffn" for i in range(cfg.n_layer - cfg.n_moba_layer)]
    prefixes += [f"moba_blocks.{i}.ffn" for i in range(cfg.n_moba_layer)]
    return prefixes


def branch_expert_state_dicts(cfg: RWKVXConfig, sd: Dict[str, torch.Tensor], prefix: str
                               ) -> List[Dict[str, torch.Tensor]]:
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
            continue
        rest = k[len(experts_prefix):]
        e_id_str, _, suffix = rest.partition(".")
        if not e_id_str.isdigit():
            continue
        e_id = int(e_id_str)
        if 0 <= e_id < cfg.num_experts:
            experts[e_id][suffix] = v
    return experts


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


def _merge_key(m) -> Tuple[str, str]:
    if isinstance(m, list) and len(m) == 2:
        return str(m[0]), str(m[1])
    if isinstance(m, str):
        parts = m.split(" ", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    raise ValueError(f"invalid BPE merge rule: {m!r}")


def _union_merges(tokenizers: List[dict]) -> List:
    """Build one BPE merge order that preserves every input tokenizer's ordering.

    Simply appending each branch's merge list changes BPE ranks and therefore changes
    tokenization. A topological merge preserves every tokenizer's pair ordering and fails
    if the input rankings contain an actual ordering conflict.
    """
    nodes: Dict[Tuple[str, str], object] = {}
    edges: Dict[Tuple[str, str], set] = {}
    indegree: Dict[Tuple[str, str], int] = {}
    order_hint: Dict[Tuple[str, str], int] = {}

    for tok_idx, tok in enumerate(tokenizers):
        prev = None
        for pos, raw in enumerate(tok["model"]["merges"]):
            key = _merge_key(raw)
            nodes.setdefault(key, raw)
            edges.setdefault(key, set())
            indegree.setdefault(key, 0)
            order_hint.setdefault(key, tok_idx * 10**9 + pos)
            if prev is not None and key not in edges[prev]:
                edges[prev].add(key)
                indegree[key] += 1
            prev = key

    ready = sorted((k for k, degree in indegree.items() if degree == 0), key=order_hint.get)
    result = []
    while ready:
        key = ready.pop(0)
        result.append(nodes[key])
        for nxt in sorted(edges[key], key=order_hint.get):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=order_hint.get)

    if len(result) != len(nodes):
        raise ValueError("tokenizer BPE merge orders conflict; no valid union preserves all tokenizer rankings")
    return result


def merge_tokenizers(base_dir: Path, branch_dirs: List[Path]) -> Tuple[dict, int, dict]:
    base_tok = _load_tokenizer_json(base_dir)
    base_vocab: Dict[str, int] = base_tok["model"]["vocab"]

    merged_vocab: Dict[str, int] = dict(base_vocab)
    next_id = max(merged_vocab.values(), default=-1) + 1

    added_tokens_total = 0
    conflicting_branches = []
    tokenizers = [base_tok]

    for bd in branch_dirs:
        if bd.resolve() == base_dir.resolve():
            continue
        branch_tok = _load_tokenizer_json(bd)
        tokenizers.append(branch_tok)
        branch_vocab: Dict[str, int] = branch_tok["model"]["vocab"]

        for token, tid in branch_vocab.items():
            if token in merged_vocab:
                if merged_vocab[token] != tid:
                    conflicting_branches.append((str(bd), token, merged_vocab[token], tid))
                continue
            merged_vocab[token] = next_id
            next_id += 1
            added_tokens_total += 1

    merged_merges = _union_merges(tokenizers)

    # Every merge operand must exist in the merged vocabulary. Otherwise the tokenizer
    # contains a rule that can never be applied correctly.
    for m in merged_merges:
        left, right = _merge_key(m)
        if left not in merged_vocab or right not in merged_vocab:
            raise ValueError(f"BPE merge references missing token(s): {left!r}, {right!r}")

    merged_tok = dict(base_tok)
    merged_tok["model"] = dict(base_tok["model"])
    merged_tok["model"]["vocab"] = merged_vocab
    merged_tok["model"]["merges"] = merged_merges

    stats = {
        "base_vocab_size": len(base_vocab),
        "merged_vocab_size": len(merged_vocab),
        "added_tokens": added_tokens_total,
        "added_merges": len(merged_merges) - len(base_tok["model"]["merges"]),
        "id_conflicts": conflicting_branches,
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
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    base_cfg, base_sd = load_checkpoint(base_dir)
    if base_cfg.is_moe:
        raise ValueError(
            "MoE base checkpoints are not supported because their experts would be omitted; "
            "use a dense base checkpoint and pass MoE checkpoints via --branches"
        )
    branches = []
    for bd in branch_dirs:
        cfg, sd = load_checkpoint(bd)
        assert_compatible(base_cfg, cfg, bd)
        branches.append((bd, cfg, sd))

    per_branch_expert_counts = [cfg.num_experts if cfg.is_moe else 1 for (_, cfg, _) in branches]
    num_experts = sum(per_branch_expert_counts)
    if num_experts < 1:
        raise ValueError("merge requires at least one expert")
    if top_k > num_experts:
        raise ValueError(f"top_k ({top_k}) cannot exceed the merged expert count ({num_experts})")
    branch_summary = ", ".join(f"{bd.name}:{n}" for (bd, _, _), n in zip(branches, per_branch_expert_counts))
    print(f"[MERGE] base={base_dir} (is_moe={base_cfg.is_moe}), {len(branches)} branch(es) "
          f"contributing {num_experts} total expert(s) ({branch_summary}), top_k={top_k}")

    print("[MERGE] merging tokenizers...")
    merged_tok_json, merged_vocab_size, tok_stats = merge_tokenizers(base_dir, branch_dirs)
    print(f"[MERGE] tokenizer: base_vocab={tok_stats['base_vocab_size']} "
          f"-> merged_vocab={tok_stats['merged_vocab_size']} "
          f"(+{tok_stats['added_tokens']} tokens, +{tok_stats['added_merges']} merge rules)")

    moe_cfg = RWKVXConfig(**{**base_cfg.__dict__, "is_moe": True, "vocab_size": merged_vocab_size,
                              "num_experts": num_experts, "num_experts_per_tok": top_k})
    moe_model = RWKVXModel(moe_cfg)
    out_sd = moe_model.state_dict()

    prefixes = cmix_prefixes(base_cfg)
    expected_expert_keys = {"key.weight", "value.weight"}

    # 1) copy every non-ffn tensor straight from base, resizing emb/head if vocab grew
    ffn_marker = ".ffn."
    for k, v in base_sd.items():
        if ffn_marker in k:
            continue
        if k in ("emb.weight", "head.weight") and v.shape[0] != merged_vocab_size:
            v = resize_vocab_matrix(v, merged_vocab_size)
        if k not in out_sd:
            raise ValueError(f"shared tensor {k} is missing from merged model")
        if out_sd[k].shape != v.shape:
            raise ValueError(f"shared tensor {k} shape mismatch: branch={tuple(v.shape)}, model={tuple(out_sd[k].shape)}")
        out_sd[k] = v

    # 2) fill experts from each branch. Every expected expert must provide the complete FFN.
    for prefix in prefixes:
        e_id = 0
        for bd, cfg, sd in branches:
            branch_experts = branch_expert_state_dicts(cfg, sd, prefix)
            expected = cfg.num_experts if cfg.is_moe else 1
            if len(branch_experts) != expected:
                raise ValueError(f"{bd}: {prefix} contains {len(branch_experts)} experts, expected {expected}")
            for expert_sd in branch_experts:
                missing = expected_expert_keys - set(expert_sd)
                extra = set(expert_sd) - expected_expert_keys
                if missing:
                    raise ValueError(f"{bd}: {prefix} expert {e_id} missing tensors: {sorted(missing)}")
                if extra:
                    raise ValueError(f"{bd}: {prefix} expert {e_id} has unexpected tensors: {sorted(extra)}")
                for suffix in expected_expert_keys:
                    dst_key = f"{prefix}.experts.{e_id}.{suffix}"
                    v = expert_sd[suffix]
                    if dst_key not in out_sd:
                        raise ValueError(f"merged model is missing expert tensor {dst_key}")
                    if out_sd[dst_key].shape != v.shape:
                        raise ValueError(f"{bd}: {prefix} expert {e_id} {suffix} shape mismatch: "
                                         f"branch={tuple(v.shape)}, model={tuple(out_sd[dst_key].shape)}")
                    out_sd[dst_key] = v
                e_id += 1
        if e_id != num_experts:
            raise ValueError(f"{prefix}: filled {e_id} experts, expected {num_experts}")

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

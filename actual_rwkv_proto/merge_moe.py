#!/usr/bin/env python3
########################################################################################################
# merge_moe.py
#
# Combines any number of RWKV-X checkpoints produced by train.py (same base architecture -- same
# n_layer/n_embd/n_moba_layer/vocab_size, e.g. one base pretrain + several SFT/domain fine-tunes)
# into a single Mixture-of-Experts checkpoint:
#
#   - Everything EXCEPT the Channel-Mix FFN (RWKV_CMix_x070) is taken from the base checkpoint and
#     shared: embeddings, RWKV-7 TimeMix, MOBA attention, layernorms, output head.
#   - Each branch's Channel-Mix FFN (key/value/x_k) becomes one expert.
#   - A learned router (blocks.N.ffn.gate.weight) is added per FFN, top-k routed.
#
# This "MoE-upcycled Channel-Mix" is THIS PROJECT'S OWN EXTENSION (rwkv_x_core.RWKV_CMix_MoE) --
# it is not part of upstream howard-hou/RWKV-X, which has no MoE support. A merged checkpoint from
# this script can only be loaded back with RWKVXModel (rwkv_x_core.py, is_moe=True in config.json),
# not with the real `rwkv-x` pip package.
########################################################################################################

import argparse
import json
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file, save_file

from rwkv_x_core import RWKVXConfig, RWKVXModel


def load_checkpoint(d: Path):
    cfg = RWKVXConfig.load(d / "config.json")
    sd = load_file(str(d / "model.safetensors"))
    return cfg, sd


def assert_compatible(base_cfg: RWKVXConfig, branch_cfg: RWKVXConfig, branch_path: Path):
    for field in ("n_embd", "n_layer", "n_moba_layer", "head_size", "vocab_size"):
        bv, ov = getattr(base_cfg, field), getattr(branch_cfg, field)
        if bv != ov:
            raise ValueError(
                f"{branch_path}: {field}={ov} does not match base {field}={bv}. "
                "All branches must share the exact same architecture to merge."
            )


def cmix_prefixes(cfg: RWKVXConfig) -> List[str]:
    """Every ffn.* prefix across both rwkv_blocks and moba_blocks -- both use RWKV_CMix_x070."""
    prefixes = [f"rwkv_blocks.{i}.ffn" for i in range(cfg.n_layer - cfg.n_moba_layer)]
    prefixes += [f"moba_blocks.{i}.ffn" for i in range(cfg.n_moba_layer)]
    return prefixes


def merge(base_dir: Path, branch_dirs: List[Path], out_dir: Path, top_k: int = 1):
    base_cfg, base_sd = load_checkpoint(base_dir)
    branches = []
    for bd in branch_dirs:
        cfg, sd = load_checkpoint(bd)
        assert_compatible(base_cfg, cfg, bd)
        branches.append((bd, sd))

    num_experts = len(branches)
    print(f"[MERGE] base={base_dir}, {num_experts} expert branches, top_k={top_k}")

    moe_cfg = RWKVXConfig(**{**base_cfg.__dict__, "is_moe": True,
                              "num_experts": num_experts, "num_experts_per_tok": min(top_k, num_experts)})
    moe_model = RWKVXModel(moe_cfg)
    out_sd = moe_model.state_dict()  # start from a fresh init, then overwrite with real weights

    prefixes = cmix_prefixes(base_cfg)

    # 1) copy every non-ffn tensor straight from base
    ffn_marker = ".ffn."
    for k, v in base_sd.items():
        if ffn_marker not in k:
            if k in out_sd and out_sd[k].shape == v.shape:
                out_sd[k] = v
            else:
                print(f"[WARN] shape/key mismatch for shared tensor {k}, keeping fresh init")

    # 2) fill each expert's ffn from the corresponding branch
    for prefix in prefixes:
        for e_id, (bd, sd) in enumerate(branches):
            src_prefix = f"{prefix}."
            dst_prefix = f"{prefix}.experts.{e_id}."
            found_any = False
            for k, v in sd.items():
                if k.startswith(src_prefix):
                    suffix = k[len(src_prefix):]
                    dst_key = dst_prefix + suffix
                    if dst_key in out_sd and out_sd[dst_key].shape == v.shape:
                        out_sd[dst_key] = v
                        found_any = True
            if not found_any:
                print(f"[WARN] branch {bd} has no {src_prefix}* tensors; expert {e_id} at {prefix} "
                      f"keeps random init")
        # router: leave at RWKVXModel's own random init (already in out_sd from moe_model construction)

    moe_model.load_state_dict(out_sd, strict=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    moe_model.save_pretrained(out_dir)

    meta = {
        "engine": "rwkv-x godfather merge_moe.py",
        "base_model": str(base_dir),
        "branches": [str(b) for b in branch_dirs],
        "num_experts": num_experts,
        "top_k": moe_cfg.num_experts_per_tok,
        "note": "Custom MoE-upcycled Channel-Mix, requires rwkv_x_core.RWKVXModel to load "
                "(not compatible with upstream rwkv-x pip package).",
    }
    (out_dir / "merge_config.json").write_text(json.dumps(meta, indent=2))
    print(f"[DONE] merged model -> {out_dir} ({moe_model.num_parameters()/1e6:.1f}M params)")


def main():
    p = argparse.ArgumentParser(description="Merge N RWKV-X checkpoints into a Channel-Mix MoE model")
    p.add_argument("--base", required=True, type=str, help="base checkpoint dir (provides everything except FFN experts)")
    p.add_argument("--branches", required=True, nargs="+", type=str,
                    help="one or more checkpoint dirs, each becomes one expert")
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--top_k", type=int, default=1, help="experts activated per token")
    args = p.parse_args()

    merge(Path(args.base), [Path(b) for b in args.branches], Path(args.out), top_k=args.top_k)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge N SmaulLinear dense checkpoints into one SwiGLU-MoE model.
Each branch's block FFN becomes one expert; attention/embeddings/norms/head come from base."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from smaul_linear import LinearConfig, SmaulLinear


def _load(d: Path):
    cfg = LinearConfig.load(d / "config.json")
    return cfg, load_file(str(d / "model.safetensors"), device="cpu")


def merge(base_dir: Path, branch_dirs: list, out_dir: Path, top_k: int = 1):
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    base_cfg, base_sd = _load(base_dir)
    branches = [_load(bd) for bd in branch_dirs]
    for cfg, _ in branches:
        for f in ("vocab_size", "d_model", "n_layer", "n_heads", "precision"):
            if getattr(cfg, f) != getattr(base_cfg, f):
                raise ValueError(f"branch {f}={getattr(cfg, f)} != base {f}={getattr(base_cfg, f)}")
    n_exp = len(branches)
    if top_k > n_exp:
        raise ValueError(f"top_k ({top_k}) exceeds expert count ({n_exp})")
    moe_cfg = LinearConfig(**{**base_cfg.__dict__, "is_moe": True, "num_experts": n_exp, "num_experts_per_tok": top_k})
    model = SmaulLinear(moe_cfg)
    out_sd = model.state_dict()
    for k, v in base_sd.items():
        if ".ffn." in k or ".gate." in k:
            continue
        if k not in out_sd:
            raise ValueError(f"shared tensor {k} missing from MoE model")
        if tuple(out_sd[k].shape) != tuple(v.shape):
            raise ValueError(f"shared tensor {k} shape mismatch: {tuple(v.shape)} vs {tuple(out_sd[k].shape)}")
        out_sd[k] = v
    for e, (_, sd) in enumerate(branches):
        for k, v in sd.items():
            if ".ffn." not in k:
                continue
            dst = k.replace(".ffn.", f".ffn.experts.{e}.", 1)
            if dst not in out_sd:
                raise ValueError(f"MoE model missing expert tensor {dst}")
            if tuple(out_sd[dst].shape) != tuple(v.shape):
                raise ValueError(f"expert tensor {dst} shape mismatch")
            out_sd[dst] = v
    model.load_state_dict(out_sd, strict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    (out_dir / "merge_config.json").write_text(json.dumps({"base": str(base_dir), "branches": [str(b) for b in branch_dirs],
        "num_experts": n_exp, "top_k": top_k, "note": "SmaulLinear SwiGLU-MoE; routers freshly initialized."}, indent=2))
    print(f"[DONE] merged -> {out_dir} ({n_exp} experts, top_k={top_k})")


def main():
    p = argparse.ArgumentParser(description="Merge SmaulLinear checkpoints into MoE")
    p.add_argument("--base", required=True, type=str)
    p.add_argument("--branches", required=True, nargs="+", type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--top_k", type=int, default=1)
    args = p.parse_args()
    merge(Path(args.base), [Path(b) for b in args.branches], Path(args.out), args.top_k)


if __name__ == "__main__":
    main()

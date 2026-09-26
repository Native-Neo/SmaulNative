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
    if not (d / "config.json").exists() or not (d / "model.safetensors").exists():
        raise FileNotFoundError(f"checkpoint incomplete in {d}: need config.json + model.safetensors")
    cfg = LinearConfig.load(d / "config.json")
    if cfg.is_moe:
        raise ValueError(f"branch {d} is already MoE; merge expects dense checkpoints")
    return cfg, load_file(str(d / "model.safetensors"), device="cpu")


def _tokenizer_bytes(d: Path) -> bytes | None:
    p = d / "tokenizer.json"
    return p.read_bytes() if p.exists() else None


def merge(base_dir: Path, branch_dirs: list, out_dir: Path, top_k: int = 1, force: bool = False):
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if out_dir.is_file():
        raise NotADirectoryError(f"--out {out_dir} is a file, not a directory")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty {out_dir}; pass --force")
    base_cfg, base_sd = _load(base_dir)
    branches = [_load(bd) for bd in branch_dirs]
    for cfg, _ in branches:
        for f in ("vocab_size", "d_model", "n_layer", "n_heads", "precision", "ffn_mult", "tile", "eps"):
            if getattr(cfg, f) != getattr(base_cfg, f):
                raise ValueError(f"branch {f}={getattr(cfg, f)} != base {f}={getattr(base_cfg, f)}")
    # Tokenizers must match or the merged model tokenizes differently per expert.
    # Copy base tokenizer so the output is directly usable with LinearInference.
    base_tok = _tokenizer_bytes(base_dir)
    if base_tok is None:
        print(f"[WARN] base {base_dir} has no tokenizer.json; merged dir will need one for inference")
    else:
        for bd in branch_dirs:
            tok = _tokenizer_bytes(Path(bd))
            if tok is None:
                print(f"[WARN] branch {bd} has no tokenizer.json; skipping equality check")
            elif tok != base_tok:
                raise ValueError(f"tokenizer mismatch: {bd} != {base_dir}")
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
    print("[WARN] MoE routers (gate) are freshly initialized; retrain/finetune before use")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    if base_tok is not None:
        (out_dir / "tokenizer.json").write_bytes(base_tok)
    (out_dir / "merge_config.json").write_text(json.dumps({"base": str(base_dir), "branches": [str(b) for b in branch_dirs],
        "num_experts": n_exp, "top_k": top_k, "note": "SmaulLinear SwiGLU-MoE; routers freshly initialized."}, indent=2))
    print(f"[DONE] merged -> {out_dir} ({n_exp} experts, top_k={top_k})")


def main():
    p = argparse.ArgumentParser(description="Merge SmaulLinear checkpoints into MoE")
    p.add_argument("--base", required=True, type=str)
    p.add_argument("--branches", required=True, nargs="+", type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Allow overwriting non-empty --out")
    args = p.parse_args()
    merge(Path(args.base), [Path(b) for b in args.branches], Path(args.out), args.top_k, force=args.force)


if __name__ == "__main__":
    main()

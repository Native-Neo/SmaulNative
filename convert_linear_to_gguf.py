#!/usr/bin/env python3
"""Convert a SmaulLinear checkpoint to GGUF (FP32/F16, FP8 weights dequantized per-tile at export)."""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

MAX_VOCAB_IDS = 1_000_000


def _load_tokenizer(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"could not load tokenizer {path}: {exc}") from exc
    vocab = data.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json has no vocabulary")
    max_id = -1
    for token, idx in vocab.items():
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"invalid tokenizer id for {token!r}: {idx!r}")
        if idx > MAX_VOCAB_IDS:
            raise ValueError(f"tokenizer id {idx} exceeds cap {MAX_VOCAB_IDS} (crafted file?)")
        max_id = max(max_id, idx)
    tokens = [None] * (max_id + 1)
    for token, idx in vocab.items():
        tokens[idx] = token
    if any(t is None for t in tokens):
        raise ValueError("tokenizer vocabulary contains gaps")
    return tokens


def _dequant(state, expected_tile: int | None = None):
    from fp8_tile import decode_tile
    out = {}
    for k, v in state.items():
        if k.endswith(".w8"):
            base = k[:-3]
            if base + ".sc" not in state:
                raise ValueError(f"FP8 weight {k} has no matching {base}.sc")
            sc = state[base + ".sc"]
            if sc.dim() != 2 or sc.shape[0] != v.shape[0] or sc.shape[1] <= 0:
                raise ValueError(f"bad scale shape for {k}: {tuple(sc.shape)}")
            out_f, in_f = v.shape
            n_tiles = sc.shape[1]
            if in_f % n_tiles != 0:
                raise ValueError(f"in_f ({in_f}) not divisible by tiles ({n_tiles}) for {k}")
            tile = in_f // n_tiles
            if tile <= 0:
                raise ValueError(f"zero tile for {k}")
            if expected_tile is not None and tile != expected_tile:
                raise ValueError(f"tile mismatch for {k}: {tile} != config {expected_tile}")
            parts = [decode_tile(v, sc, 0, out_f, t, tile, torch.float32) for t in range(n_tiles)]
            out[base + ".weight"] = torch.cat(parts, 1)
        elif k.endswith(".sc"):
            continue
        else:
            out[k] = v
    return out


def convert(input_dir: Path, output: Path, dtype: str, overwrite: bool = False):
    try:
        import gguf
    except ImportError as exc:
        raise SystemExit("Missing dependency: install the gguf Python package") from exc
    for p in (input_dir / "config.json", input_dir / "model.safetensors", input_dir / "tokenizer.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    try:
        cfg = json.loads((input_dir / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"could not load config: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError("config.json must contain an object")
    for key in ("vocab_size", "d_model", "n_layer", "n_heads"):
        v = cfg.get(key)
        if not isinstance(v, int) or v <= 0 or v > 1_000_000:
            raise ValueError(f"config {key} invalid: {v!r}")
    ctx_len = int(cfg.get("ctx_len", cfg.get("context_length", 512)))
    if ctx_len <= 0 or ctx_len > 1_000_000:
        ctx_len = 512
    tokens = _load_tokenizer(input_dir / "tokenizer.json")
    if len(tokens) != int(cfg["vocab_size"]):
        raise ValueError(f"tokenizer vocab is {len(tokens)}, checkpoint expects {cfg['vocab_size']}")
    from fp8_tile import decode_tile
    raw = load_file(str(input_dir / "model.safetensors"), device="cpu")
    expected_tile = int(cfg.get("tile", 64))
    output.parent.mkdir(parents=True, exist_ok=True)
    cast = torch.float16 if dtype == "f16" else torch.float32
    tmp = output.with_suffix(output.suffix + ".tmp")
    writer = gguf.GGUFWriter(str(tmp), "smaul_linear")
    writer.add_name("SmaulLinear")
    writer.add_description("SmaulLinear FP8-trained checkpoint exported from SmaulNative")
    writer.add_uint32("vocab_size", int(cfg["vocab_size"]))
    writer.add_uint32("context_length", ctx_len)
    writer.add_uint32("embedding_length", int(cfg["d_model"]))
    writer.add_uint32("block_count", int(cfg["n_layer"]))
    writer.add_uint32("attention.head_count", int(cfg["n_heads"]))
    writer.add_uint32("smaul_linear.tile", int(cfg.get("tile", 64)))
    writer.add_bool("smaul_linear.is_moe", bool(cfg.get("is_moe", False)))
    writer.add_uint32("smaul_linear.num_experts", int(cfg.get("num_experts", 1)))
    writer.add_tokenizer_model("smaul")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    # Stream tensors: dequant one FP8 weight at a time and free the source
    # immediately, so peak RAM is ~1 tensor, not 2× the whole model.
    count = 0
    for orig in sorted(list(raw.keys())):
        if orig.endswith(".sc"):
            continue
        tensor = raw.pop(orig)
        name = orig
        if orig.endswith(".w8"):
            base = orig[:-3]
            sc = raw.pop(base + ".sc", None)
            if sc is None:
                raise ValueError(f"FP8 weight {orig} has no matching {base}.sc")
            out_f, in_f = tensor.shape
            n_tiles = sc.shape[1]
            tile = in_f // n_tiles
            if tile != expected_tile:
                raise ValueError(f"tile mismatch for {orig}: {tile} != {expected_tile}")
            parts = [decode_tile(tensor, sc, 0, out_f, t, tile, torch.float32) for t in range(n_tiles)]
            tensor = torch.cat(parts, 1)
            del parts, sc
            name = base + ".weight"
        t = tensor.to(cast).contiguous().numpy() if torch.is_floating_point(tensor) else tensor.numpy()
        writer.add_tensor(name, t)
        count += 1
        del tensor, t
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    os.replace(tmp, output)
    print(f"[GGUF] wrote {output} | tensors: {count} | dtype: {dtype}")


def main():
    p = argparse.ArgumentParser(description="Convert SmaulLinear to GGUF")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    convert(args.input_dir, args.output, args.dtype, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert a SmaulLinear checkpoint to GGUF (FP32/F16, FP8 weights dequantized per-tile at export)."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def _load_tokenizer(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json has no vocabulary")
    tokens = [None] * (max(vocab.values()) + 1)
    for token, idx in vocab.items():
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"invalid tokenizer id for {token!r}: {idx!r}")
        tokens[idx] = token
    if any(t is None for t in tokens):
        raise ValueError("tokenizer vocabulary contains gaps")
    return tokens


def _dequant(state):
    from fp8_tile import decode_tile
    out = {}
    for k, v in state.items():
        if k.endswith(".w8"):
            base = k[:-3]
            sc = state[base + ".sc"]
            out_f, in_f = v.shape
            tile = in_f // sc.shape[1] if sc.shape[1] else in_f
            parts = [decode_tile(v, sc, 0, out_f, t, tile, torch.float32) for t in range(sc.shape[1])]
            out[base + ".weight"] = torch.cat(parts, 1) if parts else v.float()
        elif k.endswith(".sc"):
            continue
        else:
            out[k] = v
    return out


def convert(input_dir: Path, output: Path, dtype: str):
    try:
        import gguf
    except ImportError as exc:
        raise SystemExit("Missing dependency: install the gguf Python package") from exc
    for p in (input_dir / "config.json", input_dir / "model.safetensors", input_dir / "tokenizer.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    cfg = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    tokens = _load_tokenizer(input_dir / "tokenizer.json")
    if len(tokens) != int(cfg["vocab_size"]):
        raise ValueError(f"tokenizer vocab is {len(tokens)}, checkpoint expects {cfg['vocab_size']}")
    state = _dequant(load_file(str(input_dir / "model.safetensors"), device="cpu"))
    output.parent.mkdir(parents=True, exist_ok=True)
    cast = torch.float16 if dtype == "f16" else torch.float32
    writer = gguf.GGUFWriter(str(output), "smaul_linear")
    writer.add_name("SmaulLinear")
    writer.add_description("SmaulLinear FP8-trained checkpoint exported from SmaulNative")
    writer.add_uint32("vocab_size", int(cfg["vocab_size"]))
    writer.add_uint32("context_length", 2048)
    writer.add_uint32("embedding_length", int(cfg["d_model"]))
    writer.add_uint32("block_count", int(cfg["n_layer"]))
    writer.add_uint32("attention.head_count", int(cfg["n_heads"]))
    writer.add_uint32("smaul_linear.tile", int(cfg.get("tile", 64)))
    writer.add_bool("smaul_linear.is_moe", bool(cfg.get("is_moe", False)))
    writer.add_uint32("smaul_linear.num_experts", int(cfg.get("num_experts", 1)))
    writer.add_tokenizer_model("smaul")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    for name, tensor in state.items():
        t = tensor.to(cast).contiguous().numpy() if torch.is_floating_point(tensor) else tensor.numpy()
        writer.add_tensor(name, t)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"[GGUF] wrote {output} | tensors: {len(state)} | dtype: {dtype}")


def main():
    p = argparse.ArgumentParser(description="Convert SmaulLinear to GGUF")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    args = p.parse_args()
    convert(args.input_dir, args.output, args.dtype)


if __name__ == "__main__":
    main()

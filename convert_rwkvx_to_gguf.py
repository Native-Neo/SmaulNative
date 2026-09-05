#!/usr/bin/env python3
"""Convert a SmaulNative RWKV-X checkpoint to an FP32/FP16 GGUF container."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def _load_tokenizer(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    vocab = model.get("vocab")
    merges = model.get("merges", [])
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json has no BPE vocabulary")
    tokens = [None] * (max(vocab.values()) + 1)
    for token, idx in vocab.items():
        if not isinstance(idx, int) or idx < 0:
            raise ValueError(f"invalid tokenizer id for {token!r}: {idx!r}")
        tokens[idx] = token
    if any(t is None for t in tokens):
        raise ValueError("tokenizer vocabulary contains gaps")
    return tokens, merges


def _write_metadata(writer, cfg, tokens, merges):
    writer.add_name("SmaulNative RWKV-X")
    writer.add_description("RWKV-X checkpoint exported from SmaulNative")
    writer.add_uint32("vocab_size", int(cfg["vocab_size"]))
    writer.add_uint32("context_length", int(cfg.get("ctx_len_hint", 2048)))
    writer.add_uint32("embedding_length", int(cfg["n_embd"]))
    writer.add_uint32("block_count", int(cfg["n_layer"]))
    writer.add_uint32("attention.head_count", int(cfg["n_embd"] // cfg["head_size"]))
    writer.add_uint32("attention.head_count_kv", int(cfg["n_embd"] // cfg["head_size"]))
    writer.add_uint32("rwkv_x.head_size", int(cfg["head_size"]))
    writer.add_uint32("rwkv_x.n_moba_layer", int(cfg["n_moba_layer"]))
    writer.add_uint32("rwkv_x.moba_chunk_size", int(cfg["moba_chunk_size"]))
    writer.add_uint32("rwkv_x.moba_topk", int(cfg["moba_topk"]))
    writer.add_uint32("rwkv_x.wkv_chunk_size", int(cfg["wkv_chunk_size"]))
    writer.add_uint32("rwkv_x.head_size_divisor", int(cfg["head_size_divisor"]))
    writer.add_bool("rwkv_x.is_moe", bool(cfg.get("is_moe", False)))
    writer.add_uint32("rwkv_x.num_experts", int(cfg.get("num_experts", 1)))
    writer.add_uint32("rwkv_x.num_experts_per_tok", int(cfg.get("num_experts_per_tok", 1)))

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_merges([" ".join(m) if isinstance(m, list) else str(m) for m in merges])


def convert(input_dir: Path, output: Path, dtype: str):
    try:
        import gguf
    except ImportError as exc:
        raise SystemExit("Missing dependency: install the gguf Python package") from exc

    config_path = input_dir / "config.json"
    weights_path = input_dir / "model.safetensors"
    tokenizer_path = input_dir / "tokenizer.json"
    for path in (config_path, weights_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    tokens, merges = _load_tokenizer(tokenizer_path)
    state = load_file(str(weights_path), device="cpu")

    expected_vocab = int(cfg["vocab_size"])
    if len(tokens) != expected_vocab:
        raise ValueError(f"tokenizer vocab is {len(tokens)}, checkpoint expects {expected_vocab}")

    output.parent.mkdir(parents=True, exist_ok=True)
    requested_dtype = torch.float16 if dtype == "f16" else torch.float32
    writer = gguf.GGUFWriter(str(output), "rwkv_x")
    _write_metadata(writer, cfg, tokens, merges)

    for name, tensor in state.items():
        if not torch.is_floating_point(tensor):
            writer.add_tensor(name, tensor.numpy())
            continue
        writer.add_tensor(name, tensor.to(requested_dtype).contiguous().numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"[GGUF] wrote {output}")
    print(f"[GGUF] architecture: rwkv_x | tensors: {len(state)} | dtype: {dtype}")


def main():
    p = argparse.ArgumentParser(description="Convert SmaulNative RWKV-X to GGUF")
    p.add_argument("input_dir", type=Path, help="checkpoint directory containing config/model/tokenizer")
    p.add_argument("output", type=Path, help="output .gguf path")
    p.add_argument("--dtype", choices=("f32", "f16"), default="f16")
    args = p.parse_args()
    convert(args.input_dir, args.output, args.dtype)


if __name__ == "__main__":
    main()

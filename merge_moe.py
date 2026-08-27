#!/usr/bin/env python3
# merge_moe.py
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file

from rwkv_x_core import RWKVXConfig, RWKVXModel


def load_checkpoint(directory: Path):
    config_path = directory / "config.json"
    model_path = directory / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors: {model_path}")
    return RWKVXConfig.load(config_path), load_file(str(model_path), device="cpu")


def load_tokenizer(directory: Path) -> dict:
    path = directory / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_compatible(base_config: RWKVXConfig, branch_config: RWKVXConfig, branch_path: Path):
    for field in ("n_embd", "n_layer", "n_moba_layer", "head_size"):
        base_value, branch_value = getattr(base_config, field), getattr(branch_config, field)
        if base_value != branch_value:
            raise ValueError(f"{branch_path}: incompatible {field}: base={base_value}, branch={branch_value}")


def get_vocab(tokenizer: dict) -> Dict[str, int]:
    try:
        vocab = tokenizer["model"]["vocab"]
    except KeyError as e:
        raise ValueError("tokenizer.json does not contain model.vocab") from e
    if not isinstance(vocab, dict):
        raise ValueError("tokenizer model.vocab must be a dictionary")
    return vocab


def get_merges(tokenizer: dict) -> List:
    merges = tokenizer.get("model", {}).get("merges", [])
    if not isinstance(merges, list):
        raise ValueError("tokenizer model.merges must be a list")
    return merges


def merge_tokenizers(base_dir: Path, branch_dirs: List[Path]) -> Tuple[dict, int, dict]:
    base_tokenizer = load_tokenizer(base_dir)
    merged_vocab = dict(get_vocab(base_tokenizer))
    merged_merges = list(get_merges(base_tokenizer))
    merge_set = {tuple(m) if isinstance(m, list) else m for m in merged_merges}
    next_token_id = max(merged_vocab.values(), default=-1) + 1
    added_tokens = added_merges = 0

    for branch_dir in branch_dirs:
        if branch_dir.resolve() == base_dir.resolve():
            continue

        branch_tokenizer = load_tokenizer(branch_dir)

        for token, branch_id in get_vocab(branch_tokenizer).items():
            if token in merged_vocab:
                existing_id = merged_vocab[token]
                if existing_id != branch_id:
                    raise ValueError(
                        "Tokenizer ID conflict detected.\n"
                        f"Checkpoint: {branch_dir}\nToken: {token!r}\n"
                        f"Merged/base ID: {existing_id}\nBranch ID: {branch_id}\n\n"
                        "This checkpoint cannot be safely merged without remapping "
                        "embedding and output weight rows."
                    )
                continue
            merged_vocab[token] = next_token_id
            next_token_id += 1
            added_tokens += 1

        for merge in get_merges(branch_tokenizer):
            key = tuple(merge) if isinstance(merge, list) else merge
            if key in merge_set:
                continue
            merge_set.add(key)
            merged_merges.append(merge)
            added_merges += 1

    merged_tokenizer = dict(base_tokenizer)
    merged_model = dict(base_tokenizer["model"])
    merged_model["vocab"] = merged_vocab
    merged_model["merges"] = merged_merges
    merged_tokenizer["model"] = merged_model

    stats = {
        "base_vocab_size": len(get_vocab(base_tokenizer)),
        "merged_vocab_size": len(merged_vocab),
        "added_tokens": added_tokens,
        "added_merges": added_merges,
    }
    return merged_tokenizer, len(merged_vocab), stats


def resize_vocab_matrix(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    current_size = tensor.shape[0]
    if target_size == current_size:
        return tensor
    if target_size < current_size:
        return tensor[:target_size].clone()

    output = torch.empty((target_size, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    output[:current_size].copy_(tensor)

    source = tensor.float()
    mean, std = source.mean(), source.std(unbiased=False)
    if not torch.isfinite(std) or std.item() <= 1e-12:
        std = torch.tensor(0.02, dtype=torch.float32)

    extra = torch.normal(mean=float(mean), std=float(std), size=(target_size - current_size, *tensor.shape[1:]))
    output[current_size:].copy_(extra.to(dtype=tensor.dtype))
    return output


def cmix_prefixes(config: RWKVXConfig) -> List[str]:
    rwkv_layers = config.n_layer - config.n_moba_layer
    return [f"rwkv_blocks.{i}.ffn" for i in range(rwkv_layers)] + \
           [f"moba_blocks.{i}.ffn" for i in range(config.n_moba_layer)]


def copy_shared_weights(base_state: Dict[str, torch.Tensor], output_state: Dict[str, torch.Tensor], target_vocab_size: int):
    copied = set()
    for key, value in base_state.items():
        if ".ffn." in key:
            continue
        if key not in output_state:
            raise KeyError(f"Output model is missing shared tensor: {key}")

        if key in ("emb.weight", "head.weight"):
            value = resize_vocab_matrix(value, target_vocab_size)

        if output_state[key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch for {key}: expected {tuple(output_state[key].shape)}, got {tuple(value.shape)}"
            )
        output_state[key] = value
        copied.add(key)
    return copied


def copy_expert(output_state: Dict[str, torch.Tensor], branch_state: Dict[str, torch.Tensor],
                 prefix: str, expert_id: int, branch_path: Path):
    source_prefix = f"{prefix}."
    destination_prefix = f"{prefix}.experts.{expert_id}."
    copied = 0

    for key, value in branch_state.items():
        if not key.startswith(source_prefix):
            continue
        destination_key = destination_prefix + key[len(source_prefix):]

        if destination_key not in output_state:
            raise KeyError(f"MoE model is missing expert tensor: {destination_key}")
        if output_state[destination_key].shape != value.shape:
            raise ValueError(
                f"Shape mismatch in {branch_path}\nSource: {key} {tuple(value.shape)}\n"
                f"Target: {destination_key} {tuple(output_state[destination_key].shape)}"
            )
        output_state[destination_key] = value
        copied += 1

    if copied == 0:
        raise ValueError(f"{branch_path} contains no FFN weights for {prefix}")
    return copied


def atomic_json_save(data: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_tokenizer_save(tokenizer: dict, output_path: Path):
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output_path)


def merge(base_dir: Path, branch_dirs: List[Path], output_dir: Path, top_k: int):
    if not branch_dirs:
        raise ValueError("At least one branch is required.")
    if top_k < 1:
        raise ValueError("--top_k must be at least 1.")

    base_dir = base_dir.resolve()
    branch_dirs = [b.resolve() for b in branch_dirs]
    output_dir = output_dir.resolve()

    if output_dir == base_dir:
        raise ValueError("--out cannot overwrite --base.")
    for branch_dir in branch_dirs:
        if output_dir == branch_dir:
            raise ValueError("--out cannot overwrite a branch.")

    base_config, base_state = load_checkpoint(base_dir)
    branches = []
    for branch_dir in branch_dirs:
        branch_config, branch_state = load_checkpoint(branch_dir)
        assert_compatible(base_config, branch_config, branch_dir)
        branches.append((branch_dir, branch_config, branch_state))

    merged_tokenizer, merged_vocab_size, tokenizer_stats = merge_tokenizers(base_dir, branch_dirs)

    num_experts = len(branches)
    top_k = min(top_k, num_experts)

    print(f"[MERGE] Base: {base_dir}")
    print(f"[MERGE] Branches: {num_experts}")
    print(f"[MERGE] Top-K: {top_k}")
    print(f"[MERGE] Vocabulary: {tokenizer_stats['base_vocab_size']} -> {merged_vocab_size}")

    config_data = dict(base_config.__dict__)
    config_data.update(is_moe=True, vocab_size=merged_vocab_size, num_experts=num_experts, num_experts_per_tok=top_k)
    moe_config = RWKVXConfig(**config_data)
    moe_model = RWKVXModel(moe_config)
    output_state = moe_model.state_dict()

    print("[MERGE] Copying shared weights...")
    copy_shared_weights(base_state, output_state, merged_vocab_size)

    print("[MERGE] Copying experts...")
    for prefix in cmix_prefixes(base_config):
        for expert_id, (branch_dir, _, branch_state) in enumerate(branches):
            copy_expert(output_state, branch_state, prefix, expert_id, branch_dir)

    missing, unexpected = moe_model.load_state_dict(output_state, strict=False)
    if missing:
        raise RuntimeError("Missing model tensors:\n" + "\n".join(missing))
    if unexpected:
        raise RuntimeError("Unexpected model tensors:\n" + "\n".join(unexpected))

    output_dir.mkdir(parents=True, exist_ok=True)
    print("[SAVE] Writing merged model...")
    moe_model.save_pretrained(output_dir)
    atomic_tokenizer_save(merged_tokenizer, output_dir / "tokenizer.json")

    metadata = {
        "base_model": str(base_dir),
        "branches": [str(b) for b in branch_dirs],
        "num_experts": num_experts,
        "top_k": top_k,
        "vocab_size": merged_vocab_size,
        "tokenizer_merge": tokenizer_stats,
    }
    atomic_json_save(metadata, output_dir / "merge_config.json")

    parameter_count = moe_model.num_parameters() / 1_000_000
    print(f"\n[DONE]\nModel: {output_dir}\nParameters: {parameter_count:.1f}M\n"
          f"Experts: {num_experts}\nTop-K: {top_k}\nVocabulary: {merged_vocab_size}")


def parse_args():
    p = argparse.ArgumentParser(description="Merge RWKV-X checkpoints into a Channel-Mix MoE model.")
    p.add_argument("--base", required=True, type=str)
    p.add_argument("--branches", required=True, nargs="+", type=str)
    p.add_argument("--out", required=True, type=str)
    p.add_argument("--top_k", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    merge(
        base_dir=Path(args.base),
        branch_dirs=[Path(b) for b in args.branches],
        output_dir=Path(args.out),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

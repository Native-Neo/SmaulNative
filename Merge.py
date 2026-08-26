#!/usr/bin/env python3
"""
Merge.py

Converts:

    ./SmaulNative-Base

and:

    ./branches/
        cyber/
        math/
        code/
        tool_calling/
        thinking/
        defense/
        offense/
        biology/
        chemistry/
        quantum/

into:

    ./SmaulNative-MoE-1.8B/

The output uses a sparse MoE layout with:
    - shared attention backbone from the base model
    - 10 FFN experts per transformer layer
    - Top-2 routing
    - merged and deduplicated tokenizer
    - resized embedding/lm_head matrices
    - safetensors output

Memory strategy:
    - Never load all checkpoints at once.
    - Index checkpoint files only.
    - Load one tensor at a time.
    - Convert one layer at a time.
    - Stream tensors into sharded safetensors output.

Dependencies:

    pip install torch transformers tokenizers safetensors tqdm

Usage:

    python3 Merge.py

"""

import gc
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
)
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path("./SmaulNative-Base")
BRANCHES_DIR = Path("./branches")
OUTPUT_DIR = Path("./SmaulNative-MoE-1.8B")

BRANCH_NAMES = [
    "cyber",
    "math",
    "code",
    "tool_calling",
    "thinking",
    "defense",
    "offense",
    "biology",
    "chemistry",
    "quantum",
]

NUM_EXPERTS = len(BRANCH_NAMES)
NUM_EXPERTS_PER_TOKEN = 2

ROUTER_INIT_STD = 0.02

# Keep individual output shards reasonably small.
MAX_SHARD_SIZE_BYTES = 1024 * 1024 * 1024

# CPU dtype for converted output.
# Change to torch.float16 if the source checkpoints are fp16 and you want
# approximately half the disk footprint.
OUTPUT_DTYPE = torch.float16

# If tied embeddings are used, output embedding and lm_head are still written
# explicitly unless the source checkpoint contains only one of them.
COPY_BASE_NON_MLP_WEIGHTS = True


# ============================================================================
# Utilities
# ============================================================================

def die(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def ensure_dependencies() -> None:
    if not torch.cuda.is_available():
        pass


def clean_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def normalize_dtype(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype=OUTPUT_DTYPE)
    return tensor


def cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().contiguous()


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cleanup(*objects) -> None:
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    empty_cuda_cache()


# ============================================================================
# Checkpoint index
# ============================================================================

class CheckpointIndex:
    """
    Provides lazy tensor lookup across:
        - model.safetensors
        - sharded model-xxxxx.safetensors
        - pytorch_model.bin
        - sharded pytorch_model.bin
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.safetensor_map: Dict[str, Path] = {}
        self.torch_file_map: Dict[str, Path] = {}
        self._torch_cache_path: Optional[Path] = None
        self._torch_cache: Optional[Dict[str, torch.Tensor]] = None

        if not directory.exists():
            die(f"Checkpoint directory does not exist: {directory}")

        self._build_index()

    def _build_index(self) -> None:
        safetensors_files = sorted(self.directory.glob("*.safetensors"))

        for file_path in safetensors_files:
            try:
                with safe_open(str(file_path), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key in self.safetensor_map:
                            die(
                                f"Duplicate tensor key '{key}' found in "
                                f"{self.directory}"
                            )
                        self.safetensor_map[key] = file_path
            except Exception as exc:
                die(f"Failed to index safetensors file {file_path}: {exc}")

        if self.safetensor_map:
            return

        bin_files = sorted(self.directory.glob("*.bin"))

        if not bin_files:
            die(
                f"No .safetensors or .bin checkpoint files found in "
                f"{self.directory}"
            )

        for file_path in bin_files:
            try:
                state = torch.load(
                    file_path,
                    map_location="cpu",
                    weights_only=False,
                )

                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]

                if not isinstance(state, dict):
                    die(f"Unsupported checkpoint format: {file_path}")

                for key in state.keys():
                    if key in self.torch_file_map:
                        die(
                            f"Duplicate tensor key '{key}' found in "
                            f"{self.directory}"
                        )
                    self.torch_file_map[key] = file_path

                del state
                gc.collect()

            except Exception as exc:
                die(f"Failed to index checkpoint {file_path}: {exc}")

    def keys(self) -> List[str]:
        if self.safetensor_map:
            return list(self.safetensor_map.keys())
        return list(self.torch_file_map.keys())

    def has(self, key: str) -> bool:
        return (
            key in self.safetensor_map
            or key in self.torch_file_map
        )

    def get_tensor(self, key: str) -> torch.Tensor:
        if key in self.safetensor_map:
            file_path = self.safetensor_map[key]

            with safe_open(
                str(file_path),
                framework="pt",
                device="cpu",
            ) as f:
                tensor = f.get_tensor(key)

            return tensor

        if key in self.torch_file_map:
            file_path = self.torch_file_map[key]

            if self._torch_cache_path != file_path:
                if self._torch_cache is not None:
                    del self._torch_cache
                    self._torch_cache = None
                    gc.collect()

                state = torch.load(
                    file_path,
                    map_location="cpu",
                    weights_only=False,
                )

                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]

                if not isinstance(state, dict):
                    die(f"Unsupported checkpoint format: {file_path}")

                self._torch_cache = state
                self._torch_cache_path = file_path

            if key not in self._torch_cache:
                die(f"Tensor '{key}' disappeared from {file_path}")

            return self._torch_cache[key].clone()

        die(
            f"Tensor '{key}' was not found in checkpoint: "
            f"{self.directory}"
        )


# ============================================================================
# Streaming safetensors writer
# ============================================================================

class ShardedSafeTensorWriter:
    """
    Buffers a controlled amount of tensors and writes sequential safetensors
    shards. This avoids retaining the complete ~1.8B parameter MoE state
    dictionary in RAM.
    """

    def __init__(
        self,
        output_dir: Path,
        max_shard_size: int,
    ):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size

        self.current_tensors: Dict[str, torch.Tensor] = OrderedDict()
        self.current_size = 0

        self.shard_paths: List[Path] = []
        self.weight_map: Dict[str, str] = OrderedDict()

        self.shard_index = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = cpu_contiguous(tensor)

        size = tensor_nbytes(tensor)

        if (
            self.current_tensors
            and self.current_size + size > self.max_shard_size
        ):
            self.flush()

        self.current_tensors[key] = tensor
        self.current_size += size

        if size >= self.max_shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.current_tensors:
            return

        filename = (
            f"model-{self.shard_index:05d}.safetensors"
        )

        path = self.output_dir / filename

        metadata = {
            "format": "pt",
        }

        save_file(
            dict(self.current_tensors),
            str(path),
            metadata=metadata,
        )

        for key in self.current_tensors.keys():
            self.weight_map[key] = filename

        self.shard_paths.append(path)

        self.current_tensors.clear()
        self.current_size = 0
        self.shard_index += 1

        gc.collect()

    def finalize(self) -> None:
        self.flush()

        if not self.shard_paths:
            die("No tensors were written.")

        total_size = sum(
            path.stat().st_size
            for path in self.shard_paths
        )

        if len(self.shard_paths) == 1:
            final_path = self.output_dir / "model.safetensors"

            if final_path.exists():
                final_path.unlink()

            self.shard_paths[0].rename(final_path)

            for key in list(self.weight_map.keys()):
                self.weight_map[key] = "model.safetensors"

        else:
            index = {
                "metadata": {
                    "total_size": total_size,
                },
                "weight_map": self.weight_map,
            }

            index_path = (
                self.output_dir
                / "model.safetensors.index.json"
            )

            with open(
                index_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    index,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )


# ============================================================================
# Tokenizer helpers
# ============================================================================

def tokenizer_vocab(tokenizer) -> Dict[str, int]:
    vocab = tokenizer.get_vocab()

    if not isinstance(vocab, dict):
        die("Tokenizer returned an invalid vocabulary.")

    return vocab


def collect_token_strings(tokenizer) -> List[str]:
    """
    Collect normal vocabulary tokens, added tokens and all configured
    special tokens.
    """

    collected = []

    vocab = tokenizer_vocab(tokenizer)

    for token, _ in sorted(
        vocab.items(),
        key=lambda item: item[1],
    ):
        collected.append(token)

    added_vocab = tokenizer.get_added_vocab()

    for token, _ in sorted(
        added_vocab.items(),
        key=lambda item: item[1],
    ):
        collected.append(token)

    special_map = tokenizer.special_tokens_map

    for _, value in special_map.items():
        if value is None:
            continue

        if isinstance(value, list):
            for item in value:
                if hasattr(item, "content"):
                    collected.append(item.content)
                else:
                    collected.append(str(item))
        else:
            if hasattr(value, "content"):
                collected.append(value.content)
            else:
                collected.append(str(value))

    return collected


def load_tokenizer_json(directory: Path) -> Tokenizer:
    tokenizer_json = directory / "tokenizer.json"

    if not tokenizer_json.exists():
        die(f"Missing tokenizer.json in {directory}")

    try:
        return Tokenizer.from_file(str(tokenizer_json))
    except Exception as exc:
        die(f"Failed to load tokenizer.json from {directory}: {exc}")


def merge_tokenizers(
    base_dir: Path,
    branch_dirs: List[Path],
    output_dir: Path,
) -> Tuple[PreTrainedTokenizerFast, Dict[str, int], Dict[str, int]]:
    """
    Base token IDs remain unchanged.

    New branch tokens are appended by using the tokenizer backend's
    add_tokens mechanism.

    Returns:
        unified tokenizer
        base token -> id map
        branch-added token -> unified id map
    """

    print("\n=== Merging tokenizers ===")

    base_tokenizer = AutoTokenizer.from_pretrained(
        str(base_dir),
        use_fast=True,
        trust_remote_code=False,
    )

    if not isinstance(
        base_tokenizer,
        PreTrainedTokenizerFast,
    ):
        die(
            "The base tokenizer is not a PreTrainedTokenizerFast. "
            "A fast tokenizer is required for this converter."
        )

    base_vocab = tokenizer_vocab(base_tokenizer)

    base_vocab_size = len(base_vocab)

    unified = PreTrainedTokenizerFast(
        tokenizer_object=load_tokenizer_json(base_dir),
        **base_tokenizer.init_kwargs,
    )

    # Preserve the exact base special-token configuration.
    if base_tokenizer.bos_token is not None:
        unified.bos_token = base_tokenizer.bos_token

    if base_tokenizer.eos_token is not None:
        unified.eos_token = base_tokenizer.eos_token

    if base_tokenizer.unk_token is not None:
        unified.unk_token = base_tokenizer.unk_token

    if base_tokenizer.pad_token is not None:
        unified.pad_token = base_tokenizer.pad_token

    if base_tokenizer.sep_token is not None:
        unified.sep_token = base_tokenizer.sep_token

    if base_tokenizer.cls_token is not None:
        unified.cls_token = base_tokenizer.cls_token

    if base_tokenizer.mask_token is not None:
        unified.mask_token = base_tokenizer.mask_token

    existing_tokens = set(base_vocab.keys())

    unique_new_tokens: List[str] = []

    for branch_dir in branch_dirs:
        print(f"Scanning tokenizer: {branch_dir}")

        branch_tokenizer = AutoTokenizer.from_pretrained(
            str(branch_dir),
            use_fast=True,
            trust_remote_code=False,
        )

        for token in collect_token_strings(branch_tokenizer):
            if token not in existing_tokens:
                existing_tokens.add(token)
                unique_new_tokens.append(token)

        special_map = branch_tokenizer.special_tokens_map

        additional_specials = special_map.get(
            "additional_special_tokens",
            [],
        )

        if additional_specials:
            normalized = []

            for token in additional_specials:
                if hasattr(token, "content"):
                    normalized.append(token.content)
                else:
                    normalized.append(str(token))

            existing_additional = list(
                unified.additional_special_tokens
            )

            for token in normalized:
                if token not in existing_additional:
                    existing_additional.append(token)

            unified.additional_special_tokens = existing_additional

    if unique_new_tokens:
        print(
            f"Adding {len(unique_new_tokens)} unique tokenizer tokens..."
        )

        unified.add_tokens(
            unique_new_tokens,
            special_tokens=False,
        )

    unified_vocab = tokenizer_vocab(unified)

    # Verify base token IDs are preserved exactly.
    for token, old_id in base_vocab.items():
        new_id = unified_vocab.get(token)

        if new_id != old_id:
            die(
                "Base tokenizer ID mapping changed during merge: "
                f"{repr(token)} was {old_id}, now {new_id}"
            )

    unified.save_pretrained(str(output_dir))

    base_token_to_id = dict(base_vocab)

    added_token_to_id = {
        token: token_id
        for token, token_id in unified_vocab.items()
        if token not in base_vocab
    }

    print(f"Base vocabulary size: {base_vocab_size}")
    print(f"Unified vocabulary size: {len(unified_vocab)}")
    print(
        f"New unique tokens: "
        f"{len(unified_vocab) - base_vocab_size}"
    )

    return (
        unified,
        base_token_to_id,
        added_token_to_id,
    )


# ============================================================================
# Config merging
# ============================================================================

def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        die(f"Failed to read JSON file {path}: {exc}")


def is_structural_key(key: str) -> bool:
    structural_keys = {
        "model_type",
        "architectures",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "rms_norm_eps",
        "attention_bias",
        "attention_dropout",
        "hidden_act",
        "tie_word_embeddings",
        "pretraining_tp",
        "head_dim",
        "mlp_bias",
    }

    return key in structural_keys


def merge_configs(
    base_dir: Path,
    branch_dirs: List[Path],
    output_dir: Path,
    vocab_size: int,
) -> Dict:
    print("\n=== Merging configs ===")

    base_config_path = base_dir / "config.json"

    if not base_config_path.exists():
        die(f"Missing base config.json: {base_config_path}")

    base_config = load_json(base_config_path)

    merged = dict(base_config)

    for branch_dir in branch_dirs:
        branch_config = load_json(
            branch_dir / "config.json"
        )

        for key, value in branch_config.items():
            if key not in merged:
                merged[key] = value
            elif is_structural_key(key):
                # Base architecture wins on structural conflicts.
                continue

    hidden_size = merged.get("hidden_size")

    if hidden_size is None:
        die("Base config is missing hidden_size.")

    merged["vocab_size"] = int(vocab_size)

    # Mixtral-compatible model type and architecture metadata.
    merged["model_type"] = "mixtral"

    merged["architectures"] = [
        "MixtralForCausalLM"
    ]

    merged["num_local_experts"] = NUM_EXPERTS
    merged["num_experts_per_tok"] = NUM_EXPERTS_PER_TOKEN

    # Mixtral sparse MLP configuration.
    merged["router_jitter_noise"] = 0.0
    merged["output_router_logits"] = False

    merged["torch_dtype"] = (
        "float16"
        if OUTPUT_DTYPE == torch.float16
        else "bfloat16"
        if OUTPUT_DTYPE == torch.bfloat16
        else "float32"
    )

    config_path = output_dir / "config.json"

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            merged,
            f,
            indent=2,
            ensure_ascii=False,
        )

    generation_config = load_json(
        base_dir / "generation_config.json"
    )

    if not generation_config:
        generation_config = {
            "_from_model_config": True,
        }

    generation_config_path = (
        output_dir / "generation_config.json"
    )

    with open(
        generation_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            generation_config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Final vocab_size: {vocab_size}")
    print(f"num_local_experts: {NUM_EXPERTS}")
    print(
        f"num_experts_per_tok: "
        f"{NUM_EXPERTS_PER_TOKEN}"
    )

    return merged


# ============================================================================
# Model key detection
# ============================================================================

def detect_model_prefix(index: CheckpointIndex) -> str:
    keys = index.keys()

    candidates = [
        "model.",
        "",
    ]

    for prefix in candidates:
        probe = (
            prefix
            + "embed_tokens.weight"
        )

        if probe in keys:
            return prefix

    # Standard HF LlamaForCausalLM generally uses:
    # model.embed_tokens.weight
    #
    # Some custom checkpoints may have nested prefixes.
    for key in keys:
        if key.endswith("embed_tokens.weight"):
            return key[:-len("embed_tokens.weight")]

    die(
        "Could not detect model tensor prefix from checkpoint."
    )


def detect_lm_head_key(index: CheckpointIndex) -> Optional[str]:
    candidates = [
        "lm_head.weight",
        "model.lm_head.weight",
    ]

    for candidate in candidates:
        if index.has(candidate):
            return candidate

    for key in index.keys():
        if key.endswith("lm_head.weight"):
            return key

    return None


def layer_prefix(prefix: str, layer_idx: int) -> str:
    return (
        f"{prefix}layers.{layer_idx}."
    )


# ============================================================================
# Embedding merge
# ============================================================================

def initialize_new_rows(
    source: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """
    Initialize appended vocabulary rows using source mean/std.
    """

    if count <= 0:
        return torch.empty(
            (0, source.shape[1]),
            dtype=source.dtype,
        )

    source_float = source.float()

    mean = source_float.mean(
        dim=0,
        keepdim=True,
    )

    std = source_float.std(
        dim=0,
        keepdim=True,
        unbiased=False,
    )

    std = torch.clamp(
        std,
        min=1e-6,
    )

    new_rows = (
        torch.randn(
            count,
            source.shape[1],
            dtype=torch.float32,
        )
        * std
        + mean
    )

    return new_rows.to(source.dtype)


def build_token_alignment(
    tokenizer,
) -> Dict[str, int]:
    return tokenizer_vocab(tokenizer)


def merge_embedding_from_branches(
    base_embedding: torch.Tensor,
    branch_indexes: List[CheckpointIndex],
    branch_dirs: List[Path],
    base_embedding_key: str,
    final_vocab: Dict[str, int],
    base_vocab: Dict[str, int],
    kind: str,
) -> torch.Tensor:
    """
    Creates a final embedding/lm_head matrix.

    Base rows remain the starting point.

    For each branch:
        - If its tokenizer token maps to a base token, branch rows may replace
          the base row when the branch row differs.
        - If it maps to a newly appended unified token, copy that row into its
          unified position.

    Branch precedence follows BRANCH_NAMES.
    """

    old_vocab_size = base_embedding.shape[0]
    new_vocab_size = len(final_vocab)

    if new_vocab_size < old_vocab_size:
        die(
            "Unified vocabulary is unexpectedly smaller than "
            "the base embedding vocabulary."
        )

    result = torch.empty(
        (
            new_vocab_size,
            base_embedding.shape[1],
        ),
        dtype=base_embedding.dtype,
    )

    result[:old_vocab_size].copy_(
        base_embedding
    )

    if new_vocab_size > old_vocab_size:
        result[old_vocab_size:].copy_(
            initialize_new_rows(
                base_embedding,
                new_vocab_size - old_vocab_size,
            )
        )

    print(
        f"Aligning branch {kind} rows..."
    )

    for branch_index, branch_dir in zip(
        branch_indexes,
        branch_dirs,
    ):
        branch_tokenizer = AutoTokenizer.from_pretrained(
            str(branch_dir),
            use_fast=True,
            trust_remote_code=False,
        )

        branch_vocab = tokenizer_vocab(
            branch_tokenizer
        )

        branch_key = base_embedding_key

        if not branch_index.has(branch_key):
            if kind == "lm_head":
                alternative = detect_lm_head_key(
                    branch_index
                )

                if alternative is None:
                    print(
                        f"Warning: {branch_dir} has no "
                        f"{kind}; skipping."
                    )
                    continue

                branch_key = alternative

            else:
                die(
                    f"Missing {branch_key} in "
                    f"{branch_dir}"
                )

        branch_matrix = branch_index.get_tensor(
            branch_key
        )

        if branch_matrix.ndim != 2:
            die(
                f"Invalid {kind} tensor shape in "
                f"{branch_dir}: {branch_matrix.shape}"
            )

        for token, branch_id in branch_vocab.items():
            if branch_id < 0:
                continue

            if branch_id >= branch_matrix.shape[0]:
                continue

            unified_id = final_vocab.get(token)

            if unified_id is None:
                continue

            result[unified_id].copy_(
                branch_matrix[branch_id]
            )

        cleanup(
            branch_matrix,
            branch_tokenizer,
        )

    return result


# ============================================================================
# Backbone conversion
# ============================================================================

def is_mlp_tensor_name(
    tensor_name: str,
) -> bool:
    return (
        ".mlp." in tensor_name
        or ".block_sparse_moe." in tensor_name
    )


def convert_shared_layer_weights(
    base_index: CheckpointIndex,
    writer: ShardedSafeTensorWriter,
    model_prefix: str,
    layer_idx: int,
) -> None:
    """
    Copy shared attention and normalization tensors.

    Base:
        model.layers.L.self_attn.*
        model.layers.L.input_layernorm.*
        model.layers.L.post_attention_layernorm.*

    Output keeps the same names.
    """

    prefix = layer_prefix(
        model_prefix,
        layer_idx,
    )

    allowed_suffixes = (
        "self_attn.",
        "input_layernorm.",
        "post_attention_layernorm.",
    )

    for key in base_index.keys():
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix):]

        if not suffix.startswith(
            allowed_suffixes
        ):
            continue

        tensor = base_index.get_tensor(key)

        writer.add(
            key,
            normalize_dtype(tensor),
        )

        cleanup(tensor)


# ============================================================================
# MoE expert conversion
# ============================================================================

def branch_mlp_key(
    model_prefix: str,
    layer_idx: int,
    projection: str,
) -> str:
    return (
        f"{model_prefix}"
        f"layers.{layer_idx}."
        f"mlp.{projection}.weight"
    )


def output_expert_key(
    model_prefix: str,
    layer_idx: int,
    expert_idx: int,
    projection: str,
) -> str:
    return (
        f"{model_prefix}"
        f"layers.{layer_idx}."
        f"block_sparse_moe."
        f"experts.{expert_idx}."
        f"{projection}.weight"
    )


def output_router_key(
    model_prefix: str,
    layer_idx: int,
) -> str:
    return (
        f"{model_prefix}"
        f"layers.{layer_idx}."
        f"block_sparse_moe.gate.weight"
    )


def convert_layer_experts(
    branch_indexes: List[CheckpointIndex],
    writer: ShardedSafeTensorWriter,
    model_prefix: str,
    layer_idx: int,
    hidden_size: int,
) -> None:
    """
    Converts the 10 branch FFNs into 10 independent sparse MoE experts.
    """

    for expert_idx, branch_index in enumerate(
        branch_indexes
    ):
        for projection in (
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            source_key = branch_mlp_key(
                model_prefix,
                layer_idx,
                projection,
            )

            if not branch_index.has(
                source_key
            ):
                die(
                    f"Missing branch MLP tensor:\n"
                    f"  checkpoint: "
                    f"{branch_index.directory}\n"
                    f"  tensor: {source_key}"
                )

            tensor = branch_index.get_tensor(
                source_key
            )

            target_key = output_expert_key(
                model_prefix,
                layer_idx,
                expert_idx,
                projection,
            )

            writer.add(
                target_key,
                normalize_dtype(tensor),
            )

            cleanup(tensor)

    router = (
        torch.randn(
            NUM_EXPERTS,
            hidden_size,
            dtype=torch.float32,
        )
        * ROUTER_INIT_STD
    )

    writer.add(
        output_router_key(
            model_prefix,
            layer_idx,
        ),
        router.to(OUTPUT_DTYPE),
    )

    cleanup(router)


# ============================================================================
# Global model weights
# ============================================================================

def copy_global_non_layer_weights(
    base_index: CheckpointIndex,
    writer: ShardedSafeTensorWriter,
    model_prefix: str,
    embedding_key: str,
    lm_head_key: Optional[str],
) -> None:
    """
    Copies non-layer tensors except embeddings/lm_head, which are handled
    separately.
    """

    excluded = {
        embedding_key,
    }

    if lm_head_key is not None:
        excluded.add(lm_head_key)

    layer_marker = (
        f"{model_prefix}layers."
    )

    for key in base_index.keys():
        if key in excluded:
            continue

        if key.startswith(layer_marker):
            continue

        tensor = base_index.get_tensor(key)

        writer.add(
            key,
            normalize_dtype(tensor),
        )

        cleanup(tensor)


# ============================================================================
# Main conversion
# ============================================================================

def validate_inputs() -> List[Path]:
    if not BASE_DIR.exists():
        die(
            f"Base directory does not exist: "
            f"{BASE_DIR}"
        )

    branch_dirs = []

    for branch_name in BRANCH_NAMES:
        path = BRANCHES_DIR / branch_name

        if not path.exists():
            die(
                f"Missing branch directory: {path}"
            )

        branch_dirs.append(path)

    return branch_dirs


def convert() -> None:
    print("=" * 78)
    print("SmaulNative Branches -> 10-Expert MoE Converter")
    print("=" * 78)

    branch_dirs = validate_inputs()

    clean_output_directory(
        OUTPUT_DIR
    )

    # ------------------------------------------------------------------------
    # Tokenizers
    # ------------------------------------------------------------------------

    (
        unified_tokenizer,
        base_vocab,
        _,
    ) = merge_tokenizers(
        BASE_DIR,
        branch_dirs,
        OUTPUT_DIR,
    )

    final_vocab = build_token_alignment(
        unified_tokenizer
    )

    final_vocab_size = len(
        final_vocab
    )

    cleanup(unified_tokenizer)

    # ------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------

    merged_config = merge_configs(
        BASE_DIR,
        branch_dirs,
        OUTPUT_DIR,
        final_vocab_size,
    )

    hidden_size = merged_config.get(
        "hidden_size"
    )

    num_hidden_layers = merged_config.get(
        "num_hidden_layers"
    )

    if hidden_size is None:
        die("Final config has no hidden_size.")

    if num_hidden_layers is None:
        die(
            "Final config has no "
            "num_hidden_layers."
        )

    hidden_size = int(hidden_size)
    num_hidden_layers = int(
        num_hidden_layers
    )

    # ------------------------------------------------------------------------
    # Checkpoint indexes
    # ------------------------------------------------------------------------

    print("\n=== Indexing checkpoints ===")

    base_index = CheckpointIndex(
        BASE_DIR
    )

    branch_indexes = []

    for branch_dir in tqdm(
        branch_dirs,
        desc="Indexing branches",
    ):
        branch_indexes.append(
            CheckpointIndex(
                branch_dir
            )
        )

    model_prefix = detect_model_prefix(
        base_index
    )

    embedding_key = (
        model_prefix
        + "embed_tokens.weight"
    )

    lm_head_key = detect_lm_head_key(
        base_index
    )

    if not base_index.has(
        embedding_key
    ):
        die(
            f"Base embedding tensor not found: "
            f"{embedding_key}"
        )

    print(
        f"Detected model prefix: "
        f"'{model_prefix}'"
    )

    print(
        f"Detected embedding key: "
        f"{embedding_key}"
    )

    print(
        f"Detected lm_head key: "
        f"{lm_head_key}"
    )

    # ------------------------------------------------------------------------
    # Writer
    # ------------------------------------------------------------------------

    writer = ShardedSafeTensorWriter(
        OUTPUT_DIR,
        MAX_SHARD_SIZE_BYTES,
    )

    # ------------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------------

    print(
        "\n=== Converting embeddings ==="
    )

    base_embedding = (
        base_index.get_tensor(
            embedding_key
        )
    )

    merged_embedding = (
        merge_embedding_from_branches(
            base_embedding=base_embedding,
            branch_indexes=branch_indexes,
            branch_dirs=branch_dirs,
            base_embedding_key=embedding_key,
            final_vocab=final_vocab,
            base_vocab=base_vocab,
            kind="embedding",
        )
    )

    writer.add(
        embedding_key,
        normalize_dtype(
            merged_embedding
        ),
    )

    cleanup(
        merged_embedding,
        base_embedding,
    )

    # ------------------------------------------------------------------------
    # LM head
    # ------------------------------------------------------------------------

    if lm_head_key is not None:
        print(
            "\n=== Converting lm_head ==="
        )

        base_lm_head = (
            base_index.get_tensor(
                lm_head_key
            )
        )

        merged_lm_head = (
            merge_embedding_from_branches(
                base_embedding=base_lm_head,
                branch_indexes=branch_indexes,
                branch_dirs=branch_dirs,
                base_embedding_key=lm_head_key,
                final_vocab=final_vocab,
                base_vocab=base_vocab,
                kind="lm_head",
            )
        )

        writer.add(
            lm_head_key,
            normalize_dtype(
                merged_lm_head
            ),
        )

        cleanup(
            merged_lm_head,
            base_lm_head,
        )

    # ------------------------------------------------------------------------
    # Global backbone
    # ------------------------------------------------------------------------

    print(
        "\n=== Copying global shared backbone ==="
    )

    copy_global_non_layer_weights(
        base_index=base_index,
        writer=writer,
        model_prefix=model_prefix,
        embedding_key=embedding_key,
        lm_head_key=lm_head_key,
    )

    # ------------------------------------------------------------------------
    # Transformer layers
    # ------------------------------------------------------------------------

    print(
        "\n=== Converting Transformer layers to MoE ==="
    )

    progress = tqdm(
        range(num_hidden_layers),
        desc="Converting layers",
        unit="layer",
    )

    for layer_idx in progress:
        progress.set_postfix(
            layer=layer_idx,
            experts=NUM_EXPERTS,
        )

        # Shared backbone:
        # attention + layer norms from the base model.
        convert_shared_layer_weights(
            base_index=base_index,
            writer=writer,
            model_prefix=model_prefix,
            layer_idx=layer_idx,
        )

        # Specialized FFNs become MoE experts.
        convert_layer_experts(
            branch_indexes=branch_indexes,
            writer=writer,
            model_prefix=model_prefix,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
        )

        gc.collect()

    # ------------------------------------------------------------------------
    # Write safetensors index
    # ------------------------------------------------------------------------

    print(
        "\n=== Finalizing safetensors ==="
    )

    writer.finalize()

    # ------------------------------------------------------------------------
    # Final metadata
    # ------------------------------------------------------------------------

    conversion_info = {
        "base_model": str(BASE_DIR),
        "branches": BRANCH_NAMES,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": (
            NUM_EXPERTS_PER_TOKEN
        ),
        "shared_backbone": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ],
        "expert_source": {
            str(i): name
            for i, name in enumerate(
                BRANCH_NAMES
            )
        },
        "router_initialization": {
            "distribution": "normal",
            "mean": 0.0,
            "std": ROUTER_INIT_STD,
        },
        "output_dtype": str(
            OUTPUT_DTYPE
        ),
        "final_vocab_size": (
            final_vocab_size
        ),
        "num_hidden_layers": (
            num_hidden_layers
        ),
        "hidden_size": hidden_size,
    }

    with open(
        OUTPUT_DIR / "conversion_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            conversion_info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 78)
    print("Conversion completed successfully.")
    print("=" * 78)

    print(
        f"\nOutput directory:\n"
        f"  {OUTPUT_DIR.resolve()}"
    )

    print(
        f"\nFinal vocabulary size: "
        f"{final_vocab_size}"
    )

    print(
        f"Experts per layer: "
        f"{NUM_EXPERTS}"
    )

    print(
        f"Experts selected per token: "
        f"{NUM_EXPERTS_PER_TOKEN}"
    )

    print(
        "\nExpert mapping:"
    )

    for i, name in enumerate(
        BRANCH_NAMES
    ):
        print(
            f"  Expert {i}: {name}"
        )

    print(
        "\nOutput files:"
    )

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):
        if path.is_file():
            size_mb = (
                path.stat().st_size
                / (1024 * 1024)
            )

            print(
                f"  {path.name} "
                f"({size_mb:.2f} MB)"
            )


# ============================================================================
# Entrypoint
# ============================================================================

if __name__ == "__main__":
    try:
        torch.set_grad_enabled(False)
        convert()

    except KeyboardInterrupt:
        print(
            "\n\nConversion interrupted by user.",
            file=sys.stderr,
        )
        sys.exit(130)

    except Exception as exc:
        print(
            "\n\nConversion failed:",
            file=sys.stderr,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

        raise

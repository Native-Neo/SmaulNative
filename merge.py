#!/usr/bin/env python3
"""
merge.py

RWKV-X MergeKit / MoE Upcycling Engine
======================================

Converts multiple RWKV-X fine-tuned checkpoints derived from a shared
base model into an RWKV-X Mixture-of-Experts checkpoint.

Architecture targeted by this script:

    emb.weight

    blocks.N.ln1.*
    blocks.N.time_mix.*

    blocks.N.ln2.*
    blocks.N.channel_mix.*

    ln_out.*
    head.weight

MoE conversion:

    blocks.N.channel_mix.PARAM

becomes:

    blocks.N.channel_mix.experts.0.PARAM
    blocks.N.channel_mix.experts.1.PARAM
    blocks.N.channel_mix.experts.2.PARAM
    ...

and:

    blocks.N.channel_mix.gate.weight

is added with shape:

    [num_experts, hidden_size]

Shared parameters remain from the base model:

    emb.*
    blocks.N.ln1.*
    blocks.N.time_mix.*
    blocks.N.ln2.*
    ln_out.*
    head.*

The following Channel-Mix parameters automatically become experts:

    key.*
    value.*
    receptance.*
    time_mix_k
    time_mix_r

The implementation is intentionally parameter-name based rather than
hardcoding a specific RWKV version, so it also supports compatible
future variants using:

    blocks.N.channel_mix.*

Usage:

    python merge.py --config config.yaml


Example config.yaml
-------------------

target_directory: "./SmaulNative-Merged"

base_model: "./SmaulNative-Base"

algorithm: "moe_upcycle"

# Fraction of total experts used per token.
#
# Example:
#   4 experts
#   density 0.5
#
# = top-2 routing
density: 0.5

# Global merge metadata weight.
weight: 1.0

# Keep RAM usage low.
max_shard_size: "512MB"

seed: 42

addedexperts:
  - path: "./branches"
    weight: 1.0


Directory example:

branches/
├── code/
│   ├── config.json
│   └── model.safetensors
├── english/
│   ├── config.json
│   └── model.safetensors
└── math/
    ├── config.json
    └── model.safetensors


Install:

    pip install torch safetensors pyyaml tqdm psutil


IMPORTANT
---------

The resulting checkpoint requires an RWKV-X model implementation that
understands:

    channel_mix.experts.N.*
    channel_mix.gate.weight

A normal single-expert RWKVXChannelMix implementation will not be able
to load these parameters.

This script writes:

    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    merge_config.json

and either:

    model.safetensors

or sharded Hugging Face safetensors:

    model-00001-of-000XX.safetensors
    ...
    model.safetensors.index.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import shutil
import sys

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch
import yaml

from safetensors import safe_open
from safetensors.torch import save_file

from tqdm import tqdm


# ============================================================
# CONSTANTS
# ============================================================

VERSION = "3.0.0"

SUPPORTED_ALGORITHM = "moe_upcycle"


# ============================================================
# LOGGING / ERROR HANDLING
# ============================================================

def log(message: str) -> None:
    print(f"[RWKV-MergeKit] {message}")


def warn(message: str) -> None:
    print(f"[RWKV-MergeKit WARNING] {message}")


def fatal(message: str) -> None:
    print(
        f"\n[RWKV-MergeKit ERROR] {message}",
        file=sys.stderr,
    )
    raise SystemExit(1)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def natural_key(value: str):
    """
    Natural sorting:

        branch1
        branch2
        branch10

    instead of:

        branch1
        branch10
        branch2
    """

    return [
        int(part)
        if part.isdigit()
        else part.lower()

        for part in re.split(
            r"(\d+)",
            value,
        )
    ]


def human_bytes(value: int) -> str:

    value = float(value)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):

        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PB"


def parse_size(value) -> int:

    if isinstance(value, int):
        return value

    text = str(value).strip().upper()

    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*(B|KB|MB|GB|TB)?",
        text,
    )

    if not match:
        raise ValueError(
            f"Invalid size value: {value!r}"
        )

    number = float(
        match.group(1)
    )

    unit = (
        match.group(2)
        or "B"
    )

    units = {
        "B": 1,
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3,
        "TB": 1024 ** 4,
    }

    return int(
        number * units[unit]
    )


def memory_status() -> str:

    process = psutil.Process(
        os.getpid()
    )

    process_memory = (
        process.memory_info().rss
    )

    virtual_memory = (
        psutil.virtual_memory()
    )

    return (
        f"RAM={human_bytes(process_memory)} "
        f"available={human_bytes(virtual_memory.available)}"
    )


def cleanup() -> None:

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_json(path: Path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def save_json(
    data,
    path: Path,
) -> None:

    temporary = Path(
        str(path) + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )

    os.replace(
        temporary,
        path,
    )


# ============================================================
# RWKV-X PARAMETER CLASSIFICATION
# ============================================================

def is_embedding_parameter(
    key: str,
) -> bool:

    return key in (
        "emb.weight",
        "model.emb.weight",
        "rwkv_x.emb.weight",
    )


def is_head_parameter(
    key: str,
) -> bool:

    return key in (
        "head.weight",
        "lm_head.weight",
        "model.head.weight",
        "rwkv_x.head.weight",
    )


def is_channel_mix_parameter(
    key: str,
) -> bool:
    """
    Detects parameters belonging to:

        blocks.N.channel_mix.*

    excluding already-MoE checkpoints.
    """

    return (
        ".channel_mix." in key
        and ".experts." not in key
        and ".gate." not in key
    )


def is_time_mix_parameter(
    key: str,
) -> bool:

    return (
        ".time_mix." in key
    )


def get_channel_mix_prefix(
    key: str,
) -> Optional[str]:

    marker = ".channel_mix."

    position = key.find(
        marker
    )

    if position < 0:
        return None

    return key[
        :position + len(".channel_mix")
    ]


def get_channel_mix_suffix(
    key: str,
) -> Optional[str]:

    prefix = get_channel_mix_prefix(
        key
    )

    if prefix is None:
        return None

    prefix = prefix + "."

    if not key.startswith(
        prefix
    ):
        return None

    return key[
        len(prefix):
    ]


def make_expert_key(
    original_key: str,
    expert_id: int,
) -> str:

    prefix = get_channel_mix_prefix(
        original_key
    )

    suffix = get_channel_mix_suffix(
        original_key
    )

    if prefix is None:
        raise ValueError(
            f"Cannot determine Channel-Mix prefix "
            f"for {original_key}"
        )

    if suffix is None:
        raise ValueError(
            f"Cannot determine Channel-Mix suffix "
            f"for {original_key}"
        )

    return (
        f"{prefix}.experts."
        f"{expert_id}."
        f"{suffix}"
    )


def make_gate_key(
    channel_mix_prefix: str,
) -> str:

    return (
        f"{channel_mix_prefix}.gate.weight"
    )


def get_block_id(
    key: str,
) -> Optional[int]:

    match = re.search(
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
        key,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


# ============================================================
# SAFETENSOR CHECKPOINT
# ============================================================

class SafeTensorCheckpoint:
    """
    Lightweight safetensor checkpoint index.

    Does NOT load the entire checkpoint.

    Each tensor is loaded individually.
    """

    def __init__(
        self,
        directory: Path,
    ):

        self.directory = Path(
            directory
        )

        self.tensor_to_file: Dict[
            str,
            Path,
        ] = {}

        self.files: List[
            Path
        ] = []

        self._build_index()

    def _build_index(
        self,
    ) -> None:

        if not self.directory.exists():

            raise FileNotFoundError(
                self.directory
            )

        index_path = (
            self.directory
            / "model.safetensors.index.json"
        )

        # ----------------------------------------
        # Hugging Face sharded checkpoint
        # ----------------------------------------

        if index_path.exists():

            index = load_json(
                index_path
            )

            weight_map = (
                index.get(
                    "weight_map",
                    {},
                )
            )

            if not weight_map:

                raise RuntimeError(
                    f"Invalid safetensor index: "
                    f"{index_path}"
                )

            for key, filename in (
                weight_map.items()
            ):

                shard = (
                    self.directory
                    / filename
                )

                if not shard.exists():

                    raise FileNotFoundError(
                        f"Missing checkpoint shard: "
                        f"{shard}"
                    )

                self.tensor_to_file[
                    key
                ] = shard

            self.files = sorted(
                set(
                    self.tensor_to_file.values()
                ),
                key=lambda item:
                    natural_key(
                        item.name
                    ),
            )

            return

        # ----------------------------------------
        # Single or manually sharded checkpoint
        # ----------------------------------------

        files = sorted(
            self.directory.glob(
                "*.safetensors"
            ),
            key=lambda item:
                natural_key(
                    item.name
                ),
        )

        if not files:

            raise FileNotFoundError(
                f"No .safetensors files found in "
                f"{self.directory}"
            )

        for file_path in files:

            with safe_open(
                str(file_path),
                framework="pt",
                device="cpu",
            ) as handle:

                for key in handle.keys():

                    if key in self.tensor_to_file:

                        raise RuntimeError(
                            f"Duplicate tensor {key!r} "
                            f"found in checkpoint"
                        )

                    self.tensor_to_file[
                        key
                    ] = file_path

        self.files = files

    def keys(
        self,
    ) -> List[str]:

        return sorted(
            self.tensor_to_file.keys(),
            key=natural_key,
        )

    def has(
        self,
        key: str,
    ) -> bool:

        return key in self.tensor_to_file

    def get(
        self,
        key: str,
    ) -> torch.Tensor:

        path = self.tensor_to_file.get(
            key
        )

        if path is None:

            raise KeyError(
                f"Tensor not found: {key}"
            )

        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as handle:

            tensor = handle.get_tensor(
                key
            )

        return (
            tensor
            .detach()
            .cpu()
        )

    def metadata(
        self,
    ) -> Dict[str, str]:

        if not self.files:
            return {}

        with safe_open(
            str(self.files[0]),
            framework="pt",
            device="cpu",
        ) as handle:

            metadata = handle.metadata()

        if metadata is None:
            return {}

        return dict(
            metadata
        )


# ============================================================
# INCREMENTAL SAFETENSOR WRITER
# ============================================================

class ShardedSafeTensorWriter:
    """
    Writes Hugging Face-compatible safetensors incrementally.

    Only the current output shard is kept in memory.
    """

    def __init__(
        self,
        output_directory: Path,
        max_shard_size: int,
        metadata: Optional[
            Dict[str, str]
        ] = None,
    ):

        self.output_directory = Path(
            output_directory
        )

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.max_shard_size = (
            max_shard_size
        )

        self.metadata = (
            metadata
            or {"format": "pt"}
        )

        self.current_tensors = OrderedDict()

        self.current_size = 0

        self.shards: List[
            Tuple[Path, List[str]]
        ] = []

        self.shard_number = 0

    @staticmethod
    def tensor_size(
        tensor: torch.Tensor,
    ) -> int:

        return (
            tensor.numel()
            * tensor.element_size()
        )

    def add(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> None:

        tensor = (
            tensor
            .detach()
            .cpu()
            .contiguous()
        )

        size = self.tensor_size(
            tensor
        )

        if (
            self.current_tensors
            and self.current_size + size
            > self.max_shard_size
        ):

            self.flush()

        self.current_tensors[
            key
        ] = tensor

        self.current_size += size

    def flush(
        self,
    ) -> None:

        if not self.current_tensors:
            return

        self.shard_number += 1

        filename = (
            f"model-{self.shard_number:05d}"
            f".safetensors"
        )

        final_path = (
            self.output_directory
            / filename
        )

        temporary_path = Path(
            str(final_path)
            + ".tmp"
        )

        save_file(
            dict(
                self.current_tensors
            ),
            str(temporary_path),
            metadata=self.metadata,
        )

        os.replace(
            temporary_path,
            final_path,
        )

        keys = list(
            self.current_tensors.keys()
        )

        self.shards.append(
            (
                final_path,
                keys,
            )
        )

        self.current_tensors.clear()

        self.current_size = 0

        cleanup()

    def finalize(
        self,
    ) -> None:

        self.flush()

        if not self.shards:

            raise RuntimeError(
                "No output tensors were written"
            )

        # ----------------------------------------
        # Single file checkpoint
        # ----------------------------------------

        if len(self.shards) == 1:

            source = self.shards[0][0]

            destination = (
                self.output_directory
                / "model.safetensors"
            )

            if destination.exists():
                destination.unlink()

            os.replace(
                source,
                destination,
            )

            self.shards = [
                (
                    destination,
                    self.shards[0][1],
                )
            ]

            return

        # ----------------------------------------
        # Multi-shard checkpoint
        # ----------------------------------------

        weight_map = {}

        total_size = 0

        shard_count = len(
            self.shards
        )

        new_shards = []

        for index, (
            old_path,
            keys,
        ) in enumerate(
            self.shards,
            start=1,
        ):

            filename = (
                f"model-{index:05d}"
                f"-of-{shard_count:05d}"
                f".safetensors"
            )

            new_path = (
                self.output_directory
                / filename
            )

            os.replace(
                old_path,
                new_path,
            )

            total_size += (
                new_path.stat().st_size
            )

            for key in keys:

                weight_map[
                    key
                ] = filename

            new_shards.append(
                (
                    new_path,
                    keys,
                )
            )

        self.shards = new_shards

        index = {
            "metadata": {
                "total_size": total_size,
                "format": "pt",
            },
            "weight_map": weight_map,
        }

        save_json(
            index,
            self.output_directory
            / "model.safetensors.index.json",
        )


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class BranchSpec:

    path: Path
    weight: float


@dataclass
class MergeConfig:

    target_directory: Path

    base_model: Path

    algorithm: str

    density: float

    weight: float

    max_shard_size: int

    seed: int

    branches: List[
        BranchSpec
    ]


def load_yaml_config(
    path: Path,
) -> Dict:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        config = yaml.safe_load(
            handle
        )

    if not isinstance(
        config,
        dict,
    ):

        fatal(
            "The YAML configuration must "
            "contain a top-level mapping"
        )

    return config


def checkpoint_exists(
    directory: Path,
) -> bool:

    return (
        (
            directory
            / "model.safetensors"
        ).exists()
        or (
            directory
            / "model.safetensors.index.json"
        ).exists()
        or any(
            directory.glob(
                "*.safetensors"
            )
        )
    )


def discover_branches(
    raw_config: Dict,
) -> List[BranchSpec]:

    entries = raw_config.get(
        "addedexperts",
        [],
    )

    if not isinstance(
        entries,
        list,
    ):

        fatal(
            "'addedexperts' must be a YAML list"
        )

    result: List[
        BranchSpec
    ] = []

    for entry in entries:

        if not isinstance(
            entry,
            dict,
        ):

            fatal(
                "Every addedexperts entry "
                "must be a mapping"
            )

        raw_path = entry.get(
            "path"
        )

        if not raw_path:

            fatal(
                "Every addedexperts entry "
                "requires a path"
            )

        root = Path(
            raw_path
        ).expanduser().resolve()

        weight = float(
            entry.get(
                "weight",
                1.0,
            )
        )

        if not root.exists():

            fatal(
                f"Expert path does not exist: "
                f"{root}"
            )

        # ----------------------------------------
        # Path itself is a checkpoint
        # ----------------------------------------

        if checkpoint_exists(
            root
        ):

            result.append(
                BranchSpec(
                    path=root,
                    weight=weight,
                )
            )

            continue

        # ----------------------------------------
        # Search child folders alphabetically
        # ----------------------------------------

        children = sorted(
            [
                child
                for child in root.iterdir()

                if child.is_dir()
                and checkpoint_exists(
                    child
                )
            ],
            key=lambda child:
                natural_key(
                    child.name
                ),
        )

        for child in children:

            result.append(
                BranchSpec(
                    path=child,
                    weight=weight,
                )
            )

    if not result:

        fatal(
            "No expert checkpoints were found"
        )

    return result


def parse_merge_config(
    raw_config: Dict,
) -> MergeConfig:

    base_value = raw_config.get(
        "base_model"
    )

    if not base_value:

        fatal(
            "base_model is required"
        )

    target_value = raw_config.get(
        "target_directory",
        "./RWKV-Merged",
    )

    algorithm = str(
        raw_config.get(
            "algorithm",
            SUPPORTED_ALGORITHM,
        )
    ).lower()

    if algorithm != SUPPORTED_ALGORITHM:

        fatal(
            f"Unsupported algorithm: "
            f"{algorithm}"
        )

    density = float(
        raw_config.get(
            "density",
            1.0,
        )
    )

    if not (
        0.0 < density <= 1.0
    ):

        fatal(
            "density must be greater than 0 "
            "and less than or equal to 1"
        )

    weight = float(
        raw_config.get(
            "weight",
            1.0,
        )
    )

    max_shard_size = parse_size(
        raw_config.get(
            "max_shard_size",
            "512MB",
        )
    )

    seed = int(
        raw_config.get(
            "seed",
            42,
        )
    )

    branches = discover_branches(
        raw_config
    )

    return MergeConfig(
        target_directory=Path(
            target_value
        ).expanduser().resolve(),

        base_model=Path(
            base_value
        ).expanduser().resolve(),

        algorithm=algorithm,

        density=density,

        weight=weight,

        max_shard_size=max_shard_size,

        seed=seed,

        branches=branches,
    )


# ============================================================
# MODEL CONFIG HELPERS
# ============================================================

def infer_hidden_size(
    config: Dict,
    checkpoint: SafeTensorCheckpoint,
) -> Optional[int]:

    candidate_names = (
        "hidden_size",
        "n_embd",
        "d_model",
        "dim",
    )

    for name in candidate_names:

        value = config.get(
            name
        )

        if (
            isinstance(
                value,
                int,
            )
            and value > 0
        ):

            return value

    # ----------------------------------------
    # Infer from embedding
    # ----------------------------------------

    for key in checkpoint.keys():

        if is_embedding_parameter(
            key
        ):

            tensor = checkpoint.get(
                key
            )

            try:

                if tensor.ndim == 2:

                    return int(
                        tensor.shape[1]
                    )

            finally:

                del tensor
                cleanup()

    return None


def infer_vocab_size(
    config: Dict,
    checkpoint: SafeTensorCheckpoint,
) -> Optional[int]:

    value = config.get(
        "vocab_size"
    )

    if (
        isinstance(
            value,
            int,
        )
        and value > 0
    ):

        return value

    for key in checkpoint.keys():

        if is_embedding_parameter(
            key
        ):

            tensor = checkpoint.get(
                key
            )

            try:

                if tensor.ndim == 2:

                    return int(
                        tensor.shape[0]
                    )

            finally:

                del tensor
                cleanup()

    return None


# ============================================================
# TOKENIZER HANDLING
# ============================================================

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


def copy_tokenizer_files(
    source: Path,
    destination: Path,
) -> None:

    for filename in TOKENIZER_FILES:

        source_file = (
            source
            / filename
        )

        destination_file = (
            destination
            / filename
        )

        if source_file.exists():

            shutil.copy2(
                source_file,
                destination_file,
            )


def read_tokenizer_vocab(
    directory: Path,
) -> Optional[
    Dict[str, int]
]:

    tokenizer_path = (
        directory
        / "tokenizer.json"
    )

    if not tokenizer_path.exists():

        return None

    try:

        data = load_json(
            tokenizer_path
        )

    except Exception:

        return None

    model = data.get(
        "model",
        {}
    )

    vocab = model.get(
        "vocab"
    )

    if not isinstance(
        vocab,
        dict,
    ):

        return None

    result = {}

    for token, token_id in (
        vocab.items()
    ):

        if isinstance(
            token_id,
            int,
        ):

            result[token] = token_id

    return result


def merge_tokenizers(
    base_directory: Path,
    branches: List[
        BranchSpec
    ],
    output_directory: Path,
) -> Tuple[
    int,
    Dict
]:
    """
    Conservative tokenizer merging.

    Base IDs are always preserved.

    If branch tokenizers contain new tokens and the tokenizer uses
    a dictionary-style vocab, the new tokens are appended.

    If an existing token has a conflicting ID, the base tokenizer
    is preserved and the conflict is recorded.

    This avoids silently corrupting token IDs.
    """

    copy_tokenizer_files(
        base_directory,
        output_directory,
    )

    base_vocab = read_tokenizer_vocab(
        base_directory
    )

    if base_vocab is None:

        return (
            0,
            {
                "merged": False,
                "reason":
                    "Could not inspect tokenizer.json vocab",
                "added_tokens": [],
            },
        )

    merged_vocab = dict(
        base_vocab
    )

    next_id = (
        max(
            merged_vocab.values(),
            default=-1,
        )
        + 1
    )

    added_tokens = []

    conflicts = []

    for branch in branches:

        branch_vocab = (
            read_tokenizer_vocab(
                branch.path
            )
        )

        if branch_vocab is None:

            continue

        for token, token_id in (
            branch_vocab.items()
        ):

            if token in base_vocab:

                if (
                    base_vocab[token]
                    != token_id
                ):

                    conflicts.append(
                        {
                            "branch": str(
                                branch.path
                            ),
                            "token": token,
                            "base_id":
                                base_vocab[token],
                            "branch_id":
                                token_id,
                        }
                    )

        if conflicts:

            continue

        for token, token_id in sorted(
            branch_vocab.items(),
            key=lambda item:
                item[1],
        ):

            if token not in merged_vocab:

                merged_vocab[
                    token
                ] = next_id

                next_id += 1

                added_tokens.append(
                    token
                )

    # ----------------------------------------
    # Conflict detected
    # ----------------------------------------

    if conflicts:

        return (
            len(
                base_vocab
            ),
            {
                "merged": False,
                "reason":
                    "Tokenizer ID conflicts detected",
                "added_tokens": [],
                "conflicts": conflicts,
            },
        )

    # ----------------------------------------
    # Nothing new
    # ----------------------------------------

    if not added_tokens:

        return (
            len(
                base_vocab
            ),
            {
                "merged": True,
                "reason": None,
                "added_tokens": [],
                "conflicts": [],
            },
        )

    tokenizer_path = (
        output_directory
        / "tokenizer.json"
    )

    data = load_json(
        tokenizer_path
    )

    if not isinstance(
        data.get(
            "model",
            {},
        ).get(
            "vocab"
        ),
        dict,
    ):

        return (
            len(
                base_vocab
            ),
            {
                "merged": False,
                "reason":
                    "Tokenizer does not expose "
                    "a mutable dictionary vocabulary",
                "added_tokens": [],
                "conflicts": [],
            },
        )

    data[
        "model"
    ][
        "vocab"
    ] = merged_vocab

    save_json(
        data,
        tokenizer_path,
    )

    tokenizer_config_path = (
        output_directory
        / "tokenizer_config.json"
    )

    tokenizer_config = {}

    if tokenizer_config_path.exists():

        try:

            tokenizer_config = load_json(
                tokenizer_config_path
            )

        except Exception:

            tokenizer_config = {}

    tokenizer_config[
        "vocab_size"
    ] = len(
        merged_vocab
    )

    save_json(
        tokenizer_config,
        tokenizer_config_path,
    )

    return (
        len(
            merged_vocab
        ),
        {
            "merged": True,
            "reason": None,
            "added_tokens": added_tokens,
            "conflicts": [],
        },
    )


# ============================================================
# VOCABULARY MATRIX RESIZING
# ============================================================

def resize_vocab_matrix(
    tensor: torch.Tensor,
    target_size: int,
) -> torch.Tensor:

    if tensor.ndim < 1:

        return tensor

    current_size = (
        tensor.shape[0]
    )

    if target_size <= current_size:

        return tensor

    output_shape = (
        target_size,
        *tensor.shape[1:],
    )

    output = torch.empty(
        output_shape,
        dtype=tensor.dtype,
        device="cpu",
    )

    output[
        :current_size
    ].copy_(
        tensor
    )

    source = tensor.float()

    mean = source.mean()

    std = source.std(
        unbiased=False
    )

    if (
        not torch.isfinite(
            std
        )
        or std.item() <= 1e-12
    ):

        std = torch.tensor(
            0.02,
            dtype=torch.float32,
        )

    extra_shape = (
        target_size - current_size,
        *tensor.shape[1:],
    )

    generated = torch.normal(
        mean=float(
            mean.item()
        ),
        std=float(
            std.item()
        ),
        size=extra_shape,
        dtype=torch.float32,
    )

    output[
        current_size:
    ].copy_(
        generated.to(
            tensor.dtype
        )
    )

    return output


# ============================================================
# ROUTING
# ============================================================

def calculate_top_k(
    expert_count: int,
    density: float,
) -> int:

    return max(
        1,
        min(
            expert_count,
            int(
                math.ceil(
                    expert_count
                    * density
                )
            ),
        ),
    )


def create_router(
    expert_count: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:

    router = torch.empty(
        (
            expert_count,
            hidden_size,
        ),
        dtype=torch.float32,
        device="cpu",
    )

    torch.nn.init.normal_(
        router,
        mean=0.0,
        std=0.02,
    )

    return router.to(
        dtype
    )


# ============================================================
# CONFIG WRITING
# ============================================================

def write_output_config(
    base_directory: Path,
    output_directory: Path,
    expert_count: int,
    density: float,
    vocab_size: int,
    hidden_size: Optional[
        int
    ],
) -> None:

    source = (
        base_directory
        / "config.json"
    )

    if source.exists():

        config = load_json(
            source
        )

    else:

        config = {}

    if not isinstance(
        config,
        dict,
    ):

        config = {}

    top_k = calculate_top_k(
        expert_count,
        density,
    )

    if vocab_size > 0:

        config[
            "vocab_size"
        ] = vocab_size

    if (
        hidden_size is not None
        and "hidden_size"
        not in config
    ):

        config[
            "hidden_size"
        ] = hidden_size

    config[
        "is_moe"
    ] = True

    config[
        "num_experts"
    ] = expert_count

    config[
        "num_local_experts"
    ] = expert_count

    config[
        "num_experts_per_tok"
    ] = top_k

    config[
        "moe_top_k"
    ] = top_k

    config[
        "moe_expert_density"
    ] = density

    config[
        "rwkv_moe"
    ] = {
        "enabled": True,

        "architecture":
            "RWKV-X Channel-Mix MoE",

        "algorithm":
            "moe_upcycle",

        "num_experts":
            expert_count,

        "top_k":
            top_k,

        "density":
            density,

        "shared_modules": [
            "emb",
            "ln1",
            "time_mix",
            "ln2",
            "ln_out",
            "head",
        ],

        "expert_module":
            "channel_mix",

        "expert_layout":
            "blocks.N.channel_mix."
            "experts.E.PARAM",

        "router_layout":
            "blocks.N.channel_mix."
            "gate.weight",

        "router_initialization": {
            "distribution":
                "normal",

            "mean":
                0.0,

            "std":
                0.02,
        },
    }

    save_json(
        config,
        output_directory
        / "config.json",
    )


# ============================================================
# AUXILIARY FILE COPYING
# ============================================================

def copy_auxiliary_files(
    source: Path,
    destination: Path,
) -> None:

    files = (
        "generation_config.json",
        "README.md",
        "LICENSE",
        "LICENSE.txt",
    )

    for filename in files:

        source_file = (
            source
            / filename
        )

        if source_file.exists():

            shutil.copy2(
                source_file,
                destination
                / filename,
            )


# ============================================================
# CHECKPOINT VALIDATION
# ============================================================

def validate_output(
    output: SafeTensorCheckpoint,
    base: SafeTensorCheckpoint,
    expert_count: int,
) -> None:

    output_keys = set(
        output.keys()
    )

    # ----------------------------------------
    # Shared parameters
    # ----------------------------------------

    missing_shared = []

    for key in base.keys():

        if is_channel_mix_parameter(
            key
        ):

            continue

        if key not in output_keys:

            missing_shared.append(
                key
            )

    if missing_shared:

        raise RuntimeError(
            "Merged checkpoint is missing "
            f"{len(missing_shared)} shared tensors.\n"
            f"Examples: "
            f"{missing_shared[:10]}"
        )

    # ----------------------------------------
    # Expert parameters
    # ----------------------------------------

    missing_experts = []

    for key in base.keys():

        if not is_channel_mix_parameter(
            key
        ):

            continue

        for expert_id in range(
            expert_count
        ):

            expert_key = make_expert_key(
                key,
                expert_id,
            )

            if expert_key not in output_keys:

                missing_experts.append(
                    expert_key
                )

    if missing_experts:

        raise RuntimeError(
            "Merged checkpoint is missing "
            f"{len(missing_experts)} expert tensors.\n"
            f"Examples: "
            f"{missing_experts[:10]}"
        )

    # ----------------------------------------
    # Router parameters
    # ----------------------------------------

    prefixes = set()

    for key in base.keys():

        if is_channel_mix_parameter(
            key
        ):

            prefix = (
                get_channel_mix_prefix(
                    key
                )
            )

            if prefix:

                prefixes.add(
                    prefix
                )

    missing_gates = []

    for prefix in prefixes:

        gate_key = make_gate_key(
            prefix
        )

        if gate_key not in output_keys:

            missing_gates.append(
                gate_key
            )

    if missing_gates:

        raise RuntimeError(
            "Merged checkpoint is missing "
            f"{len(missing_gates)} routers.\n"
            f"Examples: "
            f"{missing_gates[:10]}"
        )


# ============================================================
# MAIN MERGE ENGINE
# ============================================================

def merge_moe_upcycle(
    merge_config: MergeConfig,
    base: SafeTensorCheckpoint,
    branches: List[
        SafeTensorCheckpoint
    ],
    output_directory: Path,
    target_vocab_size: int,
    hidden_size: int,
) -> None:
    """
    Streaming MoE conversion.

    RAM behavior:

    Shared tensor:
        load one tensor
        write it
        release it

    Expert tensor:
        load one branch tensor
        write it
        release it

    Router:
        one [experts, hidden_size] matrix
        per Channel-Mix layer
    """

    random.seed(
        merge_config.seed
    )

    torch.manual_seed(
        merge_config.seed
    )

    metadata = base.metadata()

    metadata.update(
        {
            "format": "pt",

            "rwkv_mergekit_version":
                VERSION,

            "merge_algorithm":
                "moe_upcycle",

            "rwkv_architecture":
                "RWKV-X-Channel-Mix-MoE",

            "num_experts":
                str(
                    len(
                        branches
                    )
                ),

            "expert_density":
                str(
                    merge_config.density
                ),
        }
    )

    writer = ShardedSafeTensorWriter(
        output_directory=output_directory,

        max_shard_size=(
            merge_config.max_shard_size
        ),

        metadata=metadata,
    )

    base_keys = base.keys()

    # ----------------------------------------
    # Classify tensors
    # ----------------------------------------

    shared_keys = []

    channel_mix_keys = []

    channel_mix_prefixes = set()

    for key in base_keys:

        if is_channel_mix_parameter(
            key
        ):

            channel_mix_keys.append(
                key
            )

            prefix = (
                get_channel_mix_prefix(
                    key
                )
            )

            if prefix:

                channel_mix_prefixes.add(
                    prefix
                )

        else:

            shared_keys.append(
                key
            )

    # ----------------------------------------
    # Progress
    # ----------------------------------------

    total_operations = (
        len(shared_keys)
        + len(channel_mix_keys)
        + len(channel_mix_prefixes)
    )

    progress = tqdm(
        total=total_operations,

        desc="RWKV-X MoE Merge",

        unit="operation",

        dynamic_ncols=True,
    )

    try:

        # ====================================================
        # SHARED PARAMETERS
        # ====================================================

        log(
            f"Writing {len(shared_keys):,} "
            f"shared tensors..."
        )

        for key in shared_keys:

            tensor = base.get(
                key
            )

            # --------------------------------
            # Vocabulary expansion
            # --------------------------------

            if (
                target_vocab_size > 0
                and (
                    is_embedding_parameter(
                        key
                    )
                    or is_head_parameter(
                        key
                    )
                )
            ):

                tensor = resize_vocab_matrix(
                    tensor,
                    target_vocab_size,
                )

            writer.add(
                key,
                tensor,
            )

            del tensor

            cleanup()

            progress.update(
                1
            )

            progress.set_postfix_str(
                memory_status()
            )

        # ====================================================
        # CHANNEL-MIX EXPERTS
        # ====================================================

        log(
            f"Converting "
            f"{len(channel_mix_keys):,} "
            f"Channel-Mix tensors into "
            f"{len(branches)} experts..."
        )

        for key in channel_mix_keys:

            expected_shape = None

            expected_dtype = None

            for expert_id, branch in enumerate(
                branches
            ):

                # ----------------------------
                # Branch parameter
                # ----------------------------

                if branch.has(
                    key
                ):

                    tensor = branch.get(
                        key
                    )

                # ----------------------------
                # Fallback to base
                # ----------------------------

                else:

                    warn(
                        f"Expert {expert_id} "
                        f"does not contain "
                        f"{key}. "
                        f"Using base parameter."
                    )

                    tensor = base.get(
                        key
                    )

                # ----------------------------
                # Shape validation
                # ----------------------------

                current_shape = tuple(
                    tensor.shape
                )

                if expected_shape is None:

                    expected_shape = (
                        current_shape
                    )

                    expected_dtype = (
                        tensor.dtype
                    )

                elif (
                    current_shape
                    != expected_shape
                ):

                    raise RuntimeError(
                        f"Expert shape mismatch "
                        f"for tensor {key!r}.\n"
                        f"Expected: "
                        f"{expected_shape}\n"
                        f"Got: "
                        f"{current_shape}\n"
                        f"Expert: "
                        f"{expert_id}"
                    )

                # ----------------------------
                # Expert key
                # ----------------------------

                output_key = make_expert_key(
                    key,
                    expert_id,
                )

                writer.add(
                    output_key,
                    tensor,
                )

                del tensor

                cleanup()

            progress.update(
                1
            )

            progress.set_postfix_str(
                memory_status()
            )

        # ====================================================
        # ROUTERS
        # ====================================================

        log(
            f"Creating "
            f"{len(channel_mix_prefixes):,} "
            f"Channel-Mix routers..."
        )

        for prefix in sorted(
            channel_mix_prefixes,
            key=natural_key,
        ):

            # Determine dtype from a parameter
            # inside this Channel-Mix module.

            prefix_keys = [
                key
                for key in channel_mix_keys

                if get_channel_mix_prefix(
                    key
                )
                == prefix
            ]

            if not prefix_keys:

                raise RuntimeError(
                    f"No parameters found for "
                    f"{prefix}"
                )

            reference_key = (
                prefix_keys[0]
            )

            reference_tensor = base.get(
                reference_key
            )

            dtype = (
                reference_tensor.dtype
            )

            del reference_tensor

            cleanup()

            router = create_router(
                expert_count=len(
                    branches
                ),

                hidden_size=hidden_size,

                dtype=dtype,
            )

            gate_key = make_gate_key(
                prefix
            )

            writer.add(
                gate_key,
                router,
            )

            del router

            cleanup()

            progress.update(
                1
            )

            progress.set_postfix_str(
                memory_status()
            )

    finally:

        progress.close()

    writer.finalize()


# ============================================================
# MERGE METADATA
# ============================================================

def write_merge_metadata(
    merge_config: MergeConfig,
    output_directory: Path,
    expert_count: int,
    hidden_size: int,
    vocab_size: int,
    tokenizer_info: Dict,
) -> None:

    metadata = {
        "engine":
            "RWKV MergeKit",

        "version":
            VERSION,

        "algorithm":
            merge_config.algorithm,

        "base_model":
            str(
                merge_config.base_model
            ),

        "target_directory":
            str(
                merge_config.target_directory
            ),

        "num_experts":
            expert_count,

        "density":
            merge_config.density,

        "top_k":
            calculate_top_k(
                expert_count,
                merge_config.density,
            ),

        "hidden_size":
            hidden_size,

        "vocab_size":
            vocab_size,

        "seed":
            merge_config.seed,

        "max_shard_size":
            merge_config.max_shard_size,

        "weight":
            merge_config.weight,

        "branches": [
            {
                "expert_id":
                    expert_id,

                "path":
                    str(
                        branch.path
                    ),

                "weight":
                    branch.weight,
            }

            for expert_id, branch
            in enumerate(
                merge_config.branches
            )
        ],

        "tokenizer":
            tokenizer_info,

        "shared_parameters": {
            "embeddings":
                "base",

            "time_mix":
                "base",

            "layer_norm":
                "base",

            "output":
                "base",
        },

        "expert_parameters": {
            "module":
                "channel_mix",

            "layout":
                "blocks.N.channel_mix."
                "experts.E.PARAM",

            "parameters": [
                "key.*",
                "value.*",
                "receptance.*",
                "time_mix_k",
                "time_mix_r",
            ],
        },

        "router": {
            "layout":
                "blocks.N.channel_mix."
                "gate.weight",

            "shape":
                [
                    expert_count,
                    hidden_size,
                ],

            "initialization":
                {
                    "distribution":
                        "normal",

                    "mean":
                        0.0,

                    "std":
                        0.02,
                },
        },
    }

    save_json(
        metadata,
        output_directory
        / "merge_config.json",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "RWKV-X MergeKit MoE "
            "Upcycling Engine"
        )
    )

    parser.add_argument(
        "--config",

        required=True,

        type=str,

        help=(
            "Path to merge YAML "
            "configuration"
        ),
    )

    args = parser.parse_args()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    if not config_path.exists():

        fatal(
            f"Config does not exist: "
            f"{config_path}"
        )

    raw_config = load_yaml_config(
        config_path
    )

    merge_config = parse_merge_config(
        raw_config
    )

    # --------------------------------------------------------
    # Validate base
    # --------------------------------------------------------

    if not merge_config.base_model.exists():

        fatal(
            f"Base model directory "
            f"does not exist: "
            f"{merge_config.base_model}"
        )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    output_directory = (
        merge_config.target_directory
    )

    if output_directory.exists():

        existing_files = list(
            output_directory.iterdir()
        )

        if existing_files:

            fatal(
                f"Output directory already exists "
                f"and is not empty:\n"
                f"{output_directory}\n\n"
                f"Choose another target_directory "
                f"or empty this directory first."
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Startup information
    # --------------------------------------------------------

    log(
        f"RWKV MergeKit v{VERSION}"
    )

    log(
        f"Base model: "
        f"{merge_config.base_model}"
    )

    log(
        f"Target directory: "
        f"{output_directory}"
    )

    log(
        f"Algorithm: "
        f"{merge_config.algorithm}"
    )

    log(
        f"Expert density: "
        f"{merge_config.density}"
    )

    log(
        f"Max shard size: "
        f"{human_bytes(merge_config.max_shard_size)}"
    )

    log(
        f"Initial {memory_status()}"
    )

    # --------------------------------------------------------
    # Branch information
    # --------------------------------------------------------

    log(
        f"Discovered "
        f"{len(merge_config.branches)} "
        f"expert checkpoints:"
    )

    for expert_id, branch in enumerate(
        merge_config.branches
    ):

        log(
            f"  Expert {expert_id}: "
            f"{branch.path.name} "
            f"({branch.path})"
        )

    # --------------------------------------------------------
    # Index base checkpoint
    # --------------------------------------------------------

    log(
        "Indexing base checkpoint..."
    )

    base_checkpoint = (
        SafeTensorCheckpoint(
            merge_config.base_model
        )
    )

    log(
        f"Base tensor count: "
        f"{len(base_checkpoint.keys()):,}"
    )

    # --------------------------------------------------------
    # Index branches
    # --------------------------------------------------------

    branch_checkpoints = []

    for expert_id, branch in enumerate(
        merge_config.branches
    ):

        log(
            f"Indexing expert "
            f"{expert_id}: "
            f"{branch.path.name}"
        )

        checkpoint = (
            SafeTensorCheckpoint(
                branch.path
            )
        )

        branch_checkpoints.append(
            checkpoint
        )

        log(
            f"  tensors: "
            f"{len(checkpoint.keys()):,}"
        )

    # --------------------------------------------------------
    # Load base config
    # --------------------------------------------------------

    base_config_path = (
        merge_config.base_model
        / "config.json"
    )

    if base_config_path.exists():

        base_model_config = load_json(
            base_config_path
        )

    else:

        warn(
            "Base config.json was not found. "
            "Hidden size will be inferred "
            "from emb.weight."
        )

        base_model_config = {}

    if not isinstance(
        base_model_config,
        dict,
    ):

        base_model_config = {}

    # --------------------------------------------------------
    # Hidden size
    # --------------------------------------------------------

    hidden_size = infer_hidden_size(
        base_model_config,
        base_checkpoint,
    )

    if hidden_size is None:

        fatal(
            "Could not determine hidden_size "
            "from config.json or emb.weight."
        )

    log(
        f"Hidden size: "
        f"{hidden_size:,}"
    )

    # --------------------------------------------------------
    # Base vocabulary
    # --------------------------------------------------------

    base_vocab_size = infer_vocab_size(
        base_model_config,
        base_checkpoint,
    )

    if base_vocab_size is None:

        base_vocab_size = 0

    log(
        f"Base vocabulary size: "
        f"{base_vocab_size:,}"
    )

    # --------------------------------------------------------
    # Tokenizer merging
    # --------------------------------------------------------

    log(
        "Processing tokenizers..."
    )

    merged_vocab_size, tokenizer_info = (
        merge_tokenizers(
            base_directory=(
                merge_config.base_model
            ),

            branches=(
                merge_config.branches
            ),

            output_directory=(
                output_directory
            ),
        )
    )

    if merged_vocab_size <= 0:

        merged_vocab_size = (
            base_vocab_size
        )

    if tokenizer_info.get(
        "merged"
    ):

        log(
            f"Tokenizer vocabulary size: "
            f"{merged_vocab_size:,}"
        )

        added_tokens = (
            tokenizer_info.get(
                "added_tokens",
                [],
            )
        )

        if added_tokens:

            log(
                f"Added {len(added_tokens):,} "
                f"new tokenizer tokens."
            )

    else:

        warn(
            f"Tokenizer merge skipped: "
            f"{tokenizer_info.get('reason')}"
        )

    # --------------------------------------------------------
    # Write MoE config
    # --------------------------------------------------------

    write_output_config(
        base_directory=(
            merge_config.base_model
        ),

        output_directory=(
            output_directory
        ),

        expert_count=len(
            branch_checkpoints
        ),

        density=(
            merge_config.density
        ),

        vocab_size=(
            merged_vocab_size
        ),

        hidden_size=hidden_size,
    )

    # --------------------------------------------------------
    # Copy other Hugging Face files
    # --------------------------------------------------------

    copy_auxiliary_files(
        merge_config.base_model,
        output_directory,
    )

    # --------------------------------------------------------
    # Execute merge
    # --------------------------------------------------------

    log(
        "Starting RWKV-X MoE upcycling..."
    )

    merge_moe_upcycle(
        merge_config=merge_config,

        base=base_checkpoint,

        branches=branch_checkpoints,

        output_directory=(
            output_directory
        ),

        target_vocab_size=(
            merged_vocab_size
        ),

        hidden_size=hidden_size,
    )

    # --------------------------------------------------------
    # Validate result
    # --------------------------------------------------------

    log(
        "Validating merged checkpoint..."
    )

    output_checkpoint = (
        SafeTensorCheckpoint(
            output_directory
        )
    )

    validate_output(
        output=output_checkpoint,

        base=base_checkpoint,

        expert_count=len(
            branch_checkpoints
        ),
    )

    # --------------------------------------------------------
    # Write metadata
    # --------------------------------------------------------

    write_merge_metadata(
        merge_config=merge_config,

        output_directory=(
            output_directory
        ),

        expert_count=len(
            branch_checkpoints
        ),

        hidden_size=hidden_size,

        vocab_size=(
            merged_vocab_size
        ),

        tokenizer_info=tokenizer_info,
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    top_k = calculate_top_k(
        len(
            branch_checkpoints
        ),
        merge_config.density,
    )

    log("")
    log(
        "=" * 60
    )

    log(
        "MERGE COMPLETED SUCCESSFULLY"
    )

    log(
        "=" * 60
    )

    log(
        f"Experts: "
        f"{len(branch_checkpoints)}"
    )

    log(
        f"Density: "
        f"{merge_config.density}"
    )

    log(
        f"Top-K experts/token: "
        f"{top_k}"
    )

    log(
        f"Hidden size: "
        f"{hidden_size:,}"
    )

    log(
        f"Vocabulary size: "
        f"{merged_vocab_size:,}"
    )

    log(
        f"Output tensors: "
        f"{len(output_checkpoint.keys()):,}"
    )

    log(
        f"Final {memory_status()}"
    )

    log(
        f"Output: "
        f"{output_directory}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[RWKV-MergeKit] "
            "Merge interrupted by user.",
            file=sys.stderr,
        )

        raise SystemExit(
            130
        )

    except SystemExit:

        raise

    except Exception as error:

        print(
            f"\n[RWKV-MergeKit ERROR] "
            f"{type(error).__name__}: "
            f"{error}",
            file=sys.stderr,
        )

        raise

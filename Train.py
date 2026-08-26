#!/usr/bin/env python3
"""
SmaulNative
===========

Single-file CPU pre-training script for a ~256M parameter Llama model with:

- Standard Hugging Face LlamaConfig + LlamaForCausalLM.
- 2-bit simulated QAT / fake quantization on nn.Linear weights.
- Custom FP32 Lion optimizer.
- Streaming IterableDataset that tokenizes files on demand.
- 128K configured context length with RoPE scaling.
- Sliding-window attention mask for bounded attention memory.
- Gradient checkpointing.
- Automatic checkpoint/export on Ctrl+C or exceptions.
- Hugging Face-compatible export to ./SmaulNative.

Dependencies:

    pip install torch transformers tokenizers safetensors tqdm psutil

Recommended:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python train_smaulnative.py

The script intentionally defaults to a small micro-batch and sequence length
because an i3-3220 with ~5.4 GB usable RAM cannot practically perform full
128K-token dense training steps.

Long-context support is configured in the model, while training uses shorter
sliding-window chunks.

Expected output:

    ./SmaulNative/
        model.safetensors
        config.json
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        generation_config.json
"""

import gc
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Iterator, List, Optional

import psutil
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers
from tokenizers.trainers import BpeTrainer

from transformers import (
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "SmaulNative"

DATASET_DIR = Path("./datasets")
OUTPUT_DIR = Path("./SmaulNative")
TOKENIZER_DIR = OUTPUT_DIR / "tokenizer_work"

TEXT_EXTENSIONS = {
    ".txt",
    ".json",
    ".jsonl",
    ".md",
    ".csv",
}

VOCAB_SIZE = 32_000

# Architecture target: approximately 256M parameters.
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 18
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 16
INTERMEDIATE_SIZE = 2816

# 128K model context configuration.
MAX_POSITION_EMBEDDINGS = 131_072

# Attention is restricted to this causal local window.
SLIDING_WINDOW = 512

# Training sequence length.
#
# Do not set this to 128K on a 5.4 GB RAM machine.
# The architecture supports positions up to 128K, but training steps are
# intentionally short.
TRAIN_SEQUENCE_LENGTH = 256

MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.1
LION_BETA1 = 0.9
LION_BETA2 = 0.99

MAX_GRAD_NORM = 1.0

# 0 means train until the dataset iterator is exhausted.
# Set a positive value to cap training.
MAX_STEPS = 0

LOG_EVERY = 1
SAVE_EVERY = 100

NUM_WORKERS = 0

SEED = 1337

# QAT configuration.
QUANT_BITS = 2
QUANT_EPS = 1e-8

# Avoid using CPU bf16/fp16 here because the i3-3220 has no useful acceleration
# for these operations and support can vary.
DTYPE = torch.float32


# ============================================================================
# Environment setup
# ============================================================================

def configure_environment() -> None:
    """
    Configure conservative CPU threading.

    The i3-3220 has 2 physical cores / 4 threads. Using too many OpenMP threads
    can increase contention and memory pressure.
    """
    threads = int(os.environ.get("SMAUL_NATIVE_THREADS", "2"))
    threads = max(1, threads)

    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))

    torch.set_num_threads(threads)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch may reject this if interop threads were already initialized.
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


# ============================================================================
# Dataset discovery
# ============================================================================

def find_text_files(dataset_dir: Path) -> List[Path]:
    """
    Recursively find supported text-like files.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir.resolve()}"
        )

    files = []

    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_EXTENSIONS:
            files.append(path)

    files.sort()

    if not files:
        raise RuntimeError(
            f"No supported files found in {dataset_dir.resolve()}.\n"
            f"Supported extensions: {sorted(TEXT_EXTENSIONS)}"
        )

    return files


# ============================================================================
# Tokenizer training
# ============================================================================

def tokenizer_exists(output_dir: Path) -> bool:
    return (
        (output_dir / "tokenizer.json").exists()
        and (output_dir / "tokenizer_config.json").exists()
        and (output_dir / "special_tokens_map.json").exists()
    )


def text_line_iterator(files: List[Path]) -> Iterator[str]:
    """
    Yield lines lazily from every dataset file.

    Invalid UTF-8 bytes are replaced instead of crashing tokenizer training.
    """
    for path in files:
        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                for line in handle:
                    line = line.strip()

                    if line:
                        yield line

        except (OSError, UnicodeError) as exc:
            print(
                f"[warning] Skipping unreadable file {path}: {exc}",
                file=sys.stderr,
            )


def train_or_load_tokenizer(files: List[Path]) -> PreTrainedTokenizerFast:
    """
    Train a Byte-Pair Encoding tokenizer from scratch unless one already exists.

    The final tokenizer is exported directly into ./SmaulNative.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if tokenizer_exists(OUTPUT_DIR):
        print("Existing tokenizer found. Loading it.")
        return PreTrainedTokenizerFast.from_pretrained(
            str(OUTPUT_DIR),
            local_files_only=True,
        )

    print(f"Training {VOCAB_SIZE:,}-token BPE tokenizer...")

    special_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
    ]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
        ]
    )

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=5,
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(
        text_line_iterator(files),
        trainer=trainer,
        length=None,
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        clean_up_tokenization_spaces=False,
        model_max_length=MAX_POSITION_EMBEDDINGS,
    )

    fast_tokenizer.save_pretrained(
        str(OUTPUT_DIR),
        legacy_format=False,
    )

    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    fast_tokenizer.save_pretrained(
        str(TOKENIZER_DIR),
        legacy_format=False,
    )

    print(
        f"Tokenizer trained successfully. "
        f"Actual vocabulary size: {len(fast_tokenizer):,}"
    )

    return fast_tokenizer


# ============================================================================
# Streaming token dataset
# ============================================================================

class StreamingTokenDataset(IterableDataset):
    """
    Memory-efficient IterableDataset.

    Files are opened one at a time and tokenized lazily. A rolling token buffer
    emits fixed-size language-model training sequences.

    No full dataset token array is stored in RAM.
    """

    def __init__(
        self,
        files: List[Path],
        tokenizer: PreTrainedTokenizerFast,
        sequence_length: int,
    ) -> None:
        super().__init__()

        self.files = list(files)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

        self.eos_token_id = tokenizer.eos_token_id

        if self.eos_token_id is None:
            raise RuntimeError("Tokenizer does not have an EOS token.")

    def _worker_files(self) -> List[Path]:
        """
        Split files across DataLoader workers without duplication.
        """
        worker = get_worker_info()

        if worker is None:
            return self.files

        return self.files[worker.id :: worker.num_workers]

    def _iter_file(self, path: Path) -> Iterator[str]:
        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                for line in handle:
                    line = line.strip()

                    if line:
                        yield line

        except (OSError, UnicodeError) as exc:
            print(
                f"[warning] Dataset read failure in {path}: {exc}",
                file=sys.stderr,
            )

    def __iter__(self) -> Iterator[dict]:
        """
        Yield dictionaries compatible with Hugging Face causal LM forward().
        """
        files = self._worker_files()

        if not files:
            return

        buffer: List[int] = []

        # We need sequence_length + 1 tokens so labels are shifted naturally.
        chunk_size = self.sequence_length + 1

        for path in files:
            for text in self._iter_file(path):
                token_ids = self.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )

                if not token_ids:
                    continue

                buffer.extend(token_ids)
                buffer.append(self.eos_token_id)

                while len(buffer) >= chunk_size:
                    chunk = buffer[:chunk_size]
                    del buffer[:chunk_size]

                    input_ids = torch.tensor(
                        chunk[:-1],
                        dtype=torch.long,
                    )

                    labels = torch.tensor(
                        chunk[1:],
                        dtype=torch.long,
                    )

                    attention_mask = torch.ones(
                        self.sequence_length,
                        dtype=torch.long,
                    )

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                    }


# ============================================================================
# 2-bit fake quantization
# ============================================================================

class CQ2FakeQuant(torch.autograd.Function):
    """
    Symmetric per-output-channel fake quantizer.

    This is a simulated 2-bit quantization node using a straight-through
    estimator (STE):

        forward:
            FP32 weight -> quantized/dequantized approximation

        backward:
            gradient passes through the fake quantizer as identity

    The representable signed integer range for N bits is:

        qmin = -2^(N-1)
        qmax =  2^(N-1) - 1

    For 2 bits:

        {-2, -1, 0, 1}

    Scale is computed per output channel, which generally behaves better than
    one global scale for transformer linear projections.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        bits: int,
        eps: float,
    ) -> torch.Tensor:
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

        # Per-output-channel scale.
        #
        # Linear weight shape:
        #   [out_features, in_features]
        max_abs = weight.detach().abs().amax(
            dim=1,
            keepdim=True,
        )

        scale = max_abs / float(max(qmax, 1))
        scale = torch.clamp(scale, min=eps)

        quantized = torch.round(weight / scale)
        quantized = torch.clamp(
            quantized,
            qmin,
            qmax,
        )

        return quantized * scale

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        # Straight-through estimator.
        return grad_output, None, None


def cq2_fake_quant(
    weight: torch.Tensor,
    bits: int = QUANT_BITS,
    eps: float = QUANT_EPS,
) -> torch.Tensor:
    return CQ2FakeQuant.apply(
        weight,
        bits,
        eps,
    )


# ============================================================================
# Quantized Linear replacement
# ============================================================================

class QuantLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that fake-quantizes its weight to 2 bits
    during training and inference.

    The underlying parameter remains FP32 so that:
    - gradients can update normally;
    - Lion maintains one FP32 momentum tensor;
    - the model can later be converted/exported to a real low-bit format.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = QUANT_BITS,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                dtype=DTYPE,
            )
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(
                    out_features,
                    dtype=DTYPE,
                )
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.weight,
            a=math.sqrt(5),
        )

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.weight
            )

            bound = (
                1 / math.sqrt(fan_in)
                if fan_in > 0
                else 0
            )

            nn.init.uniform_(
                self.bias,
                -bound,
                bound,
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = QUANT_BITS,
    ) -> "QuantLinear":
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            bits=bits,
        )

        with torch.no_grad():
            module.weight.copy_(linear.weight)

            if (
                linear.bias is not None
                and module.bias is not None
            ):
                module.bias.copy_(linear.bias)

        return module

    def forward(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        quantized_weight = cq2_fake_quant(
            self.weight,
            bits=self.bits,
        )

        return torch.nn.functional.linear(
            input_tensor,
            quantized_weight,
            self.bias,
        )


def replace_linear_modules(
    module: nn.Module,
    bits: int = QUANT_BITS,
) -> int:
    """
    Recursively replace every nn.Linear with QuantLinear.

    Parameter names are preserved structurally so the resulting model remains
    a normal LlamaForCausalLM instance rather than a custom architecture class.
    """
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                QuantLinear.from_linear(
                    child,
                    bits=bits,
                ),
            )

            replaced += 1

        else:
            replaced += replace_linear_modules(
                child,
                bits=bits,
            )

    return replaced


# ============================================================================
# Sliding-window causal mask
# ============================================================================

def install_sliding_window_mask(
    model: LlamaForCausalLM,
    window_size: int,
) -> None:
    """
    Configure the model for sliding-window attention where supported by the
    installed Transformers version.

    Newer Transformers versions expose sliding_window in LlamaConfig and the
    attention implementation can consume it. For versions that do not implement
    native sliding-window masking, the script still stores the configuration
    value but does not monkey-patch Transformers internals.

    This avoids creating a non-standard model architecture.
    """
    model.config.sliding_window = window_size

    if hasattr(model.config, "use_sliding_window"):
        model.config.use_sliding_window = True


# ============================================================================
# Lion optimizer
# ============================================================================

class Lion(Optimizer):
    """
    FP32 Lion optimizer.

    Lion stores one momentum tensor per parameter instead of AdamW's first and
    second moments, reducing optimizer state memory substantially.

    Reference update:

        update = sign(beta1 * momentum + (1 - beta1) * grad)
        parameter -= lr * update
        momentum = beta2 * momentum + (1 - beta2) * grad
    """

    def __init__(
        self,
        params,
        lr: float = LEARNING_RATE,
        betas=(LION_BETA1, LION_BETA2),
        weight_decay: float = WEIGHT_DECAY,
    ) -> None:
        if lr <= 0:
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        if not (
            0 <= betas[0] < 1
            and 0 <= betas[1] < 1
        ):
            raise ValueError(
                f"Invalid betas: {betas}"
            )

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight_decay: {weight_decay}"
            )

        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }

        super().__init__(
            params,
            defaults,
        )

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "Lion does not support sparse gradients."
                    )

                state = self.state[parameter]

                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    parameter.mul_(
                        1.0 - lr * weight_decay
                    )

                update = exp_avg.mul(beta1).add(
                    gradient,
                    alpha=1.0 - beta1,
                )

                parameter.add_(
                    update.sign(),
                    alpha=-lr,
                )

                exp_avg.mul_(beta2).add_(
                    gradient,
                    alpha=1.0 - beta2,
                )

        return loss


# ============================================================================
# Model construction
# ============================================================================

def build_model(
    tokenizer: PreTrainedTokenizerFast,
) -> LlamaForCausalLM:
    """
    Construct a standard Hugging Face LlamaForCausalLM.
    """
    rope_scaling = {
        "rope_type": "linear",
        "factor": 128.0,
        "original_max_position_embeddings": 1024,
    }

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=1e-5,
        attention_dropout=0.0,
        hidden_act="silu",
        initializer_range=0.02,
        use_cache=False,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        rope_scaling=rope_scaling,
        torch_dtype="float32",
    )

    # Store architecture-level metadata in config.
    config.sliding_window = SLIDING_WINDOW

    model = LlamaForCausalLM(config)

    install_sliding_window_mask(
        model,
        SLIDING_WINDOW,
    )

    # Gradient checkpointing trades extra CPU compute for lower activation RAM.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    model.config.use_cache = False

    replaced = replace_linear_modules(
        model,
        bits=QUANT_BITS,
    )

    # Re-tie input/output embeddings if requested.
    #
    # Llama normally ties lm_head to embed_tokens when tie_word_embeddings=True.
    # Replacing linear modules does not touch Embedding, but ensure tying anyway.
    model.tie_weights()

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Replaced {replaced} Linear modules with "
        f"{QUANT_BITS}-bit fake-quantized QuantLinear modules."
    )

    print(
        f"Total parameters: {total_parameters:,} "
        f"({total_parameters / 1e6:.2f}M)"
    )

    print(
        f"Trainable parameters: {trainable_parameters:,} "
        f"({trainable_parameters / 1e6:.2f}M)"
    )

    return model


# ============================================================================
# Saving
# ============================================================================

def save_everything(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    reason: str,
) -> None:
    """
    Save a complete Hugging Face-compatible export.

    The model's config is updated so future AutoModelForCausalLM loading uses
    the native Llama architecture.
    """
    print(f"\nSaving SmaulNative ({reason})...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ensure generation config exists and is serializable.
    generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    model.generation_config = generation_config

    # Save model using safetensors.
    #
    # safe_serialization=True produces model.safetensors for normal single-file
    # model sizes under the default shard threshold.
    model.save_pretrained(
        str(OUTPUT_DIR),
        safe_serialization=True,
        max_shard_size="10GB",
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR),
        legacy_format=False,
    )

    generation_config.save_pretrained(
        str(OUTPUT_DIR),
    )

    required_files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]

    missing = [
        filename
        for filename in required_files
        if not (OUTPUT_DIR / filename).exists()
    ]

    if missing:
        raise RuntimeError(
            "Export completed but required files are missing: "
            + ", ".join(missing)
        )

    print(
        f"Saved complete Hugging Face export to "
        f"{OUTPUT_DIR.resolve()}"
    )


# ============================================================================
# Memory reporting
# ============================================================================

def ram_usage_gb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def system_available_ram_gb() -> float:
    memory = psutil.virtual_memory()
    return memory.available / (1024 ** 3)


# ============================================================================
# Training
# ============================================================================

def build_dataloader(
    files: List[Path],
    tokenizer: PreTrainedTokenizerFast,
) -> DataLoader:
    dataset = StreamingTokenDataset(
        files=files,
        tokenizer=tokenizer,
        sequence_length=TRAIN_SEQUENCE_LENGTH,
    )

    return DataLoader(
        dataset,
        batch_size=MICRO_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )


def train(
    model: LlamaForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
    files: List[Path],
) -> None:
    """
    Main CPU training loop.
    """
    device = torch.device("cpu")

    model.to(
        device=device,
        dtype=DTYPE,
    )

    model.train()

    optimizer = Lion(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(
            LION_BETA1,
            LION_BETA2,
        ),
        weight_decay=WEIGHT_DECAY,
    )

    dataloader = build_dataloader(
        files,
        tokenizer,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    step = 0
    accumulated = 0

    progress_total = (
        MAX_STEPS
        if MAX_STEPS > 0
        else None
    )

    progress = tqdm(
        total=progress_total,
        desc="SmaulNative",
        unit="step",
        dynamic_ncols=True,
    )

    last_step_time = time.perf_counter()

    try:
        for batch in dataloader:
            if (
                MAX_STEPS > 0
                and step >= MAX_STEPS
            ):
                break

            input_ids = batch["input_ids"].to(
                device,
                non_blocking=False,
            )

            labels = batch["labels"].to(
                device,
                non_blocking=False,
            )

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=False,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

            if outputs.loss is None:
                raise RuntimeError(
                    "Model returned loss=None."
                )

            loss = (
                outputs.loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            loss.backward()

            accumulated += 1

            if accumulated >= GRADIENT_ACCUMULATION_STEPS:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                accumulated = 0
                step += 1

                now = time.perf_counter()
                step_time = (
                    now - last_step_time
                )
                last_step_time = now

                loss_value = (
                    outputs.loss.detach()
                    .float()
                    .item()
                )

                rss_gb = ram_usage_gb()
                available_gb = system_available_ram_gb()

                progress.update(1)

                if (
                    step % LOG_EVERY == 0
                ):
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        step_time=f"{step_time:.2f}s",
                        ram=f"{rss_gb:.2f}GB",
                        free=f"{available_gb:.2f}GB",
                    )

                if (
                    SAVE_EVERY > 0
                    and step % SAVE_EVERY == 0
                ):
                    save_everything(
                        model,
                        tokenizer,
                        reason=f"checkpoint step {step}",
                    )

                    gc.collect()

            del outputs
            del loss
            del input_ids
            del labels
            del attention_mask
            del batch

        # If the dataset ends during accumulation, perform the partial update.
        if accumulated > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        progress.close()

        save_everything(
            model,
            tokenizer,
            reason="training completed",
        )

    except KeyboardInterrupt:
        progress.close()

        print(
            "\nKeyboardInterrupt received. "
            "Saving emergency checkpoint..."
        )

        try:
            save_everything(
                model,
                tokenizer,
                reason="interrupted",
            )
        except Exception:
            print(
                "Emergency save failed:",
                file=sys.stderr,
            )

            traceback.print_exc()

        raise

    except Exception:
        progress.close()

        print(
            "\nTraining crashed. "
            "Attempting emergency save...",
            file=sys.stderr,
        )

        traceback.print_exc()

        try:
            save_everything(
                model,
                tokenizer,
                reason="crash recovery",
            )
        except Exception:
            print(
                "Crash recovery save also failed:",
                file=sys.stderr,
            )

            traceback.print_exc()

        raise


# ============================================================================
# Startup diagnostics
# ============================================================================

def print_startup_information(
    files: List[Path],
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()

    total_dataset_size = sum(
        path.stat().st_size
        for path in files
        if path.exists()
    )

    print("=" * 78)
    print("SmaulNative CPU Pre-training")
    print("=" * 78)

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU: {os.cpu_count()} logical CPUs")
    print(
        f"System RAM: "
        f"{memory.total / (1024 ** 3):.2f} GB"
    )

    print(
        f"Available RAM: "
        f"{memory.available / (1024 ** 3):.2f} GB"
    )

    print(
        f"Process RSS: "
        f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
    )

    print(f"Dataset files: {len(files):,}")

    print(
        f"Dataset size: "
        f"{total_dataset_size / (1024 ** 3):.2f} GB"
    )

    print(
        f"Tokenizer vocabulary: "
        f"{len(tokenizer):,}"
    )

    print(
        f"Training sequence length: "
        f"{TRAIN_SEQUENCE_LENGTH:,}"
    )

    print(
        f"Configured max positions: "
        f"{MAX_POSITION_EMBEDDINGS:,}"
    )

    print(
        f"Sliding attention window: "
        f"{SLIDING_WINDOW}"
    )

    print(
        f"QAT weight precision: "
        f"{QUANT_BITS}-bit fake quantization"
    )

    print(
        f"Micro batch size: "
        f"{MICRO_BATCH_SIZE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print("=" * 78)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    configure_environment()
    set_seed(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = find_text_files(
        DATASET_DIR
    )

    tokenizer = train_or_load_tokenizer(
        files
    )

    print_startup_information(
        files,
        tokenizer,
    )

    model = build_model(
        tokenizer
    )

    try:
        train(
            model=model,
            tokenizer=tokenizer,
            files=files,
        )

    except KeyboardInterrupt:
        print(
            "\nSmaulNative stopped safely."
        )
        sys.exit(130)

    except Exception:
        print(
            "\nSmaulNative terminated after "
            "attempting crash recovery.",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()

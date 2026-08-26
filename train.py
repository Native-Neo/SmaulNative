#!/usr/bin/env python3
"""
train.py

RWKV-style causal language model trainer with:

- Hugging Face PreTrainedModel / PretrainedConfig compatibility
- save_pretrained() using safetensors
- AutoModelForCausalLM registration in the current process
- RWKV-style recurrent linear attention
- State passing for arbitrarily long streams
- Lion optimizer
- Streaming datasets from:
    *.txt
    *.jsonl
    *.json
    *.csv
    *.parquet
    source code files
- Byte-level resume for seekable text files
- Dataset transition with --new-data
- Optimizer resume
- Ctrl-C-only checkpoint saving

Install:

    pip install torch transformers tokenizers safetensors tqdm pandas pyarrow

Example:

    python train.py \
        --dataset_dir ./datasets \
        --output_dir ./SmaulNative-RWKV \
        --checkpoint_dir ./checkpoints \
        --batch_size 1 \
        --learning_rate 1e-4

Resume automatically:

    python train.py

Start fresh dataset accounting while preserving model/optimizer:

    python train.py --new-data

Notes:
- Parquet does not support simple raw-byte seeking in the same way as TXT/JSONL.
  For parquet, this script uses row-group / row resume metadata internally while
  keeping byte_offset = 0.
- JSON arrays also cannot be safely resumed from arbitrary byte positions without
  reparsing context, so JSON uses deterministic record-index resume.
- TXT and JSONL use true f.seek(byte_offset) + f.tell() resume.
- Ctrl-C requests a safe shutdown; checkpointing occurs after the current
  optimization step completes.
"""

import argparse
import csv
import json
import math
import os
import random
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from tqdm import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from transformers.modeling_outputs import CausalLMOutputWithPast


# ============================================================
# Global shutdown control
# ============================================================

STOP_REQUESTED = False


def sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(
        "\n[Ctrl-C] Stop requested. "
        "The current optimization step will finish, then a checkpoint will be saved."
    )


signal.signal(signal.SIGINT, sigint_handler)


# ============================================================
# Lion optimizer
# ============================================================

class Lion(Optimizer):
    """
    Lion: Evolved Sign Momentum optimizer.

    Reference update:

        update = beta1 * exp_avg + (1 - beta1) * grad
        parameter -= lr * sign(update)
        exp_avg = beta2 * exp_avg + (1 - beta2) * grad
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1")

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]

                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                update = exp_avg.mul(beta1).add(
                    grad,
                    alpha=1.0 - beta1,
                )

                p.add_(torch.sign(update), alpha=-lr)

                exp_avg.mul_(beta2).add_(
                    grad,
                    alpha=1.0 - beta2,
                )

        return loss


# ============================================================
# RWKV-X configuration
# ============================================================

class RWKVXConfig(PretrainedConfig):
    model_type = "rwkv_x"

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        intermediate_size: int = 4096,
        context_length: int = 8192,
        layer_norm_epsilon: float = 1e-5,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,

        # Channel-Mix MoE settings.
        # num_experts=1 keeps the original dense architecture.
        is_moe: bool = False,
        num_experts: int = 1,
        num_local_experts: Optional[int] = None,
        num_experts_per_tok: int = 1,
        moe_top_k: Optional[int] = None,
        moe_expert_density: float = 1.0,
        router_aux_loss_coef: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.context_length = context_length
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

        self.is_moe = bool(is_moe or int(num_experts) > 1)
        self.num_experts = max(1, int(num_experts))
        self.num_local_experts = max(
            1,
            int(num_local_experts)
            if num_local_experts is not None
            else self.num_experts,
        )

        effective_top_k = (
            moe_top_k
            if moe_top_k is not None
            else num_experts_per_tok
        )
        self.num_experts_per_tok = max(
            1,
            min(self.num_experts, int(effective_top_k)),
        )
        self.moe_top_k = self.num_experts_per_tok
        self.moe_expert_density = float(moe_expert_density)
        self.router_aux_loss_coef = float(router_aux_loss_coef)


# ============================================================
# RWKV-X time mixing
# ============================================================

class RWKVXTimeMix(nn.Module):
    """
    Simplified RWKV-style recurrent linear attention.

    The recurrent state contains:
        prev_x
        numerator
        denominator
        max_key

    This implementation is intentionally pure PyTorch and does not
    require custom CUDA kernels.
    """

    def __init__(self, config: RWKVXConfig, layer_id: int):
        super().__init__()

        n_embd = config.hidden_size

        self.layer_id = layer_id

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        ratio = layer_id / max(1, config.num_hidden_layers - 1)

        self.time_mix_k = nn.Parameter(
            torch.full((1, 1, n_embd), ratio)
        )
        self.time_mix_v = nn.Parameter(
            torch.full((1, 1, n_embd), ratio)
        )
        self.time_mix_r = nn.Parameter(
            torch.full((1, 1, n_embd), ratio)
        )

        self.time_decay = nn.Parameter(
            torch.linspace(
                -5.0,
                -1.0,
                n_embd,
            )
        )

        self.time_first = nn.Parameter(
            torch.zeros(n_embd)
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

        B, T, C = x.shape
        device = x.device
        dtype = x.dtype

        if state is None:
            prev_x = torch.zeros(
                B,
                C,
                device=device,
                dtype=dtype,
            )

            aa = torch.zeros(
                B,
                C,
                device=device,
                dtype=dtype,
            )

            bb = torch.zeros(
                B,
                C,
                device=device,
                dtype=dtype,
            )

            pp = torch.full(
                (B, C),
                -1e30,
                device=device,
                dtype=dtype,
            )
        else:
            prev_x, aa, bb, pp = state

        outputs = []

        time_decay = self.time_decay.to(dtype=dtype)
        time_first = self.time_first.to(dtype=dtype)

        for t in range(T):
            xt = x[:, t, :]

            xk = xt * self.time_mix_k[:, 0, :] + prev_x * (
                1.0 - self.time_mix_k[:, 0, :]
            )

            xv = xt * self.time_mix_v[:, 0, :] + prev_x * (
                1.0 - self.time_mix_v[:, 0, :]
            )

            xr = xt * self.time_mix_r[:, 0, :] + prev_x * (
                1.0 - self.time_mix_r[:, 0, :]
            )

            k = self.key(xk)
            v = self.value(xv)
            r = torch.sigmoid(self.receptance(xr))

            # Numerically stable exponential weighted recurrence.
            ww = time_first + k
            p = torch.maximum(pp, ww)

            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)

            numerator = e1 * aa + e2 * v
            denominator = e1 * bb + e2

            wkv = numerator / (denominator + 1e-9)

            out = self.output(r * wkv)
            outputs.append(out)

            # Advance recurrent state.
            decay_p = pp + time_decay
            new_p = torch.maximum(decay_p, k)

            e_decay = torch.exp(decay_p - new_p)
            e_key = torch.exp(k - new_p)

            aa = e_decay * aa + e_key * v
            bb = e_decay * bb + e_key
            pp = new_p

            prev_x = xt

        y = torch.stack(outputs, dim=1)

        return y, (
            prev_x.detach(),
            aa.detach(),
            bb.detach(),
            pp.detach(),
        )


# ============================================================
# RWKV-X channel mixing
# ============================================================

class RWKVXChannelMix(nn.Module):
    """
    One RWKV-X Channel-Mix expert.

    This class intentionally keeps the exact parameter names of the
    original dense model:

        key.weight
        value.weight
        receptance.weight
        time_mix_k
        time_mix_r

    RWKVXMoEChannelMix wraps multiple instances of this class and therefore
    matches merge.py's checkpoint layout:

        channel_mix.experts.0.key.weight
        channel_mix.experts.1.key.weight
        ...
        channel_mix.gate.weight
    """

    def __init__(self, config: RWKVXConfig, layer_id: int):
        super().__init__()

        n_embd = config.hidden_size
        hidden = config.intermediate_size

        self.layer_id = layer_id

        self.key = nn.Linear(
            n_embd,
            hidden,
            bias=False,
        )

        self.value = nn.Linear(
            hidden,
            n_embd,
            bias=False,
        )

        self.receptance = nn.Linear(
            n_embd,
            n_embd,
            bias=False,
        )

        ratio = layer_id / max(
            1,
            config.num_hidden_layers - 1,
        )

        self.time_mix_k = nn.Parameter(
            torch.full((1, 1, n_embd), ratio)
        )

        self.time_mix_r = nn.Parameter(
            torch.full((1, 1, n_embd), ratio)
        )

    def forward(
        self,
        x: torch.Tensor,
        prev_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, T, C = x.shape

        if prev_x is None:
            prev_x = torch.zeros(
                B,
                C,
                device=x.device,
                dtype=x.dtype,
            )

        outputs = []

        mix_k = self.time_mix_k[:, 0, :]
        mix_r = self.time_mix_r[:, 0, :]

        for t in range(T):
            xt = x[:, t, :]

            xk = xt * mix_k + prev_x * (
                1.0 - mix_k
            )

            xr = xt * mix_r + prev_x * (
                1.0 - mix_r
            )

            k = torch.relu(self.key(xk))
            k = k * k

            v = self.value(k)

            r = torch.sigmoid(
                self.receptance(xr)
            )

            outputs.append(r * v)

            prev_x = xt

        y = torch.stack(
            outputs,
            dim=1,
        )

        return y, prev_x.detach()


class RWKVXMoEChannelMix(nn.Module):
    """
    Top-k routed RWKV-X Channel-Mix MoE.

    The router is token-level and stateless. Channel recurrence remains
    inside every expert, so each expert maintains its own previous-input
    state during cached generation.

    State layout:

        (
            expert_0_prev_x,
            expert_1_prev_x,
            ...
        )

    For top-k routing, only selected experts contribute to a token's output.
    The implementation evaluates selected experts per sequence rather than
    allocating [B, T, E, C] activations, which is much safer on low-RAM CPUs.
    """

    def __init__(
        self,
        config: RWKVXConfig,
        layer_id: int,
    ):
        super().__init__()

        self.layer_id = layer_id
        self.num_experts = max(1, int(config.num_experts))
        self.top_k = max(
            1,
            min(
                self.num_experts,
                int(config.num_experts_per_tok),
            ),
        )

        self.experts = nn.ModuleList(
            [
                RWKVXChannelMix(
                    config,
                    layer_id,
                )
                for _ in range(
                    self.num_experts
                )
            ]
        )

        # Shape exactly matches merge.py:
        # [num_experts, hidden_size]
        self.gate = nn.Linear(
            config.hidden_size,
            self.num_experts,
            bias=False,
        )

        nn.init.normal_(
            self.gate.weight,
            mean=0.0,
            std=0.02,
        )

    def _normalize_state(
        self,
        state,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if state is None:
            return [
                torch.zeros(
                    batch_size,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_experts)
            ]

        if not isinstance(state, (tuple, list)):
            raise TypeError(
                "MoE Channel-Mix state must be a tuple or list "
                "containing one previous-input tensor per expert."
            )

        if len(state) != self.num_experts:
            raise ValueError(
                f"MoE Channel-Mix state has {len(state)} experts, "
                f"but this layer has {self.num_experts}."
            )

        normalized = []

        for item in state:
            if item is None:
                normalized.append(
                    torch.zeros(
                        batch_size,
                        hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                )
            else:
                normalized.append(
                    item.to(
                        device=device,
                        dtype=dtype,
                    )
                )

        return normalized

    def forward(
        self,
        x: torch.Tensor,
        state=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:

        B, T, C = x.shape

        expert_states = self._normalize_state(
            state=state,
            batch_size=B,
            hidden_size=C,
            device=x.device,
            dtype=x.dtype,
        )

        # Router logits: [B, T, E]
        router_logits = self.gate(x)

        top_values, top_indices = torch.topk(
            router_logits,
            k=self.top_k,
            dim=-1,
        )

        top_weights = torch.softmax(
            top_values,
            dim=-1,
        ).to(dtype=x.dtype)

        output = torch.zeros_like(x)
        new_states = []

        # Process each expert over the full sequence so its recurrence is
        # mathematically consistent. Only routed tokens are accumulated.
        for expert_id, expert in enumerate(self.experts):

            expert_output, expert_state = expert(
                x,
                expert_states[expert_id],
            )

            new_states.append(
                expert_state
            )

            matches = (
                top_indices == expert_id
            )

            if not torch.any(matches):
                continue

            expert_weight = torch.where(
                matches,
                top_weights,
                torch.zeros_like(top_weights),
            ).sum(
                dim=-1,
            )

            output = output + (
                expert_output
                * expert_weight.unsqueeze(-1)
            )

        return output, tuple(new_states)


# ============================================================
# RWKV-X block
# ============================================================

class RWKVXBlock(nn.Module):
    def __init__(
        self,
        config: RWKVXConfig,
        layer_id: int,
    ):
        super().__init__()

        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.ln2 = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.time_mix = RWKVXTimeMix(
            config,
            layer_id,
        )

        if config.is_moe or config.num_experts > 1:
            self.channel_mix = RWKVXMoEChannelMix(
                config,
                layer_id,
            )
        else:
            self.channel_mix = RWKVXChannelMix(
                config,
                layer_id,
            )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[Any, ...]] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[Any, ...],
    ]:

        if state is None:
            time_state = None
            channel_state = None
        else:
            time_state, channel_state = state

        y, new_time_state = self.time_mix(
            self.ln1(x),
            time_state,
        )

        x = x + self.dropout(y)

        y, new_channel_state = self.channel_mix(
            self.ln2(x),
            channel_state,
        )

        x = x + self.dropout(y)

        return x, (
            new_time_state,
            new_channel_state,
        )


# ============================================================
# RWKV-X Hugging Face model
# ============================================================

class RWKVXForCausalLM(PreTrainedModel):
    config_class = RWKVXConfig
    base_model_prefix = "rwkv_x"
    supports_gradient_checkpointing = False

    def __init__(self, config: RWKVXConfig):
        super().__init__(config)

        self.emb = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                RWKVXBlock(
                    config,
                    layer_id=i,
                )
                for i in range(
                    config.num_hidden_layers
                )
            ]
        )

        self.ln_out = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        self.head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.head.weight = self.emb.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values: Optional[
            Tuple[Any, ...]
        ] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if input_ids is None:
            raise ValueError(
                "input_ids must be provided"
            )

        x = self.emb(input_ids)

        new_states = []

        if past_key_values is None:
            past_key_values = [
                None
                for _ in range(
                    len(self.blocks)
                )
            ]

        for i, block in enumerate(
            self.blocks
        ):
            x, state = block(
                x,
                past_key_values[i],
            )

            if use_cache:
                new_states.append(state)

        x = self.ln_out(x)

        logits = self.head(x)

        loss = None

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()

            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(
                    -1,
                    shift_logits.size(-1),
                ),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=(
                tuple(new_states)
                if use_cache
                else None
            ),
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }


# Register with Hugging Face auto classes in this process.
AutoConfig.register(
    RWKVXConfig.model_type,
    RWKVXConfig,
)

AutoModelForCausalLM.register(
    RWKVXConfig,
    RWKVXForCausalLM,
)


# ============================================================
# Resume state
# ============================================================

@dataclass
class ResumeState:
    current_file_path: str = ""
    byte_offset: int = 0
    global_step: int = 0
    total_tokens_processed: int = 0

    # Extra metadata for formats where raw byte seeking is
    # not safe or meaningful.
    record_index: int = 0
    parquet_row_group: int = 0
    parquet_row_offset: int = 0

    @classmethod
    def load(
        cls,
        path: Path,
    ) -> "ResumeState":
        if not path.exists():
            return cls()

        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)

            return cls(
                current_file_path=data.get(
                    "current_file_path",
                    "",
                ),
                byte_offset=int(
                    data.get(
                        "byte_offset",
                        0,
                    )
                ),
                global_step=int(
                    data.get(
                        "global_step",
                        0,
                    )
                ),
                total_tokens_processed=int(
                    data.get(
                        "total_tokens_processed",
                        0,
                    )
                ),
                record_index=int(
                    data.get(
                        "record_index",
                        0,
                    )
                ),
                parquet_row_group=int(
                    data.get(
                        "parquet_row_group",
                        0,
                    )
                ),
                parquet_row_offset=int(
                    data.get(
                        "parquet_row_offset",
                        0,
                    )
                ),
            )

        except Exception as e:
            print(
                f"[WARN] Could not load resume state: {e}"
            )
            return cls()

    def save(self, path: Path):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = path.with_suffix(
            ".json.tmp"
        )

        data = {
            "current_file_path": self.current_file_path,
            "byte_offset": self.byte_offset,
            "global_step": self.global_step,
            "total_tokens_processed": self.total_tokens_processed,
            "record_index": self.record_index,
            "parquet_row_group": self.parquet_row_group,
            "parquet_row_offset": self.parquet_row_offset,
        }

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=2,
            )

            f.flush()
            os.fsync(f.fileno())

        os.replace(
            temp_path,
            path,
        )


# ============================================================
# Dataset discovery
# ============================================================

SUPPORTED_SUFFIXES = {
    ".txt",
    ".text",
    ".jsonl",
    ".json",
    ".csv",
    ".parquet",

    # Common source code formats.
    ".py",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cc",
    ".cxx",
    ".rs",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".cs",
    ".php",
    ".rb",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".html",
    ".css",
    ".scss",
    ".sql",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
}


def discover_files(
    dataset_dir: Path,
) -> List[Path]:

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {dataset_dir}"
        )

    files = []

    for path in dataset_dir.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_SUFFIXES
        ):
            files.append(
                path.resolve()
            )

    files.sort()

    return files


# ============================================================
# Text extraction
# ============================================================

TEXT_KEYS = (
    "text",
    "content",
    "document",
    "body",
    "code",
    "prompt",
    "completion",
)


def extract_text_from_object(
    obj: Any,
) -> str:

    if isinstance(obj, str):
        return obj

    if isinstance(obj, dict):
        for key in TEXT_KEYS:
            value = obj.get(key)

            if isinstance(value, str):
                return value

        strings = []

        for value in obj.values():
            if isinstance(
                value,
                str,
            ):
                strings.append(value)

        return "\n".join(strings)

    if isinstance(obj, list):
        return "\n".join(
            extract_text_from_object(x)
            for x in obj
        )

    return str(obj)


# ============================================================
# Streaming file iterator
# ============================================================

class DatasetStream:
    """
    Sequentially streams files.

    TXT / JSONL:
        true byte-offset seeking

    JSON / CSV:
        deterministic record-index resume

    PARQUET:
        row-group + row offset resume
    """

    def __init__(
        self,
        files: List[Path],
        resume_state: ResumeState,
    ):
        self.files = files
        self.resume_state = resume_state

    def __iter__(
        self,
    ) -> Iterator[Tuple[str, Path, int, Dict[str, int]]]:

        resume_file = (
            Path(
                self.resume_state.current_file_path
            ).resolve()
            if self.resume_state.current_file_path
            else None
        )

        started = resume_file is None

        for file_path in self.files:
            if (
                not started
                and file_path == resume_file
            ):
                started = True

            if not started:
                continue

            suffix = file_path.suffix.lower()

            try:
                if suffix in {
                    ".txt",
                    ".text",
                    ".py",
                    ".cpp",
                    ".c",
                    ".h",
                    ".hpp",
                    ".cc",
                    ".cxx",
                    ".rs",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".jsx",
                    ".java",
                    ".go",
                    ".cs",
                    ".php",
                    ".rb",
                    ".swift",
                    ".kt",
                    ".kts",
                    ".scala",
                    ".sh",
                    ".bash",
                    ".zsh",
                    ".html",
                    ".css",
                    ".scss",
                    ".sql",
                    ".md",
                    ".rst",
                    ".yaml",
                    ".yml",
                    ".toml",
                    ".xml",
                }:
                    yield from self._stream_text_file(
                        file_path
                    )

                elif suffix == ".jsonl":
                    yield from self._stream_jsonl(
                        file_path
                    )

                elif suffix == ".json":
                    yield from self._stream_json(
                        file_path
                    )

                elif suffix == ".csv":
                    yield from self._stream_csv(
                        file_path
                    )

                elif suffix == ".parquet":
                    yield from self._stream_parquet(
                        file_path
                    )

            except Exception as e:
                print(
                    f"[WARN] Skipping {file_path}: {e}"
                )

            # Resume offset applies only to the original file.
            self.resume_state.byte_offset = 0
            self.resume_state.record_index = 0
            self.resume_state.parquet_row_group = 0
            self.resume_state.parquet_row_offset = 0

    def _is_resume_file(
        self,
        path: Path,
    ) -> bool:
        return (
            self.resume_state.current_file_path
            and Path(
                self.resume_state.current_file_path
            ).resolve()
            == path.resolve()
        )

    def _stream_text_file(
        self,
        path: Path,
    ):

        start_offset = 0

        if self._is_resume_file(path):
            start_offset = max(
                0,
                self.resume_state.byte_offset,
            )

        with open(
            path,
            "rb",
        ) as f:

            f.seek(start_offset)

            if start_offset > 0:
                # If offset lands inside a line, discard the
                # partial remainder. Normally checkpoints occur
                # exactly at line boundaries.
                f.readline()

            while True:
                start = f.tell()

                raw = f.readline()

                if not raw:
                    break

                end = f.tell()

                text = raw.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                if text:
                    yield (
                        text,
                        path,
                        end,
                        {
                            "record_index": 0,
                            "parquet_row_group": 0,
                            "parquet_row_offset": 0,
                        },
                    )

    def _stream_jsonl(
        self,
        path: Path,
    ):

        start_offset = 0

        if self._is_resume_file(path):
            start_offset = max(
                0,
                self.resume_state.byte_offset,
            )

        with open(
            path,
            "rb",
        ) as f:

            f.seek(start_offset)

            if start_offset > 0:
                f.readline()

            while True:
                raw = f.readline()

                if not raw:
                    break

                end = f.tell()

                try:
                    obj = json.loads(
                        raw.decode(
                            "utf-8",
                            errors="replace",
                        )
                    )

                    text = extract_text_from_object(
                        obj
                    ).strip()

                    if text:
                        yield (
                            text,
                            path,
                            end,
                            {
                                "record_index": 0,
                                "parquet_row_group": 0,
                                "parquet_row_offset": 0,
                            },
                        )

                except json.JSONDecodeError:
                    continue

    def _stream_json(
        self,
        path: Path,
    ):

        start_index = 0

        if self._is_resume_file(path):
            start_index = max(
                0,
                self.resume_state.record_index,
            )

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            data = json.load(f)

        if isinstance(data, dict):
            if isinstance(
                data.get("data"),
                list,
            ):
                records = data["data"]
            else:
                records = [data]
        else:
            records = data

        for i in range(
            start_index,
            len(records),
        ):
            text = extract_text_from_object(
                records[i]
            ).strip()

            if text:
                yield (
                    text,
                    path,
                    0,
                    {
                        "record_index": i + 1,
                        "parquet_row_group": 0,
                        "parquet_row_offset": 0,
                    },
                )

    def _stream_csv(
        self,
        path: Path,
    ):

        start_index = 0

        if self._is_resume_file(path):
            start_index = max(
                0,
                self.resume_state.record_index,
            )

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as f:

            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if i < start_index:
                    continue

                text = extract_text_from_object(
                    row
                ).strip()

                if text:
                    yield (
                        text,
                        path,
                        0,
                        {
                            "record_index": i + 1,
                            "parquet_row_group": 0,
                            "parquet_row_offset": 0,
                        },
                    )

    def _stream_parquet(
        self,
        path: Path,
    ):

        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise RuntimeError(
                "pyarrow is required for parquet support. "
                "Install it with: pip install pyarrow"
            )

        row_group_start = 0
        row_offset_start = 0

        if self._is_resume_file(path):
            row_group_start = max(
                0,
                self.resume_state.parquet_row_group,
            )

            row_offset_start = max(
                0,
                self.resume_state.parquet_row_offset,
            )

        parquet = pq.ParquetFile(path)

        for rg in range(
            row_group_start,
            parquet.num_row_groups,
        ):
            table = parquet.read_row_group(rg)

            rows = table.to_pylist()

            local_start = (
                row_offset_start
                if rg == row_group_start
                else 0
            )

            for i in range(
                local_start,
                len(rows),
            ):
                text = extract_text_from_object(
                    rows[i]
                ).strip()

                if text:
                    yield (
                        text,
                        path,
                        0,
                        {
                            "record_index": 0,
                            "parquet_row_group": rg,
                            "parquet_row_offset": i + 1,
                        },
                    )

            row_offset_start = 0


# ============================================================
# Tokenizer helpers
# ============================================================

def find_tokenizer(
    output_dir: Path,
    checkpoint_dir: Path,
) -> Optional[Path]:

    candidates = [
        output_dir,
        checkpoint_dir,
        Path("./tokenizer"),
        Path("./"),
    ]

    for candidate in candidates:
        if (
            candidate / "tokenizer.json"
        ).exists():
            return candidate

    return None


def load_tokenizer(
    output_dir: Path,
    checkpoint_dir: Path,
):

    tokenizer_path = find_tokenizer(
        output_dir,
        checkpoint_dir,
    )

    if tokenizer_path is None:
        raise FileNotFoundError(
            "\nNo tokenizer.json found.\n\n"
            "Place your trained tokenizer in one of:\n"
            f"  {output_dir}/tokenizer.json\n"
            f"  {checkpoint_dir}/tokenizer.json\n"
            "  ./tokenizer/tokenizer.json\n"
            "  ./tokenizer.json\n\n"
            "This trainer intentionally does not train a tokenizer."
        )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_path
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens(
                {"pad_token": "<|pad|>"}
            )

    return tokenizer


# ============================================================
# Model creation/loading
# ============================================================

def count_parameters(
    model: nn.Module,
) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
    )


def create_or_load_model(
    output_dir: Path,
    tokenizer,
    device: torch.device,
):

    model_file = (
        output_dir
        / "model.safetensors"
    )

    model_index_file = (
        output_dir
        / "model.safetensors.index.json"
    )

    config_file = (
        output_dir
        / "config.json"
    )

    if (
        (model_file.exists() or model_index_file.exists())
        and config_file.exists()
    ):
        print(
            f"[RESUME] Loading model from {output_dir}"
        )

        config = RWKVXConfig.from_pretrained(
            output_dir
        )

        model = RWKVXForCausalLM.from_pretrained(
            output_dir,
            config=config,
        )

    else:
        print(
            "[INIT] Creating new RWKV-X model"
        )

        config = RWKVXConfig(
            vocab_size=len(tokenizer),
            hidden_size=1024,
            num_hidden_layers=24,
            intermediate_size=4096,
            context_length=8_000_000, # Fake Context Limitation!
            layer_norm_epsilon=1e-5,
            dropout=0.0,
            tie_word_embeddings=True,
            architectures=[
                "RWKVXForCausalLM"
            ],
        )

        model = RWKVXForCausalLM(
            config
        )

    if len(tokenizer) != model.config.vocab_size:
        print(
            "[WARN] Tokenizer vocabulary size differs "
            "from model vocabulary size. Resizing embeddings."
        )

        model.resize_token_embeddings(
            len(tokenizer)
        )

    model.to(device)

    return model


# ============================================================
# Checkpoint saving
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    resume_state,
    output_dir: Path,
    checkpoint_dir: Path,
):

    print("\n[SAVE] Saving checkpoint...")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # HF model export.
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="50GB",
    )

    tokenizer.save_pretrained(
        output_dir
    )

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=256,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    generation_config.save_pretrained(
        output_dir
    )

    optimizer_path = (
        checkpoint_dir
        / "optimizer.pt"
    )

    temp_optimizer = (
        checkpoint_dir
        / "optimizer.pt.tmp"
    )

    torch.save(
        optimizer.state_dict(),
        temp_optimizer,
    )

    os.replace(
        temp_optimizer,
        optimizer_path,
    )

    resume_state.save(
        checkpoint_dir
        / "resume_state.json"
    )

    print(
        "\n[SAVE COMPLETE]"
    )

    print(
        f"Model: {output_dir}"
    )

    print(
        f"Optimizer: {optimizer_path}"
    )

    print(
        f"Resume state: "
        f"{checkpoint_dir / 'resume_state.json'}"
    )


# ============================================================
# Batch stream
# ============================================================

def token_chunk_stream(
    dataset_stream,
    tokenizer,
    seq_len: int,
    resume_state: ResumeState,
):

    token_buffer: List[int] = []

    current_metadata = None

    eos_id = tokenizer.eos_token_id

    for (
        text,
        file_path,
        byte_offset,
        metadata,
    ) in dataset_stream:

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )

        ids = encoded["input_ids"]

        if eos_id is not None:
            ids.append(eos_id)

        token_buffer.extend(ids)

        current_metadata = (
            file_path,
            byte_offset,
            metadata,
        )

        while len(token_buffer) >= seq_len + 1:
            chunk = token_buffer[
                : seq_len + 1
            ]

            del token_buffer[
                : seq_len + 1
            ]

            yield (
                torch.tensor(
                    chunk[:-1],
                    dtype=torch.long,
                ),
                torch.tensor(
                    chunk[1:],
                    dtype=torch.long,
                ),
                current_metadata,
            )


# ============================================================
# Training
# ============================================================

def train(
    args,
    model,
    tokenizer,
    optimizer,
    resume_state: ResumeState,
    device: torch.device,
):

    dataset_dir = Path(
        args.dataset_dir
    ).resolve()

    files = discover_files(
        dataset_dir
    )

    if not files:
        raise RuntimeError(
            f"No supported files found in {dataset_dir}"
        )

    print(
        f"[DATA] Found {len(files)} files"
    )

    if args.new_data:
        print(
            "[NEW DATA] Resetting dataset position, "
            "global_step, and token counter."
        )

        resume_state.current_file_path = ""
        resume_state.byte_offset = 0
        resume_state.record_index = 0
        resume_state.parquet_row_group = 0
        resume_state.parquet_row_offset = 0
        resume_state.global_step = 0
        resume_state.total_tokens_processed = 0

    model.train()

    seq_len = min(
        2048,
        model.config.context_length,
    )

    print(
        f"[CONFIG] Sequence length per training sample: {seq_len}"
    )

    dataset_stream = DatasetStream(
        files,
        resume_state,
    )

    stream = token_chunk_stream(
        dataset_stream,
        tokenizer,
        seq_len,
        resume_state,
    )

    batch_size = args.batch_size

    progress = tqdm(
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )

    start_time = time.perf_counter()
    tokens_since_update = 0

    batch_inputs = []
    batch_labels = []
    batch_metadata = []

    try:
        for (
            input_ids,
            labels,
            metadata,
        ) in stream:

            batch_inputs.append(
                input_ids
            )

            batch_labels.append(
                labels
            )

            batch_metadata.append(
                metadata
            )

            if (
                len(batch_inputs)
                < batch_size
            ):
                continue

            input_batch = torch.stack(
                batch_inputs,
                dim=0,
            ).to(device)

            label_batch = torch.stack(
                batch_labels,
                dim=0,
            ).to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                input_ids=input_batch,
                labels=label_batch,
                use_cache=False,
            )

            loss = output.loss

            if not torch.isfinite(loss):
                print(
                    f"\n[WARN] Non-finite loss: "
                    f"{loss.item()}. Skipping step."
                )

                batch_inputs.clear()
                batch_labels.clear()
                batch_metadata.clear()

                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            # The last fully consumed item in this batch
            # defines the exact resume location.
            file_path, byte_offset, meta = (
                batch_metadata[-1]
            )

            resume_state.current_file_path = str(
                file_path.resolve()
            )

            resume_state.byte_offset = int(
                byte_offset
            )

            resume_state.record_index = int(
                meta.get(
                    "record_index",
                    0,
                )
            )

            resume_state.parquet_row_group = int(
                meta.get(
                    "parquet_row_group",
                    0,
                )
            )

            resume_state.parquet_row_offset = int(
                meta.get(
                    "parquet_row_offset",
                    0,
                )
            )

            resume_state.global_step += 1

            processed = (
                input_batch.numel()
            )

            resume_state.total_tokens_processed += (
                processed
            )

            tokens_since_update += processed

            elapsed = (
                time.perf_counter()
                - start_time
            )

            tok_per_sec = (
                tokens_since_update
                / max(elapsed, 1e-9)
            )

            progress.set_postfix(
                step=resume_state.global_step,
                loss=f"{loss.item():.4f}",
                tok_s=f"{tok_per_sec:.1f}",
                byte=resume_state.byte_offset,
                file=file_path.name[:20],
            )

            progress.update(1)

            batch_inputs.clear()
            batch_labels.clear()
            batch_metadata.clear()

            # Reset local throughput timer.
            start_time = time.perf_counter()
            tokens_since_update = 0

            if STOP_REQUESTED:
                break

    except KeyboardInterrupt:
        # Covers environments where SIGINT interrupts
        # Python directly instead of merely setting the flag.
        print(
            "\n[INTERRUPT] KeyboardInterrupt received."
        )

    finally:
        progress.close()

        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            resume_state=resume_state,
            output_dir=Path(
                args.output_dir
            ),
            checkpoint_dir=Path(
                args.checkpoint_dir
            ),
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train a RWKV-X style causal language model."
        )
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./datasets",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./SmaulNative",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--new-data",
        action="store_true",
        dest="new_data",
        help=(
            "Reset dataset position/global step/token count "
            "while preserving model and optimizer weights."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "--batch_size must be >= 1"
        )

    torch.manual_seed(42)
    random.seed(42)

    output_dir = Path(
        args.output_dir
    )

    checkpoint_dir = Path(
        args.checkpoint_dir
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    print(
        f"[DEVICE] {device}"
    )

    tokenizer = load_tokenizer(
        output_dir,
        checkpoint_dir,
    )

    print(
        f"[TOKENIZER] Vocabulary size: "
        f"{len(tokenizer):,}"
    )

    model = create_or_load_model(
        output_dir,
        tokenizer,
        device,
    )

    parameter_count = count_parameters(
        model
    )

    print(
        f"[MODEL] Parameters: "
        f"{parameter_count:,} "
        f"({parameter_count / 1e6:.2f}M)"
    )

    optimizer = Lion(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    optimizer_path = (
        checkpoint_dir
        / "optimizer.pt"
    )

    if (
        optimizer_path.exists()
        and not args.new_data
    ):
        try:
            print(
                "[RESUME] Loading optimizer state"
            )

            optimizer.load_state_dict(
                torch.load(
                    optimizer_path,
                    map_location=device,
                    weights_only=False,
                )
            )

        except Exception as e:
            print(
                f"[WARN] Could not restore optimizer: {e}"
            )

    resume_state = ResumeState.load(
        checkpoint_dir
        / "resume_state.json"
    )

    if args.new_data:
        resume_state = ResumeState()

    print(
        f"[RESUME] Step: "
        f"{resume_state.global_step:,}"
    )

    print(
        f"[RESUME] Tokens: "
        f"{resume_state.total_tokens_processed:,}"
    )

    if (
        resume_state.current_file_path
    ):
        print(
            f"[RESUME] File: "
            f"{resume_state.current_file_path}"
        )

        print(
            f"[RESUME] Byte offset: "
            f"{resume_state.byte_offset:,}"
        )

    train(
        args=args,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        resume_state=resume_state,
        device=device,
    )


if __name__ == "__main__":
    main()

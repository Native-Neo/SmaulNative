#!/usr/bin/env python3

import argparse
import contextlib
import json
import os
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

if "--cpu" in sys.argv or "--rqt" in sys.argv:
    threads = str(os.environ.get("SMAUL_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, threads)
    if "MKL_ENABLE_INSTRUCTIONS" not in os.environ:
        try:
            flags = Path("/proc/cpuinfo").read_text(errors="ignore")
            if " avx" in flags or "\navx " in flags:
                os.environ["MKL_ENABLE_INSTRUCTIONS"] = "AVX"
        except OSError:
            pass
    os.environ.setdefault("TORCHINDUCTOR_CPP_WRAPPER", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    os.environ.setdefault("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,CPP")

import torch
from torch.optim import Optimizer

from backend import backend_device, require_backend
from dataset import PretrainStream, SFTDataset, discover_files, iter_texts, load_tokenizer, tokenizer_vocab_size
from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE
from stream_data import stream_dataset
from tokenizer import ensure_tokenizer
import qat

STOP_REQUESTED = False


def _sigint_handler(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[Ctrl-C] Stop requested. Finishing current step, then saving checkpoint.")


signal.signal(signal.SIGINT, _sigint_handler)


def _format_size(num_bytes):
    units = ("B", "KiB", "MiB", "GiB")
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def _print_model_size(model):
    parameters = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    fp32_bytes = parameters * 4
    print(f"[MODEL] {parameters:,} parameters | trainable={trainable:,} | FP32={_format_size(fp32_bytes)}")
    return parameters


def _print_rqt_storage(model):
    packed_bytes = 0
    scale_bytes = 0
    for module in model.modules():
        quant = getattr(module, "quant", None)
        if quant is None or not hasattr(quant, "packed"):
            continue
        packed_bytes += quant.packed.numel() * quant.packed.element_size()
        if hasattr(quant, "scale"):
            scale_bytes += quant.scale.numel() * scale.element_size()
    master_bytes = sum(param.numel() for param in model.parameters()) * 4
    print(f"[RQT] packed storage={_format_size(packed_bytes + scale_bytes)} | master FP32={_format_size(master_bytes)}")


class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state["exp_avg"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                if weight_decay:
                    param.mul_(1 - lr * weight_decay)
                param.add_(exp_avg.mul(b1).add(param.grad, alpha=1 - b1).sign(), alpha=-lr)
                exp_avg.lerp_(param.grad, 1 - b2)
        return loss


def set_router_only_training(model, router_only):
    if not model.cfg.is_moe:
        raise ValueError("set_router_only_training requires cfg.is_moe=True")
    gates = {
        id(param)
        for module in model.modules()
        if isinstance(module, RWKV_CMix_MoE)
        for param in module.gate.parameters()
    }
    trainable_params = 0
    for param in model.parameters():
        param.requires_grad_(id(param) in gates if router_only else True)
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params


class ResumeState:
    def __init__(self):
        self.global_step = 0
        self.total_tokens = 0
        self.file_path: Optional[str] = None
        self.record_index = 0
        self.epoch = 0
        self.buffer_tokens = []

    @classmethod
    def load(cls, path):
        state = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if not isinstance(data, dict):
                    raise ValueError("resume state must be a JSON object")
                state.global_step = data.get("global_step", 0)
                state.total_tokens = data.get("total_tokens", 0)
                state.file_path = data.get("file_path")
                state.record_index = data.get("record_index", 0)
                state.epoch = data.get("epoch", 0)
                state.buffer_tokens = data.get("buffer_tokens", [])
            except Exception as exc:
                raise RuntimeError(f"could not load resume state {path}: {exc}") from exc
        return state

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.__dict__, indent=2))
        os.replace(tmp, path)


def _save_rng_state(path):
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_rng_state(path):
    if not path.exists():
        return
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if "torch" not in state:
            raise ValueError("missing torch RNG state")
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:
        raise RuntimeError(f"could not restore RNG state {path}: {exc}") from exc

#!/usr/bin/env python3
# Fake low-bit floating-point QAT for RWKV-X.

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE, RWKV_CMix_x070

_CMIX_LINEAR_NAMES = ("key", "value")
_SUPPORTED_BITS = (2, 4, 8)
_LOWBIT_EXT = None


def _load_lowbit():
    global _LOWBIT_EXT
    if _LOWBIT_EXT is None:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent / "cpu"
        _LOWBIT_EXT = load(
            name="smaulnative_lowbit",
            sources=[str(root / "lowbit_kernel.cpp")],
            extra_cflags=["-O3", "-march=native", "-mtune=native"],
            verbose=False,
        )
    return _LOWBIT_EXT


def _bits(bits=None):
    bits = bits or int(os.environ.get("SMAUL_QAT_BITS", "8"))
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be 2, 4, or 8")
    return bits


def _float_levels(bits, device):
    if bits == 2:
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    if bits == 4:
        return torch.tensor([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0], device=device)
    raise ValueError("FP8 uses E4M3")


def _quantize_codes(x, bits):
    levels = _float_levels(bits, x.device).to(x.dtype)
    scale = x.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    codes = (x.detach() / scale).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
    return codes, scale


def fake_quantize(x, bits):
    bits = _bits(bits)
    if bits == 8:
        q = x.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        codes, scale = _quantize_codes(x, bits)
        q = _float_levels(bits, x.device).to(x.dtype)[codes] * scale
    return x + (q - x).detach()


class _QATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bits):
        ctx.save_for_backward(x, weight)
        ctx.bits = bits
        if x.device.type == "cpu" and x.dtype == torch.float32 and bits < 8:
            out_features, in_features = weight.shape
            x2 = x.reshape(-1, in_features).contiguous()
            y = _load_lowbit().qat_linear(x2, weight, bits)
            return y.reshape(*x.shape[:-1], out_features)
        return F.linear(x, fake_quantize(weight, bits))

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        q = fake_quantize(weight, ctx.bits)
        x2 = x.reshape(-1, x.shape[-1])
        go = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = go.matmul(q).reshape_as(x)
        grad_weight = go.transpose(0, 1).matmul(x2)
        return grad_x, grad_weight, None


class FloatQATLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("RWKV-X Channel-Mix linears must have bias=False")
        self.weight = linear.weight
        self.bits = _bits(bits)

    def forward(self, x):
        return _QATFunction.apply(x, self.weight, self.bits)


def _iter_cmix_modules(model: RWKVXModel):
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


def prepare_qat(model: RWKVXModel, bits=None) -> int:
    bits = _bits(bits)
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, nn.Linear):
                setattr(cmix, name, FloatQATLinear(mod, bits))
                n += 1
    print(f"[QAT] FP{bits} fake quantization")
    return n


@torch.no_grad()
def calibrate(model, tokenizer, calib_texts, ctx_len, device, max_batches=64):
    return 0


# Compatibility for loading real QT checkpoints.
from qt import QuantizedLinear

#!/usr/bin/env python3
# Low-bit floating-point QAT for RWKV-X.

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE, RWKV_CMix_x070

_CMIX_LINEAR_NAMES = ("key", "value")
_SUPPORTED_BITS = (2, 4, 8)

if "--qt" in sys.argv:
    i = sys.argv.index("--qt")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1] not in {"2", "4", "8"}:
        raise SystemExit("--qt requires 2, 4, or 8")
    os.environ["SMAUL_QT_BITS"] = sys.argv[i + 1]
    sys.argv[i:i + 2] = ["--qat"]


def _bits(bits=None):
    bits = bits or int(os.environ.get("SMAUL_QT_BITS", "8"))
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be 2, 4, or 8")
    return bits


def _float_levels(bits, device):
    if bits == 2:
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    return torch.tensor([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0], device=device)


def _float_quantize(x, bits):
    if bits == 8:
        q = x.to(torch.float8_e4m3fn).to(x.dtype)
        return x + (q - x).detach()
    levels = _float_levels(bits, x.device).to(x.dtype)
    scale = x.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    y = x / scale
    idx = (y.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    q = levels[idx] * scale
    return x + (q - x).detach()


class FloatQATLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("RWKV-X Channel-Mix linears must have bias=False")
        self.weight = linear.weight
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, _float_quantize(self.weight, self.bits))


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
    print(f"[QAT] floating-point {bits}-bit weights")
    return n


def convert_qat(model: RWKVXModel, bits=None) -> int:
    _bits(bits)
    return sum(isinstance(getattr(cmix, name), FloatQATLinear) for cmix in _iter_cmix_modules(model) for name in _CMIX_LINEAR_NAMES)


@torch.no_grad()
def calibrate(model, tokenizer, calib_texts, ctx_len, device, max_batches=64):
    return 0

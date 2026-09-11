#!/usr/bin/env python3
# Low-bit floating-point QAT for RWKV-X.

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE, RWKV_CMix_x070

_CMIX_LINEAR_NAMES = ("key", "value")
_SUPPORTED_BITS = (2, 4, 8)


def _float_levels(bits, device):
    if bits == 2:
        return torch.tensor([-1.0, 0.0, 1.0], device=device)
    if bits == 4:
        return torch.tensor([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0], device=device)
    raise ValueError("bits must be 2, 4, or 8")


def _float_quantize(x, bits):
    if bits == 8:
        return x.to(torch.float8_e4m3fn).to(x.dtype)
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
        if bits not in _SUPPORTED_BITS:
            raise ValueError("bits must be 2, 4, or 8")
        self.weight = linear.weight
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _float_quantize(self.weight, self.bits)
        return F.linear(x, w)


def _iter_cmix_modules(model: RWKVXModel):
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


def prepare_qat(model: RWKVXModel, bits: int) -> int:
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be 2, 4, or 8")
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, nn.Linear):
                setattr(cmix, name, FloatQATLinear(mod, bits))
                n += 1
    return n


def convert_qat(model: RWKVXModel, bits: int) -> int:
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, FloatQATLinear):
                setattr(cmix, name, mod)
                n += 1
    return n


@torch.no_grad()
def calibrate(model, tokenizer, calib_texts, ctx_len, device, max_batches=64):
    return 0

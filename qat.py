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
    if bits == 4:
        return torch.tensor([-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0], device=device)
    raise ValueError("FP8 uses E4M3")


def _fake_quantize(x, bits):
    if bits == 8:
        q = x.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        levels = _float_levels(bits, x.device).to(x.dtype)
        scale = x.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
        idx = (x / scale).unsqueeze(-1).sub(levels).abs().argmin(dim=-1)
        q = levels[idx] * scale
    return x + (q - x).detach()


def _pack_codes(codes, bits):
    if bits == 8:
        return codes.to(torch.float8_e4m3fn), codes.numel()
    per_byte = 8 // bits
    flat = codes.reshape(-1).to(torch.uint8)
    numel = flat.numel()
    pad = (-numel) % per_byte
    if pad:
        flat = torch.cat((flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)))
    flat = flat.reshape(-1, per_byte)
    shifts = torch.arange(per_byte - 1, -1, -1, device=flat.device, dtype=torch.uint8) * bits
    return (flat << shifts).sum(dim=-1), numel


def _unpack_codes(packed, bits, numel):
    if bits == 8:
        return packed
    per_byte = 8 // bits
    shifts = torch.arange(per_byte - 1, -1, -1, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(-1)[:numel]


class FloatQATLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("RWKV-X Channel-Mix linears must have bias=False")
        self.weight = linear.weight
        self.bits = _bits(bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, _fake_quantize(self.weight, self.bits))

    @torch.no_grad()
    def pack(self):
        if self.bits == 8:
            return PackedFloatWeight(self.weight.detach().cpu().to(torch.float8_e4m3fn), None, self.weight.shape, self.bits)
        levels = _float_levels(self.bits, self.weight.device).to(self.weight.dtype)
        scale = self.weight.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(self.weight.dtype).eps)
        codes = (self.weight.detach() / scale).unsqueeze(-1).sub(levels).abs().argmin(dim=-1).to(torch.uint8)
        packed, numel = _pack_codes(codes, self.bits)
        return PackedFloatWeight(packed.cpu(), scale.cpu(), self.weight.shape, self.bits, numel)


class PackedFloatWeight(nn.Module):
    def __init__(self, codes, scale, shape, bits, numel=None):
        super().__init__()
        self.bits = _bits(bits)
        self.shape = tuple(shape)
        self.numel = numel or codes.numel()
        self.register_buffer("codes", codes)
        if scale is not None:
            self.register_buffer("scale", scale)

    def unpack(self, device, dtype):
        if self.bits == 8:
            return self.codes.to(device=device, dtype=dtype)
        levels = _float_levels(self.bits, device).to(dtype)
        codes = _unpack_codes(self.codes.to(device), self.bits, self.numel).long()
        return levels[codes].reshape(self.shape) * self.scale.to(device=device, dtype=dtype)

    def forward(self, x):
        return F.linear(x, self.unpack(x.device, x.dtype))


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
    print(f"[QAT] FP{bits} training weights")
    return n


def convert_qat(model: RWKVXModel, bits=None) -> int:
    bits = _bits(bits)
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, FloatQATLinear):
                setattr(cmix, name, mod.pack())
                n += 1
    return n


@torch.no_grad()
def calibrate(model, tokenizer, calib_texts, ctx_len, device, max_batches=64):
    return 0

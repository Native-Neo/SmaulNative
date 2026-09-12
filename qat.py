#!/usr/bin/env python3
# Low-bit floating-point QAT for RWKV-X.

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE, RWKV_CMix_x070

_CMIX_LINEAR_NAMES = ("key", "value")
_SUPPORTED_BITS = (2, 4, 8)
_LOWBIT_EXT = None

if "--qt" in sys.argv:
    i = sys.argv.index("--qt")
    if i + 1 >= len(sys.argv) or sys.argv[i + 1] not in {"2", "4", "8"}:
        raise SystemExit("--qt requires 2, 4, or 8")
    os.environ["SMAUL_QT_BITS"] = sys.argv[i + 1]
    os.environ["SMAUL_QT_REQUESTED"] = "1"
    sys.argv[i:i + 2] = ["--qat"]


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


def _quantize_codes(x, bits):
    levels = _float_levels(bits, x.device).to(x.dtype)
    scale = x.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).eps)
    codes = (x.detach() / scale).unsqueeze(-1).sub(levels).abs().argmin(dim=-1).to(torch.uint8)
    return codes, scale


def _fake_quantize(x, bits):
    if bits == 8:
        q = x.to(torch.float8_e4m3fn).to(x.dtype)
    else:
        codes, scale = _quantize_codes(x, bits)
        q = _float_levels(bits, x.device).to(x.dtype)[codes.long()] * scale
    return x + (q - x).detach()


def _pack_codes(codes, bits):
    if bits == 8:
        return codes.to(torch.float8_e4m3fn), codes.numel()
    per_byte = 8 // bits
    rows, cols = codes.shape
    flat = codes.reshape(rows, cols).to(torch.uint8)
    pad = (-cols) % per_byte
    if pad:
        flat = torch.cat((flat, torch.zeros((rows, pad), dtype=torch.uint8, device=flat.device)), dim=1)
    packed = flat.reshape(rows, -1, per_byte)
    shifts = torch.arange(per_byte - 1, -1, -1, device=flat.device, dtype=torch.uint8) * bits
    return (packed << shifts).sum(dim=-1), rows * cols


def _unpack_codes(packed, bits, numel):
    if bits == 8:
        return packed
    per_byte = 8 // bits
    shifts = torch.arange(per_byte - 1, -1, -1, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(-1)[:numel]


def _packed_linear(x, packed, scale, shape, bits, numel):
    if bits == 8:
        return F.linear(x, packed.to(device=x.device, dtype=x.dtype))
    if x.device.type != "cpu" or x.dtype != torch.float32:
        return F.linear(x, packed.unpack(x.device, x.dtype))
    out_features, in_features = shape
    if x.shape[-1] != in_features:
        raise ValueError(f"input features {x.shape[-1]} != {in_features}")
    x2 = x.reshape(-1, in_features).contiguous()
    result = _load_lowbit().packed_linear(
        x2,
        packed.codes,
        scale.reshape(-1).float().contiguous(),
        bits,
        out_features,
        in_features,
    )
    return result.reshape(*x.shape[:-1], out_features)


class _PackedQATFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bits):
        ctx.save_for_backward(x, weight)
        ctx.bits = bits
        out_features, in_features = weight.shape
        if x.device.type == "cpu" and x.dtype == torch.float32 and bits < 8:
            x2 = x.reshape(-1, in_features).contiguous()
            y = _load_lowbit().qat_linear(x2, weight, bits)
            return y.reshape(*x.shape[:-1], out_features)
        q = _fake_quantize(weight, bits)
        return F.linear(x, q)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        bits = ctx.bits
        q = _fake_quantize(weight, bits)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits < 8 and x.device.type == "cpu" and x.dtype == torch.float32:
            return _PackedQATFunction.apply(x, self.weight, self.bits)
        return F.linear(x, _fake_quantize(self.weight, self.bits))

    @torch.no_grad()
    def pack(self):
        if self.bits == 8:
            return PackedFloatWeight(self.weight.detach().cpu().to(torch.float8_e4m3fn), None, self.weight.shape, self.bits)
        codes, scale = _quantize_codes(self.weight, self.bits)
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
        return _packed_linear(x, self, self.scale, self.shape, self.bits, self.numel) if self.bits < 8 else F.linear(x, self.codes.to(device=x.device, dtype=x.dtype))


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

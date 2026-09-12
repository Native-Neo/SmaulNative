#!/usr/bin/env python3
# Real post-training FP2/FP4/FP8 quantization for RWKV-X.

import argparse
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


def _bits(bits):
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be 2, 4, or 8")
    return bits


def _levels(bits, device, dtype=torch.float32):
    if bits == 2:
        values = (-1.0, 0.0, 1.0)
    elif bits == 4:
        values = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
    else:
        raise ValueError("FP8 uses E4M3")
    return torch.tensor(values, device=device, dtype=dtype)


def _quantize_codes(weight, bits):
    levels = _levels(bits, weight.device, weight.dtype)
    scale = weight.detach().abs().amax(dim=0, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps)
    codes = (weight.detach() / scale).unsqueeze(-1).sub(levels).abs().argmin(dim=-1).to(torch.uint8)
    return codes, scale


def _pack_codes(codes, bits):
    per_byte = 8 // bits
    rows, cols = codes.shape
    pad = (-cols) % per_byte
    if pad:
        codes = torch.cat((codes, torch.zeros((rows, pad), dtype=torch.uint8, device=codes.device)), dim=1)
    grouped = codes.reshape(rows, -1, per_byte)
    shifts = torch.arange(per_byte - 1, -1, -1, device=codes.device, dtype=torch.uint8) * bits
    return (grouped << shifts).sum(dim=-1)


def _unpack_codes(packed, bits, numel):
    per_byte = 8 // bits
    shifts = torch.arange(per_byte - 1, -1, -1, device=packed.device, dtype=torch.uint8) * bits
    return ((packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(-1)[:numel]


class QuantizedLinear(nn.Module):
    def __init__(self, packed, scale, weight_shape):
        super().__init__()
        shape = tuple(int(v) for v in weight_shape)
        if len(shape) != 2:
            raise ValueError("quantized linear weight must be 2D")
        self.weight_shape = shape
        self.register_buffer("packed", packed.contiguous())
        self.register_buffer("scale", scale.contiguous())
        self.register_buffer("weight_shape", torch.tensor(shape, dtype=torch.int64))
        self.bits = self._infer_bits()

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, bits):
        bits = _bits(bits)
        weight = linear.weight.detach().cpu().float()
        if bits == 8:
            packed = weight.to(torch.float8_e4m3fn)
            scale = torch.empty(0, dtype=torch.float32)
        else:
            codes, scale = _quantize_codes(weight, bits)
            packed = _pack_codes(codes, bits)
            scale = scale.cpu().float()
        return cls(packed.cpu(), scale, weight.shape)

    def _infer_bits(self):
        if self.packed.dtype == torch.float8_e4m3fn:
            return 8
        in_features = self.weight_shape[1]
        packed_cols = self.packed.shape[1]
        for bits in (2, 4):
            if packed_cols == (in_features + 8 // bits - 1) // (8 // bits):
                return bits
        raise ValueError("cannot infer FP2/FP4 packing width")

    def unpack(self, device=None, dtype=torch.float32):
        device = device or self.packed.device
        if self.bits == 8:
            return self.packed.to(device=device, dtype=dtype)
        codes = _unpack_codes(self.packed.to(device), self.bits, self.weight_shape[0] * self.weight_shape[1]).long()
        return (_levels(self.bits, device, dtype)[codes] * self.scale.to(device=device, dtype=dtype)).reshape(self.weight_shape)

    def forward(self, x):
        if self.bits < 8 and x.device.type == "cpu" and x.dtype == torch.float32:
            out_features, in_features = self.weight_shape
            if x.shape[-1] != in_features:
                raise ValueError(f"input features {x.shape[-1]} != {in_features}")
            x2 = x.reshape(-1, in_features).contiguous()
            y = _load_lowbit().packed_linear(
                x2, self.packed, self.scale.reshape(-1).float().contiguous(),
                self.bits, out_features, in_features,
            )
            return y.reshape(*x.shape[:-1], out_features)
        return F.linear(x, self.unpack(x.device, x.dtype))


def _iter_cmix_modules(model):
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


@torch.no_grad()
def quantize_model(model, bits):
    bits = _bits(bits)
    count = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            linear = getattr(cmix, name)
            if isinstance(linear, nn.Linear):
                setattr(cmix, name, QuantizedLinear.from_linear(linear, bits))
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Post-training FP2/FP4/FP8 quantization")
    parser.add_argument("--qt", type=int, choices=_SUPPORTED_BITS, required=True)
    parser.add_argument("--model", default="./SmaulNative")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    source = Path(args.model)
    output = Path(args.output or f"{source}-FP{args.qt}")
    model = RWKVXModel.from_pretrained(source).cpu()
    count = quantize_model(model, args.qt)
    model.cfg.quantization_bits = args.qt
    model.save_pretrained(output, dtype="fp32", include_upstream=False)
    tokenizer = source / "tokenizer.json"
    if tokenizer.exists():
        output.mkdir(parents=True, exist_ok=True)
        (output / "tokenizer.json").write_bytes(tokenizer.read_bytes())
    print(f"[QT] FP{args.qt} quantized {count} linears")
    print(f"[QT] saved {output}")


if __name__ == "__main__":
    main()

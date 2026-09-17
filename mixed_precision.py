#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

_FP4 = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
_FP6 = None


def _fp6_levels(device, dtype):
    global _FP6
    if _FP6 is None:
        values = []
        for e in range(8):
            for m in range(4):
                values.append(0.0 if e == 0 and m == 0 else (m / 4) * 2 ** -2 if e == 0 else (1 + m / 4) * 2 ** (e - 3))
        _FP6 = torch.tensor(values)
    return _FP6.to(device=device, dtype=dtype)


def _levels(bits, device, dtype):
    if bits == 4:
        return _FP4.to(device=device, dtype=dtype)
    if bits == 6:
        return _fp6_levels(device, dtype)
    raise ValueError("custom floating precision must be FP4 or FP6")


def quantize(x, bits, dim=-1):
    if bits not in (4, 6, 8):
        return x
    if bits == 8:
        max_level = 448.0
        scale = x.detach().abs().amax(dim=dim, keepdim=True).div(max_level).clamp_min(torch.finfo(x.dtype).eps)
        q = (x.detach() / scale).to(torch.float8_e4m3fn).to(x.dtype) * scale
    else:
        levels = _levels(bits, x.device, x.dtype)
        max_level = levels[-1]
        scale = x.detach().abs().amax(dim=dim, keepdim=True).div(max_level).clamp_min(torch.finfo(x.dtype).eps)
        magnitude = (x.detach() / scale).abs().unsqueeze(-1)
        code = magnitude.sub(levels).abs().argmin(-1)
        q = levels[code] * scale
        q = q.copysign(x.detach())
    return x + (q - x).detach()


class MixedPrecisionLinear(nn.Module):
    def __init__(self, linear, weight_bits, activation_bits):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.weight_bits = int(weight_bits)
        self.activation_bits = int(activation_bits)

    def forward(self, x):
        xq = quantize(x.float(), self.activation_bits)
        wq = quantize(self.weight.float(), self.weight_bits, dim=1)
        return F.linear(xq, wq, self.bias.float() if self.bias is not None else None)


def _replace(root, path, weight_bits, activation_bits):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    name = parts[-1]
    linear = getattr(parent, name)
    setattr(parent, name, MixedPrecisionLinear(linear, weight_bits, activation_bits))


def apply(model):
    root = getattr(model, "_orig_mod", model)
    targets = []
    for path, module in root.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if path == "head":
            targets.append((path, 8, 8))
        elif ".att." in f".{path}.":
            targets.append((path, 8, 8))
        elif path.endswith(".key") or path.endswith(".value"):
            targets.append((path, 4, 6) if ".ffn." in f".{path}." else (path, 6, 6))
        else:
            targets.append((path, 6, 6))
    for path, wb, ab in sorted(targets, key=lambda item: item[0].count("."), reverse=True):
        _replace(root, path, wb, ab)
    for parameter in root.parameters():
        if parameter.requires_grad:
            parameter.register_hook(lambda grad: quantize(grad.float(), 8).to(grad.dtype))
    root.cfg.mixed_precision = True
    root.cfg.mixed_weight_bits = 6
    root.cfg.mixed_activation_bits = 6
    root.cfg.mixed_gradient_bits = 8
    return len(targets)


def describe(model):
    counts = {4: 0, 6: 0, 8: 0}
    for module in model.modules():
        if isinstance(module, MixedPrecisionLinear):
            counts[module.weight_bits] += module.weight.numel()
    return counts

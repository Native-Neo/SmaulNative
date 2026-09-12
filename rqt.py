#!/usr/bin/env python3
# Real packed low-bit training for RWKV-X.

import torch
import torch.nn as nn

from rwkv_x_core import RWKVXModel
from qt import QuantizedLinear

_SUPPORTED_BITS = (2, 4, 8)


def _bits(bits):
    if bits not in _SUPPORTED_BITS:
        raise ValueError("bits must be 2, 4, or 8")
    return bits


class _RQTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, quant):
        ctx.save_for_backward(x, weight)
        ctx.quant = quant
        return quant(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        q = ctx.quant.unpack(x.device, x.dtype)
        x2 = x.reshape(-1, x.shape[-1])
        go = grad_output.reshape(-1, grad_output.shape[-1])
        grad_x = go.matmul(q).reshape_as(x)
        grad_weight = go.transpose(0, 1).matmul(x2)
        return grad_x, grad_weight, None


class RealQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int):
        super().__init__()
        if linear.bias is not None:
            self.bias = linear.bias
        else:
            self.register_parameter("bias", None)
        self.weight = linear.weight
        self.bits = _bits(bits)
        self.quant = QuantizedLinear.from_linear(linear, self.bits)
        self._make_quant_nonpersistent()

    def _make_quant_nonpersistent(self):
        self.quant._non_persistent_buffers_set.update({"packed", "scale", "weight_shape"})

    @torch.no_grad()
    def refresh(self):
        self.quant = QuantizedLinear.from_linear(self._linear_view(), self.bits)
        self._make_quant_nonpersistent()

    def _linear_view(self):
        linear = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False, device=self.weight.device, dtype=self.weight.dtype)
        linear.weight = self.weight
        return linear

    def forward(self, x):
        if self.weight.device != x.device:
            self.quant = self.quant.to(x.device)
        out = _RQTFunction.apply(x, self.weight, self.quant)
        if self.bias is not None:
            out = out + self.bias
        return out


def _iter_linear_modules(model):
    root = getattr(model, "_orig_mod", model)
    for module in root.modules():
        if isinstance(module, nn.Linear):
            yield module


def prepare_rqt(model: RWKVXModel, bits):
    bits = _bits(bits)
    root = getattr(model, "_orig_mod", model)
    targets = [module for module in root.modules() if isinstance(module, nn.Linear)]
    for module in targets:
        parent = root
        parts = []
        for name, child in root.named_modules():
            if child is module:
                parts = name.split(".")
                break
        if not parts:
            continue
        for part in parts[:-1]:
            parent = getattr(parent, part)
        name = parts[-1]
        setattr(parent, name, RealQuantLinear(module, bits))
    print(f"[RQT] FP{bits} packed training weights")
    print(f"[RQT] real-quantizing {len(targets)} linear weights")
    return len(targets)


@torch.no_grad()
def refresh_rqt(model):
    root = getattr(model, "_orig_mod", model)
    for module in root.modules():
        if isinstance(module, RealQuantLinear):
            module.refresh()

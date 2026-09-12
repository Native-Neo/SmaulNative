#!/usr/bin/env python3
# Real packed low-bit training for RWKV-X.

import torch
import torch.nn as nn

from rwkv_x_core import RWKVXModel, RWKV_CMix_MoE, RWKV_CMix_x070
from qt import QuantizedLinear

_CMIX_LINEAR_NAMES = ("key", "value")
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
            raise ValueError("RWKV-X Channel-Mix linears must have bias=False")
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
        return _RQTFunction.apply(x, self.weight, self.quant)


def _iter_cmix_modules(model):
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


def prepare_rqt(model: RWKVXModel, bits):
    bits = _bits(bits)
    count = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, nn.Linear):
                setattr(cmix, name, RealQuantLinear(mod, bits))
                count += 1
    print(f"[RQT] FP{bits} packed training weights")
    return count


@torch.no_grad()
def refresh_rqt(model):
    root = getattr(model, "_orig_mod", model)
    for module in root.modules():
        if isinstance(module, RealQuantLinear):
            module.refresh()

#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

FP4, FP6, FP8 = 4, 6, 8


def _bits(bits):
    if bits not in (FP4, FP6, FP8):
        raise ValueError("RQT supports FP4, FP6, and FP8")
    return bits


def _levels(bits, device, dtype):
    if bits == FP8:
        raise ValueError("FP8 uses native torch.float8_e4m3fn")
    ebits, mbits = (2, 1) if bits == FP4 else (3, 2)
    bias = (1 << (ebits - 1)) - 1
    values = []
    for code in range(1 << bits):
        exp = (code >> mbits) & ((1 << ebits) - 1)
        mant = code & ((1 << mbits) - 1)
        sign = -1.0 if code >> (bits - 1) else 1.0
        value = (mant / (1 << mbits)) * 2 ** (1 - bias) if exp == 0 else (1 + mant / (1 << mbits)) * 2 ** (exp - bias)
        values.append(sign * value)
    return torch.tensor(values, device=device, dtype=dtype)


def _pack(codes, bits):
    if bits == FP4:
        if codes.numel() & 1:
            codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8, device=codes.device)))
        c = codes.reshape(-1, 2)
        return ((c[:, 0] << 4) | c[:, 1]).to(torch.uint8)
    if bits == FP6:
        pad = (-codes.numel()) % 4
        if pad:
            codes = torch.cat((codes, torch.zeros(pad, dtype=torch.uint8, device=codes.device)))
        c = codes.reshape(-1, 4)
        out = torch.empty(c.shape[0] * 3, dtype=torch.uint8, device=codes.device)
        out[0::3] = (c[:, 0] << 2) | (c[:, 1] >> 4)
        out[1::3] = ((c[:, 1] & 15) << 4) | (c[:, 2] >> 2)
        out[2::3] = ((c[:, 2] & 3) << 6) | c[:, 3]
        return out
    return codes.to(torch.float8_e4m3fn)


def _unpack(packed, bits, count):
    if bits == FP4:
        out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
        out[0::2], out[1::2] = packed >> 4, packed & 15
        return out[:count]
    if bits == FP6:
        p = packed.reshape(-1, 3)
        out = torch.empty(p.shape[0] * 4, dtype=torch.uint8, device=packed.device)
        out[0::4] = p[:, 0] >> 2
        out[1::4] = ((p[:, 0] & 3) << 4) | (p[:, 1] >> 4)
        out[2::4] = ((p[:, 1] & 15) << 2) | (p[:, 2] >> 6)
        out[3::4] = p[:, 2] & 63
        return out[:count]
    return packed


def _encode(weight, bits):
    if bits == FP8:
        return _pack(weight, bits), torch.empty(0, dtype=torch.float32, device=weight.device)
    levels = _levels(bits, weight.device, weight.dtype)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps) / levels.abs().max()
    codes = (weight / scale).unsqueeze(-1).sub(levels).abs().argmin(-1).to(torch.uint8)
    return _pack(codes.flatten(), bits), scale.squeeze(1).float()


class RQTLinear(nn.Module):
    def __init__(self, linear, bits):
        super().__init__()
        self.bits = _bits(bits)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_parameter("bias", linear.bias)
        self.register_buffer("packed", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(0, dtype=torch.float32))
        self._grad = None
        self._replace_weight(linear.weight)

    @torch.no_grad()
    def _replace_weight(self, weight):
        self.packed, self.scale = _encode(weight.detach().float(), self.bits)

    def unpack(self, dtype=torch.float32):
        if self.bits == FP8:
            return self.packed.to(dtype)
        codes = _unpack(self.packed, self.bits, self.in_features * self.out_features).long()
        levels = _levels(self.bits, self.packed.device, dtype)[codes]
        return (levels * self.scale.to(dtype)[:, None]).reshape(self.out_features, self.in_features)

    def _capture_grad(self, grad):
        self._grad = grad.detach().float()
        return grad

    def forward(self, x):
        weight = self.unpack(torch.float32).detach().requires_grad_(True)
        weight.register_hook(self._capture_grad)
        return F.linear(x, weight, self.bias)

    @torch.no_grad()
    def step(self, update, decay):
        weight = self.unpack(torch.float32)
        if decay:
            weight.mul_(1 - decay)
        self._replace_weight(weight - update)
        self._grad = None

    def zero_grad(self):
        self._grad = None


class RQTLion:
    def __init__(self, model, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        self.model, self.lr, self.betas, self.weight_decay = model, lr, betas, weight_decay
        self.rqt_state, self.param_state = {}, {}
        self.params = [p for p in model.parameters() if p.requires_grad]

    def _modules(self):
        return [m for m in self.model.modules() if isinstance(m, RQTLinear)]

    def zero_grad(self, set_to_none=True):
        for module in self._modules():
            module.zero_grad()
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for module in self._modules():
            if module._grad is None:
                continue
            grad = module._grad
            avg = self.rqt_state.setdefault(module, torch.zeros_like(grad, dtype=torch.float32))
            avg.mul_(b2).add_(grad, alpha=1 - b2)
            update = avg.mul(b1).add_(grad, alpha=1 - b1).sign() * self.lr
            module.step(update, self.lr * self.weight_decay)
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.float()
            avg = self.param_state.setdefault(param, torch.zeros_like(param, dtype=torch.float32))
            avg.mul_(b2).add_(grad, alpha=1 - b2)
            if self.weight_decay:
                param.mul_(1 - self.lr * self.weight_decay)
            param.add_(avg.mul(b1).add_(grad, alpha=1 - b1).sign(), alpha=-self.lr)

    def state_dict(self):
        return {"param_state": {str(i): v.cpu() for i, v in enumerate(self.param_state.values())}, "rqt_state": {str(i): v.cpu() for i, v in enumerate(self.rqt_state.values())}}


def _replace(root, name, bits):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], RQTLinear(getattr(parent, parts[-1]), bits))


def prepare_rqt(model, bits=FP6):
    bits = _bits(bits)
    root = getattr(model, "_orig_mod", model)
    targets = [(name, module) for name, module in root.named_modules() if isinstance(module, nn.Linear)]
    for name, _ in reversed(targets):
        _replace(root, name, bits)
    model.cfg.rqt_bits = bits
    print(f"[RQT] FP{bits} packed weights | {len(targets)} linear layers")
    return len(targets)


def prepare_mixed_rqt(model):
    root = getattr(model, "_orig_mod", model)
    targets = [(name, module) for name, module in root.named_modules() if isinstance(module, nn.Linear)]
    for name, _ in reversed(targets):
        bits = FP8 if name == "head" or ".att." in f".{name}." else FP6
        _replace(root, name, bits)
    model.cfg.rqt_mixed = True
    print(f"[RQT] mixed FP6/FP8 packed weights | {len(targets)} linear layers")
    return len(targets)

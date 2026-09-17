#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

FP4, FP6, FP8 = 4, 6, 8
_LEVEL_CACHE = {}


def _bits(bits):
    if bits not in (FP4, FP6, FP8):
        raise ValueError("RQT supports FP4, FP6, and FP8")
    return bits


def _levels(bits, device, dtype):
    if bits == FP8:
        raise ValueError("FP8 uses native torch.float8_e4m3fn")
    key = (bits, device.type, device.index, dtype)
    cached = _LEVEL_CACHE.get(key)
    if cached is not None:
        return cached
    ebits, mbits = (2, 1) if bits == FP4 else (3, 2)
    bias = (1 << (ebits - 1)) - 1
    values = []
    for code in range(1 << bits):
        exp = (code >> mbits) & ((1 << ebits) - 1)
        mant = code & ((1 << mbits) - 1)
        sign = -1.0 if code >> (bits - 1) else 1.0
        value = (mant / (1 << mbits)) * 2 ** (1 - bias) if exp == 0 else (1 + mant / (1 << mbits)) * 2 ** (exp - bias)
        values.append(sign * value)
    levels = torch.tensor(values, device=device, dtype=dtype)
    _LEVEL_CACHE[key] = levels
    return levels


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
        self.register_buffer("bit_width", torch.tensor(self.bits, dtype=torch.uint8))
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
        self.param_names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}

    def _modules(self):
        return [(name, module) for name, module in self.model.named_modules() if isinstance(module, RQTLinear)]

    def zero_grad(self, set_to_none=True):
        for _, module in self._modules():
            module.zero_grad()
        for param in self.params:
            param.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm):
        grads = [m._grad for _, m in self._modules() if m._grad is not None]
        grads += [p.grad.float() for p in self.params if p.grad is not None]
        if not grads:
            return torch.tensor(0.0)
        total = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for _, module in self._modules():
                if module._grad is not None:
                    module._grad.mul_(scale)
            for param in self.params:
                if param.grad is not None:
                    param.grad.mul_(scale)
        return total

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for _, module in self._modules():
            if module._grad is None:
                continue
            grad = module._grad
            avg = self.rqt_state.setdefault(module, torch.zeros_like(grad, dtype=torch.float32))
            update = avg.mul(b1).add_(grad, alpha=1 - b1).sign() * self.lr
            avg.mul_(b2).add_(grad, alpha=1 - b2)
            module.step(update, self.lr * self.weight_decay)
        for param in self.params:
            if param.grad is None:
                continue
            grad = param.grad.float()
            avg = self.param_state.setdefault(param, torch.zeros_like(param, dtype=torch.float32))
            update = avg.mul(b1).add_(grad, alpha=1 - b1).sign()
            avg.mul_(b2).add_(grad, alpha=1 - b2)
            if self.weight_decay:
                param.mul_(1 - self.lr * self.weight_decay)
            param.add_(update, alpha=-self.lr)

    def state_dict(self):
        modules = {name: state.cpu() for name, module in self._modules() if (state := self.rqt_state.get(module)) is not None}
        params = {self.param_names[id(param)]: state.cpu() for param, state in self.param_state.items() if id(param) in self.param_names}
        return {
            "version": 2,
            "lr": self.lr,
            "betas": self.betas,
            "weight_decay": self.weight_decay,
            "param_state": params,
            "rqt_state": modules,
        }

    def load_state_dict(self, state):
        if "lr" in state:
            self.lr = float(state["lr"])
        if "betas" in state:
            self.betas = tuple(state["betas"])
        if "weight_decay" in state:
            self.weight_decay = float(state["weight_decay"])
        rqt_state = state.get("rqt_state", {})
        param_state = state.get("param_state", {})
        modules = dict(self._modules())
        named_params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        if all(key.isdigit() for key in rqt_state) and rqt_state:
            rqt_values = list(rqt_state.values())
            if len(rqt_values) > len(modules):
                raise ValueError("RQT optimizer state has too many entries")
            rqt_state = {name: value for (name, _), value in zip(modules.items(), rqt_values)}
        if all(key.isdigit() for key in param_state) and param_state:
            param_values = list(param_state.values())
            if len(param_values) > len(named_params):
                raise ValueError("optimizer state has too many parameter entries")
            param_state = {name: value for (name, _), value in zip(named_params.items(), param_values)}
        self.rqt_state = {}
        for name, value in rqt_state.items():
            if name not in modules:
                raise ValueError(f"unknown RQT optimizer module: {name}")
            module = modules[name]
            if tuple(value.shape) != tuple(module.unpack().shape):
                raise ValueError(f"invalid RQT optimizer state shape for {name}")
            self.rqt_state[module] = value.to(module.packed.device, dtype=torch.float32)
        self.param_state = {}
        for name, value in param_state.items():
            if name not in named_params:
                raise ValueError(f"unknown optimizer parameter: {name}")
            param = named_params[name]
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"invalid optimizer state shape for {name}")
            self.param_state[param] = value.to(param.device, dtype=torch.float32)


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
        bits = FP8 if name == "head" or ".att." in f".{name}." else FP4 if ".ffn." in f".{name}." else FP6
        _replace(root, name, bits)
    model.cfg.rqt_mixed = True
    print(f"[RQT] mixed FP4/FP6/FP8 packed weights | {len(targets)} linear layers")
    return len(targets)


def load_rqt_checkpoint(in_dir):
    from pathlib import Path
    from safetensors.torch import load_file
    from rwkv_x_core import RWKVXConfig, RWKVXModel

    in_dir = Path(in_dir)
    cfg = RWKVXConfig.load(in_dir / "config.json")
    sd = load_file(str(in_dir / "model.safetensors"))
    model = RWKVXModel(cfg)
    packed = [k[:-7] for k in sd if k.endswith(".packed")]
    if not packed:
        raise RuntimeError("RQT checkpoint has no packed weights")
    mixed = any(sd[path + ".packed"].dtype == torch.float8_e4m3fn for path in packed) or any(path + ".bit_width" in sd for path in packed)
    for path in packed:
        parent_path, name = path.rsplit(".", 1) if "." in path else ("", path)
        parent = model.get_submodule(parent_path) if parent_path else model
        linear = getattr(parent, name)
        if mixed and path + ".bit_width" in sd:
            bits = _bits(int(sd[path + ".bit_width"].item()))
        elif mixed and sd[path + ".packed"].dtype == torch.float8_e4m3fn:
            bits = FP8
        else:
            bits = _bits(cfg.rqt_bits)
        setattr(parent, name, RQTLinear(linear, bits))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [key for key in missing if not key.endswith(".bit_width")]
    if missing or unexpected:
        raise RuntimeError(f"invalid RQT checkpoint: missing={missing}, unexpected={unexpected}")
    return model


def _install_checkpoint_loader():
    from pathlib import Path
    from safetensors.torch import load_file
    from rwkv_x_core import RWKVXModel

    original = RWKVXModel.from_pretrained.__func__

    @classmethod
    def from_pretrained(cls, in_dir):
        path = Path(in_dir)
        sd = load_file(str(path / "model.safetensors"))
        if any(key.endswith(".packed") for key in sd):
            return load_rqt_checkpoint(path)
        return original(cls, in_dir)

    RWKVXModel.from_pretrained = from_pretrained


_install_checkpoint_loader()

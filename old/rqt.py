#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

FP4, FP6, FP8 = 4, 6, 8
_LEVEL_CACHE = {}
_TABLE_CACHE = {}
_BLOCK_ROWS = 256
_RQT_EXT = None


def _bits(bits):
    if bits not in (FP4, FP6, FP8): raise ValueError("RQT supports FP4, FP6, and FP8")
    return bits


def _levels(bits, device, dtype):
    if bits == FP8: raise ValueError("FP8 uses native torch.float8_e4m3fn")
    key = (bits, device.type, device.index, dtype)
    if key in _LEVEL_CACHE: return _LEVEL_CACHE[key]
    ebits, mbits = (2, 1) if bits == FP4 else (3, 2)
    bias = (1 << (ebits - 1)) - 1
    values = []
    for code in range(1 << bits):
        exp = (code >> mbits) & ((1 << ebits) - 1); mant = code & ((1 << mbits) - 1)
        sign = -1.0 if code >> (bits - 1) else 1.0
        value = (mant / (1 << mbits)) * 2 ** (1 - bias) if exp == 0 else (1 + mant / (1 << mbits)) * 2 ** (exp - bias)
        values.append(sign * value)
    out = torch.tensor(values, device=device, dtype=dtype); _LEVEL_CACHE[key] = out
    return out


def _quant_table(bits, device, dtype):
    key = (bits, device.type, device.index, dtype)
    if key in _TABLE_CACHE: return _TABLE_CACHE[key]
    levels = _levels(bits, device, dtype); ordered, codes = torch.sort(levels)
    out = ((ordered[:-1] + ordered[1:]) * 0.5, codes.to(torch.long), ordered)
    _TABLE_CACHE[key] = out
    return out


def _packed_row_bytes(in_features, bits):
    if bits == FP4: return (in_features + 1) // 2
    if bits == FP6: return (in_features + 3) // 4 * 3
    return in_features


def _pack(codes, bits):
    if bits == FP4:
        if codes.numel() & 1: codes = torch.cat((codes, torch.zeros(1, dtype=torch.uint8, device=codes.device)))
        c = codes.reshape(-1, 2); return ((c[:, 0] << 4) | c[:, 1]).to(torch.uint8)
    if bits == FP6:
        pad = (-codes.numel()) % 4
        if pad: codes = torch.cat((codes, torch.zeros(pad, dtype=torch.uint8, device=codes.device)))
        c = codes.reshape(-1, 4); out = torch.empty(c.shape[0] * 3, dtype=torch.uint8, device=codes.device)
        out[0::3] = (c[:, 0] << 2) | (c[:, 1] >> 4); out[1::3] = ((c[:, 1] & 15) << 4) | (c[:, 2] >> 2); out[2::3] = ((c[:, 2] & 3) << 6) | c[:, 3]
        return out
    return codes.to(torch.float8_e4m3fn)


def _pack_rows(codes, bits):
    if bits == FP4:
        pad = codes.shape[1] & 1
        if pad: codes = torch.cat((codes, torch.zeros(codes.shape[0], 1, dtype=torch.uint8, device=codes.device)), dim=1)
        c = codes.reshape(codes.shape[0], -1, 2)
        return ((c[..., 0] << 4) | c[..., 1]).to(torch.uint8).reshape(-1)
    if bits == FP6:
        pad = (-codes.shape[1]) % 4
        if pad: codes = torch.cat((codes, torch.zeros(codes.shape[0], pad, dtype=torch.uint8, device=codes.device)), dim=1)
        c = codes.reshape(codes.shape[0], -1, 4)
        out = torch.empty((codes.shape[0], c.shape[1] * 3), dtype=torch.uint8, device=codes.device)
        out[:, 0::3] = (c[..., 0] << 2) | (c[..., 1] >> 4)
        out[:, 1::3] = ((c[..., 1] & 15) << 4) | (c[..., 2] >> 2)
        out[:, 2::3] = ((c[..., 2] & 3) << 6) | c[..., 3]
        return out.reshape(-1)
    return codes.to(torch.float8_e4m3fn).reshape(-1)


def _unpack(packed, bits, count):
    if bits == FP4:
        out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device); out[0::2], out[1::2] = packed >> 4, packed & 15
        return out[:count]
    if bits == FP6:
        p = packed.reshape(-1, 3); out = torch.empty(p.shape[0] * 4, dtype=torch.uint8, device=packed.device)
        out[0::4] = p[:, 0] >> 2; out[1::4] = ((p[:, 0] & 3) << 4) | (p[:, 1] >> 4); out[2::4] = ((p[:, 1] & 15) << 2) | (p[:, 2] >> 6); out[3::4] = p[:, 2] & 63
        return out[:count]
    return packed


def _encode(weight, bits):
    weight = weight.float()
    if bits == FP8: return weight.to(torch.float8_e4m3fn).reshape(-1), torch.empty(0, dtype=torch.float32, device=weight.device)
    levels = _levels(bits, weight.device, weight.dtype)
    scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(torch.finfo(weight.dtype).eps) / levels.abs().max()
    boundaries, codes, _ = _quant_table(bits, weight.device, weight.dtype)
    quant_codes = codes[torch.bucketize(weight / scale, boundaries)].to(torch.uint8)
    return _pack_rows(quant_codes, bits), scale.squeeze(1)


def _native_rqt():
    global _RQT_EXT
    if _RQT_EXT is not None: return _RQT_EXT
    try:
        from cpu_backend import _load
        _RQT_EXT = _load()
    except Exception as exc:
        _RQT_EXT = False
        warnings.warn(f"native RQT extension unavailable; using packed PyTorch fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
    return _RQT_EXT


class _RQTLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, in_features, out_features, bits, module, grad_anchor):
        ext = _native_rqt()
        x2 = x.reshape(-1, in_features).contiguous()
        ctx.save_for_backward(x2, packed, scale)
        ctx.in_features, ctx.out_features, ctx.bits, ctx.module, ctx.shape = in_features, out_features, bits, module, x.shape
        ctx.need_input_grad = ctx.needs_input_grad[0]
        out = ext.rqt_linear_forward(x2, packed, scale, in_features, out_features, bits).reshape(*x.shape[:-1], out_features)
        return out * grad_anchor

    @staticmethod
    def backward(ctx, grad_output):
        x, packed, scale = ctx.saved_tensors
        grad = grad_output.reshape(-1, ctx.out_features).contiguous().float()
        ext = _native_rqt()
        grad_x = (ext.rqt_linear_backward_input(grad, packed, scale, ctx.in_features, ctx.out_features, ctx.bits).reshape(ctx.shape)
                  if ctx.need_input_grad else None)
        if ctx.module.trainable:
            weight_grad = ext.rqt_linear_backward_weight(x, grad, ctx.in_features, ctx.out_features)
            if ctx.module._grad is None:
                ctx.module._grad = weight_grad
            else:
                ctx.module._grad.add_(weight_grad)
        return grad_x, None, None, None, None, None, None, None


class _PackedTorchLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packed, scale, in_features, out_features, bits, module, grad_anchor):
        x2 = x.reshape(-1, in_features)
        outputs = []
        for start in range(0, out_features, _BLOCK_ROWS):
            end = min(start + _BLOCK_ROWS, out_features)
            outputs.append(F.linear(x2, module.unpack_rows(start, end, x.dtype)))
        ctx.save_for_backward(x, packed, scale)
        ctx.in_features, ctx.out_features, ctx.bits, ctx.module, ctx.shape = in_features, out_features, bits, module, x.shape
        ctx.need_input_grad = ctx.needs_input_grad[0]
        return torch.cat(outputs, dim=1).reshape(*x.shape[:-1], out_features) * grad_anchor

    @staticmethod
    def backward(ctx, grad_output):
        x, packed, scale = ctx.saved_tensors
        x2 = x.reshape(-1, ctx.in_features)
        grad2 = grad_output.reshape(-1, ctx.out_features)
        grad_x = torch.zeros_like(x2) if ctx.need_input_grad else None
        weight_grad = torch.zeros(ctx.out_features, ctx.in_features, dtype=torch.float32, device=x.device) if ctx.module.trainable else None
        for start in range(0, ctx.out_features, _BLOCK_ROWS):
            end = min(start + _BLOCK_ROWS, ctx.out_features)
            weight = ctx.module.unpack_rows(start, end, x.dtype)
            block_grad = grad2[:, start:end]
            if ctx.need_input_grad:
                grad_x.add_(block_grad @ weight)
            if ctx.module.trainable:
                weight_grad[start:end].copy_(block_grad.float().transpose(0, 1) @ x2.float())
        if ctx.module.trainable:
            if ctx.module._grad is None:
                ctx.module._grad = weight_grad
            else:
                ctx.module._grad.add_(weight_grad)
        return (grad_x.reshape(ctx.shape) if ctx.need_input_grad else None), None, None, None, None, None, None, None


class RQTLinear(nn.Module):
    def __init__(self, linear, bits):
        super().__init__(); self.bits = _bits(bits); self.in_features, self.out_features = linear.in_features, linear.out_features
        self.register_parameter("bias", linear.bias); self.register_buffer("packed", torch.empty(0, dtype=torch.uint8)); self.register_buffer("scale", torch.empty(0, dtype=torch.float32)); self.register_buffer("bit_width", torch.tensor(self.bits, dtype=torch.uint8))
        self._grad = None; self.trainable = True; self._replace_weight(linear.weight)

    def _row_slice(self, start, end): return slice(start * _packed_row_bytes(self.in_features, self.bits), end * _packed_row_bytes(self.in_features, self.bits))

    @torch.no_grad()
    def _replace_weight(self, weight): self.packed, self.scale = _encode(weight.detach(), self.bits)

    def unpack_rows(self, start, end, dtype=torch.float32):
        rows = end - start; packed = self.packed[self._row_slice(start, end)]
        if self.bits == FP8: return packed.to(dtype).reshape(rows, self.in_features)
        row_bytes = _packed_row_bytes(self.in_features, self.bits)
        row_packed = packed.reshape(rows, row_bytes)
        if self.bits == FP4:
            codes = torch.empty(rows, row_bytes * 2, dtype=torch.uint8, device=packed.device)
            codes[:, 0::2] = row_packed >> 4
            codes[:, 1::2] = row_packed & 15
        else:
            groups = row_packed.reshape(rows, -1, 3)
            codes = torch.empty(rows, groups.shape[1] * 4, dtype=torch.uint8, device=packed.device)
            codes[:, 0::4] = groups[:, :, 0] >> 2
            codes[:, 1::4] = ((groups[:, :, 0] & 3) << 4) | (groups[:, :, 1] >> 4)
            codes[:, 2::4] = ((groups[:, :, 1] & 15) << 2) | (groups[:, :, 2] >> 6)
            codes[:, 3::4] = groups[:, :, 2] & 63
        codes = codes[:, :self.in_features].long()
        levels = _levels(self.bits, packed.device, dtype)[codes]
        return levels * self.scale[start:end].to(dtype)[:, None]

    def unpack(self, dtype=torch.float32): return self.unpack_rows(0, self.out_features, dtype)


    def _capture_grad(self, start, grad):
        if self._grad is None: self._grad = torch.zeros(self.out_features, self.in_features, dtype=torch.float32, device=grad.device)
        self._grad[start:start + grad.shape[0]].add_(grad.float()); return grad

    def forward(self, x):
        ext = _native_rqt()
        if ext is not None and ext is not False and self.bits in (FP4, FP6, FP8) and x.device.type == "cpu" and x.dtype == torch.float32 and x.shape[-1] == self.in_features:
            grad_anchor = torch.ones((), dtype=x.dtype, device=x.device, requires_grad=True)
            out = _RQTLinearFunction.apply(x, self.packed, self.scale, self.in_features, self.out_features, self.bits, self, grad_anchor)
            return out if self.bias is None else out + self.bias
        grad_anchor = torch.ones((), dtype=x.dtype, device=x.device, requires_grad=True)
        out = _PackedTorchLinearFunction.apply(x, self.packed, self.scale, self.in_features, self.out_features, self.bits, self, grad_anchor)
        return out if self.bias is None else out + self.bias

    @torch.no_grad()
    def step(self, update, decay):
        if update.shape != (self.out_features, self.in_features): raise ValueError("invalid RQT update shape")
        ext = _native_rqt()
        if ext is not None and ext is not False and self.bits in (FP4, FP6, FP8) and self.packed.device.type == "cpu" and update.dtype == torch.float32 and update.is_contiguous() and self.packed.is_contiguous() and self.scale.is_contiguous():
            ext.rqt_requant_step(self.packed, self.scale, update, self.in_features, self.out_features, self.bits, decay)
            self._grad = None
            return
        for start in range(0, self.out_features, _BLOCK_ROWS):
            end = min(start + _BLOCK_ROWS, self.out_features)
            weight = self.unpack_rows(start, end, torch.float32)
            if decay: weight.mul_(1 - decay)
            weight.sub_(update[start:end])
            packed, scale = _encode(weight, self.bits)
            packed_slice = self.packed[self._row_slice(start, end)]
            if packed_slice.shape != packed.shape:
                raise RuntimeError("invalid packed RQT row shape")
            packed_slice.copy_(packed)
            if self.bits != FP8:
                self.scale[start:end].copy_(scale)
        self._grad = None

    def zero_grad(self): self._grad = None


class RQTLion:
    def __init__(self, model, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01, state_dtype=torch.float32):
        if state_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("RQT optimizer state dtype must be float32, float16, or bfloat16")
        self.model, self.lr, self.betas, self.weight_decay = model, lr, betas, weight_decay
        self.state_dtype = state_dtype
        self.rqt_state, self.param_state = {}, {}
        self.params = [p for p in model.parameters() if p.requires_grad]; self.param_names = {id(p): name for name, p in model.named_parameters() if p.requires_grad}

    def _modules(self): return [(name, module) for name, module in self.model.named_modules() if isinstance(module, RQTLinear) and module.trainable]

    def zero_grad(self, set_to_none=True):
        for _, module in self._modules(): module.zero_grad()
        for param in self.params: param.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm):
        grads = [m._grad for _, m in self._modules() if m._grad is not None] + [p.grad.float() for p in self.params if p.grad is not None]
        if not grads: return torch.tensor(0.0)
        total = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for _, module in self._modules():
                if module._grad is not None: module._grad.mul_(scale)
            for param in self.params:
                if param.grad is not None: param.grad.mul_(scale)
        return total

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for _, module in self._modules():
            if module._grad is None: continue
            grad = module._grad; avg = self.rqt_state.setdefault(module, torch.zeros_like(grad, dtype=self.state_dtype))
            ext = _native_rqt()
            fused = (ext is not None and ext is not False and module.bits in (FP4, FP6, FP8) and
                     module.packed.device.type == "cpu" and grad.dtype == torch.float32 and avg.dtype == torch.float32 and
                     grad.is_contiguous() and avg.is_contiguous() and module.packed.is_contiguous() and module.scale.is_contiguous())
            if fused:
                ext.rqt_lion_step(module.packed, module.scale, grad, avg, module.in_features, module.out_features,
                                  module.bits, self.lr, b1, b2, self.lr * self.weight_decay)
                module._grad = None
            else:
                avg.mul_(b1).add_(grad, alpha=1 - b1); update = avg.sign().mul(self.lr); avg.mul_(b2).add_(grad, alpha=1 - b2)
                module.step(update, self.lr * self.weight_decay)
        for param in self.params:
            if param.grad is None: continue
            grad = param.grad.float(); avg = self.param_state.setdefault(param, torch.zeros_like(param, dtype=self.state_dtype))
            avg.mul_(b1).add_(grad, alpha=1 - b1); update = avg.sign(); avg.mul_(b2).add_(grad, alpha=1 - b2)
            if self.weight_decay: param.mul_(1 - self.lr * self.weight_decay)
            param.add_(update, alpha=-self.lr)

    def state_dict(self):
        modules = {name: state.cpu() for name, module in self._modules() if (state := self.rqt_state.get(module)) is not None}; params = {self.param_names[id(param)]: state.cpu() for param, state in self.param_state.items() if id(param) in self.param_names}
        return {"version": 5, "lr": self.lr, "betas": self.betas, "weight_decay": self.weight_decay,
                "state_dtype": str(self.state_dtype).split(".")[-1], "param_state": params, "rqt_state": modules}

    def load_state_dict(self, state):
        self.lr = float(state.get("lr", self.lr)); self.betas = tuple(state.get("betas", self.betas)); self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        dtype_name = state.get("state_dtype")
        if dtype_name is not None:
            self.state_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype_name, self.state_dtype)
        modules = dict(self._modules()); named_params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        rqt_state, param_state = state.get("rqt_state", {}), state.get("param_state", {})
        if all(key.isdigit() for key in rqt_state) and rqt_state: rqt_state = {name: value for (name, _), value in zip(modules.items(), rqt_state.values())}
        if all(key.isdigit() for key in param_state) and param_state: param_state = {name: value for (name, _), value in zip(named_params.items(), param_state.values())}
        self.rqt_state = {}
        for name, value in rqt_state.items():
            if name not in modules: raise ValueError(f"unknown RQT optimizer module: {name}")
            module = modules[name]
            if tuple(value.shape) != (module.out_features, module.in_features): raise ValueError(f"invalid RQT optimizer state shape for {name}")
            self.rqt_state[module] = value.to(module.packed.device, dtype=self.state_dtype)
        self.param_state = {}
        for name, value in param_state.items():
            if name not in named_params: raise ValueError(f"unknown optimizer parameter: {name}")
            param = named_params[name]
            if tuple(value.shape) != tuple(param.shape): raise ValueError(f"invalid optimizer state shape for {name}")
            self.param_state[param] = value.to(param.device, dtype=self.state_dtype)


class FP8SGD:
    def __init__(self, model, lr=1e-4, weight_decay=0.0):
        if lr <= 0: raise ValueError("lr must be > 0")
        self.model, self.lr, self.weight_decay = model, lr, weight_decay
        self.params = [p for p in model.parameters() if p.requires_grad]

    def _modules(self): return [(name, module) for name, module in self.model.named_modules() if isinstance(module, RQTLinear) and module.trainable]

    def zero_grad(self, set_to_none=True):
        for _, module in self._modules(): module.zero_grad()
        for param in self.params: param.grad = None

    @torch.no_grad()
    def clip_grad_norm(self, max_norm):
        grads = [m._grad for _, m in self._modules() if m._grad is not None] + [p.grad.float() for p in self.params if p.grad is not None]
        if not grads: return torch.tensor(0.0)
        total = torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt()
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for _, module in self._modules():
                if module._grad is not None: module._grad.mul_(scale)
            for param in self.params:
                if param.grad is not None: param.grad.mul_(scale)
        return total

    @torch.no_grad()
    def step(self):
        ext = _native_rqt()
        modules = self._modules()
        invalid = [name for name, module in modules if module.bits != FP8]
        if invalid:
            raise RuntimeError(f"FP8SGD found non-FP8 RQT layers: {', '.join(invalid[:8])}")
        for _, module in modules:
            if module._grad is None: continue
            grad = module._grad.float().contiguous()
            if ext is not None and ext is not False and module.packed.device.type == "cpu" and grad.is_contiguous():
                ext.fp8_sgd_step(module.packed, grad, module.in_features, module.out_features, self.lr, self.lr * self.weight_decay)
            else:
                weight = module.unpack(torch.float32)
                weight.mul_(1 - self.lr * self.weight_decay).sub_(grad, alpha=self.lr)
                module.packed.copy_(weight.to(torch.float8_e4m3fn).reshape(-1))
            module._grad = None
        for param in self.params:
            if param.grad is None: continue
            if isinstance(param, torch.Tensor) and param.dtype.is_floating_point:
                if self.weight_decay: param.mul_(1 - self.lr * self.weight_decay)
                param.add_(param.grad.float(), alpha=-self.lr)
            param.grad = None

    def state_dict(self):
        return {"version": 2, "type": "fp8_sgd", "lr": self.lr, "weight_decay": self.weight_decay}

    def load_state_dict(self, state):
        self.lr = float(state.get("lr", self.lr)); self.weight_decay = float(state.get("weight_decay", self.weight_decay))


def _replace(root, name, bits):
    parts = name.split("."); parent = root
    for part in parts[:-1]: parent = getattr(parent, part)
    current = getattr(parent, parts[-1])
    if isinstance(current, RQTLinear):
        if current.bits == bits: return
        weight = current.unpack(); current.bits = bits; current.bit_width.fill_(bits); current._replace_weight(weight); return
    setattr(parent, parts[-1], RQTLinear(current, bits))


def prepare_rqt(model, bits=FP6):
    bits = _bits(bits); root = getattr(model, "_orig_mod", model); targets = [(name, module) for name, module in root.named_modules() if isinstance(module, (nn.Linear, RQTLinear))]
    for name, _ in reversed(targets): _replace(root, name, bits)
    model.cfg.rqt_bits = bits; print(f"[RQT] FP{bits} packed compute | {len(targets)} linear layers | block_rows={_BLOCK_ROWS}"); return len(targets)


def prepare_fp8(model):
    root = getattr(model, "_orig_mod", model)
    targets = [(name, module) for name, module in root.named_modules() if isinstance(module, (nn.Linear, RQTLinear))]
    for name, _ in reversed(targets): _replace(root, name, FP8)
    model.cfg.rqt_bits = FP8
    model.cfg.fp8_training = True
    print(f"[FP8] E4M3FN packed weights | {len(targets)} linear layers | FP32 accumulation/gradients")
    return len(targets)


def prepare_mixed_rqt(model):
    root = getattr(model, "_orig_mod", model); targets = [(name, module) for name, module in root.named_modules() if isinstance(module, (nn.Linear, RQTLinear))]
    for name, _ in reversed(targets):
        bits = FP8 if name == "head" or ".att." in f".{name}." else FP4 if ".ffn." in f".{name}." else FP6; _replace(root, name, bits)
    model.cfg.rqt_mixed = True; print(f"[RQT] mixed FP4/FP6/FP8 packed compute where supported | {len(targets)} linear layers | block_rows={_BLOCK_ROWS}"); return len(targets)


def load_rqt_checkpoint(in_dir):
    from pathlib import Path
    from safetensors.torch import load_file
    from rwkv_x_core import RWKVXConfig, RWKVXModel
    in_dir = Path(in_dir); cfg = RWKVXConfig.load(in_dir / "config.json"); sd = load_file(str(in_dir / "model.safetensors")); model = RWKVXModel(cfg)
    packed = [k[:-7] for k in sd if k.endswith(".packed")]
    if not packed: raise RuntimeError("RQT checkpoint has no packed weights")
    for path in packed:
        parent_path, name = path.rsplit(".", 1) if "." in path else ("", path); parent = model.get_submodule(parent_path) if parent_path else model; linear = getattr(parent, name)
        bits = _bits(int(sd[path + ".bit_width"].item())) if path + ".bit_width" in sd else FP8 if sd[path + ".packed"].dtype == torch.float8_e4m3fn else _bits(cfg.rqt_bits)
        setattr(parent, name, RQTLinear(linear, bits))
    missing, unexpected = model.load_state_dict(sd, strict=False); missing = [key for key in missing if not key.endswith(".bit_width")]
    if missing or unexpected: raise RuntimeError(f"invalid RQT checkpoint: missing={missing}, unexpected={unexpected}")
    return model


def _install_checkpoint_loader():
    from pathlib import Path
    from safetensors.torch import load_file
    from rwkv_x_core import RWKVXModel
    original = RWKVXModel.from_pretrained.__func__
    @classmethod
    def from_pretrained(cls, in_dir):
        path = Path(in_dir); sd = load_file(str(path / "model.safetensors"))
        if any(key.endswith(".packed") for key in sd): return load_rqt_checkpoint(path)
        return original(cls, in_dir)
    RWKVXModel.from_pretrained = from_pretrained


_install_checkpoint_loader()

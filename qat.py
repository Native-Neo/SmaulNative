#!/usr/bin/env python3
# qat.py -- Quantization-Aware Training for RWKV-X, int3.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver

from rwkv_x_core import RWKVXModel, RWKV_CMix_x070, RWKV_CMix_MoE

_CMIX_LINEAR_NAMES = ("key", "value")
WBITS = 3
NUM_LEVELS = 2 ** WBITS
WEIGHT_QMIN, WEIGHT_QMAX = -(NUM_LEVELS // 2), NUM_LEVELS // 2 - 1
ACT_QMIN, ACT_QMAX = 0, NUM_LEVELS - 1


def _weight_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=WEIGHT_QMIN, quant_max=WEIGHT_QMAX, dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)()


def _activation_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=ACT_QMIN, quant_max=ACT_QMAX, dtype=torch.quint8, qscheme=torch.per_tensor_affine)()


def _pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes (values 0-7) into 3-bit packed bytes, entirely device-agnostic."""
    device = codes.device
    shifts = torch.tensor([2, 1, 0], dtype=torch.uint8, device=device)
    bits = (codes.reshape(-1).to(torch.uint8).unsqueeze(-1) >> shifts) & 1
    flat_bits = bits.reshape(-1)
    pad_len = (8 - (flat_bits.numel() % 8)) % 8
    if pad_len > 0:
        flat_bits = torch.cat([flat_bits, torch.zeros(pad_len, dtype=torch.uint8, device=device)])
    bit_groups = flat_bits.reshape(-1, 8)
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=device)
    return (bit_groups * powers).sum(dim=-1)


def _unpack_3bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Unpack 3-bit packed bytes into uint8 codes, entirely device-agnostic."""
    device = packed.device
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=device)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    flat_bits = bits.reshape(-1)[:numel * 3]
    grouped = flat_bits.reshape(numel, 3)
    return grouped[:, 0] * 4 + grouped[:, 1] * 2 + grouped[:, 2]


class QATLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None, "RWKV-X Channel-Mix linears are all bias=False"
        self.weight = linear.weight
        self.weight_fq = _weight_fake_quant()
        self.act_fq = _activation_fake_quant()
        self.fake_quant_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight)
        return F.linear(self.act_fq(x), self.weight_fq(self.weight))

    def to_quantized(self) -> "QuantizedLinear":
        scale, _zero_point = self.weight_fq.calculate_qparams()
        w = self.weight.detach().float()
        q = torch.clamp(torch.round(w / scale.unsqueeze(1)), WEIGHT_QMIN, WEIGHT_QMAX)
        codes = (q - WEIGHT_QMIN).to(torch.uint8)
        return QuantizedLinear(_pack_3bit(codes), scale.float(), w.shape)


class QuantizedLinear(nn.Module):
    """Converted post-QAT op with packed int3 weights.

    The unpacked uint8 codes are cached per device. Unpacking is entirely PyTorch-native
    (no NumPy), so it runs on whatever device the packed buffer lives on.
    """

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, shape: torch.Size):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.out_features, self.in_features = int(shape[0]), int(shape[1])
        self._code_cache = {}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
        self._code_cache.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device.type, x.device.index)
        codes = self._code_cache.get(key)
        if codes is None or codes.device != x.device:
            codes = _unpack_3bit(self.packed, self.out_features * self.in_features).to(x.device)
            self._code_cache[key] = codes
        q = codes.to(x.dtype).view(self.out_features, self.in_features) + WEIGHT_QMIN
        w = q * self.scale.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        return F.linear(x, w)


def _iter_cmix_modules(model: RWKVXModel):
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


def prepare_qat(model: RWKVXModel) -> int:
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, nn.Linear):
                setattr(cmix, name, QATLinear(mod))
                n += 1
    return n


def convert_qat(model: RWKVXModel) -> int:
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, QATLinear):
                setattr(cmix, name, mod.to_quantized())
                n += 1
    return n


@torch.no_grad()
def calibrate(model: RWKVXModel, tokenizer, calib_texts, ctx_len: int, device, max_batches: int = 64):
    was_training = model.training
    model.eval()
    buf, n_batches = [], 0
    for text in calib_texts:
        if n_batches >= max_batches:
            break
        buf.extend(tokenizer.encode(text))
        buf.append(tokenizer.eos_token_id)
        while len(buf) >= ctx_len + 1 and n_batches < max_batches:
            chunk = buf[:ctx_len + 1]
            del buf[:ctx_len]
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            model(x)
            n_batches += 1
            print(f"[QAT] calib batch {n_batches}/{max_batches}")
    model.train(was_training)
    return n_batches

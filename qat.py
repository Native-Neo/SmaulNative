#!/usr/bin/env python3
# qat.py -- Quantization-Aware Training for RWKV-X, at 3-bit (int3) precision.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

from rwkv_x_core import RWKVXModel, RWKV_CMix_x070, RWKV_CMix_MoE

# The two Linear attributes inside every RWKV_CMix_x070 -- the FFN's expand (key) and
# contract (value) projections. These are what get replaced/restored.
_CMIX_LINEAR_NAMES = ("key", "value")

WBITS = 3
NUM_LEVELS = 2 ** WBITS          # 8
WEIGHT_QMIN, WEIGHT_QMAX = -(NUM_LEVELS // 2), NUM_LEVELS // 2 - 1   # -4, 3 (symmetric signed)
ACT_QMIN, ACT_QMAX = 0, NUM_LEVELS - 1                               # 0, 7 (asymmetric unsigned)


def _weight_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAveragePerChannelMinMaxObserver,
        quant_min=WEIGHT_QMIN, quant_max=WEIGHT_QMAX, dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric, ch_axis=0,
    )()


def _activation_fake_quant() -> FakeQuantize:
    return FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=ACT_QMIN, quant_max=ACT_QMAX, dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )()


# Sub-byte packing: 8 int3 codes (values 0..7) <-> 3 bytes, MSB-first, no cross-tensor padding
# except a partial final group when numel isn't a multiple of 8.

def _pack_3bit(codes: torch.Tensor) -> torch.Tensor:
    """codes: any-shape uint8 tensor, values in [0, 7]. Returns a 1D uint8 buffer of packed bits."""
    flat = codes.reshape(-1).to(torch.uint8).cpu().numpy()
    bits = np.unpackbits(flat[:, None], axis=1, bitorder="big")[:, 5:8]  # low 3 bits of each byte
    packed = np.packbits(bits.reshape(-1))
    return torch.from_numpy(packed.copy())


def _unpack_3bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of _pack_3bit. Returns a 1D uint8 tensor of `numel` codes in [0, 7]."""
    bits = np.unpackbits(packed.cpu().numpy())[: numel * 3].reshape(numel, 3)
    codes = bits[:, 0] * 4 + bits[:, 1] * 2 + bits[:, 2]
    return torch.from_numpy(codes.astype(np.uint8).copy())


class QATLinear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear that fake-quantizes its input activation
    (per-tensor asymmetric int3) and its weight (per-channel symmetric int3) on every forward
    pass."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None, "RWKV-X Channel-Mix linears are all bias=False"
        self.weight = linear.weight  # keep the same nn.Parameter -> state_dict key stays "weight"
        self.weight_fq = _weight_fake_quant()
        self.act_fq = _activation_fake_quant()
        self.fake_quant_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return F.linear(x, self.weight)
        return F.linear(self.act_fq(x), self.weight_fq(self.weight))

    def to_quantized(self) -> "QuantizedLinear":
        """Bake the calibrated weight fake-quant range into real, packed int3 weights."""
        scale, _zero_point = self.weight_fq.calculate_qparams()  # per-channel, symmetric -> zp is 0
        w = self.weight.detach().float()
        q = torch.clamp(torch.round(w / scale.unsqueeze(1)), WEIGHT_QMIN, WEIGHT_QMAX)
        codes = (q - WEIGHT_QMIN).to(torch.uint8)  # shift to unsigned [0, 7] for packing
        return QuantizedLinear(_pack_3bit(codes), scale.float(), w.shape)


class QuantizedLinear(nn.Module):
    """The converted (post-QAT) op: packed int3 weights (8 codes / 3 bytes), unpacked and
    dequantized on the fly each forward. CPU-portable (no fbgemm/qnnpack dependency); weight
    storage is ~2.67x smaller than int8 and ~10.7x smaller than FP32. Unpacking costs some
    forward-pass time -- fine for this CPU-first experimentation project, not meant to compete
    with a packed-kernel inference engine."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, shape: torch.Size):
        super().__init__()
        self.register_buffer("packed", packed)
        self.register_buffer("scale", scale)
        self.out_features, self.in_features = int(shape[0]), int(shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        numel = self.out_features * self.in_features
        codes = _unpack_3bit(self.packed, numel).to(x.device)
        q = codes.to(x.dtype).view(self.out_features, self.in_features) + WEIGHT_QMIN
        w = q * self.scale.to(x.dtype).unsqueeze(1)
        return F.linear(x, w)


def _iter_cmix_modules(model: RWKVXModel):
    """Every RWKV_CMix_x070 instance in the model -- the dense FFN, or each MoE expert."""
    for blk in list(model.rwkv_blocks) + list(model.moba_blocks):
        ffn = blk.ffn
        if isinstance(ffn, RWKV_CMix_MoE):
            yield from ffn.experts
        elif isinstance(ffn, RWKV_CMix_x070):
            yield ffn


def prepare_qat(model: RWKVXModel) -> int:
    """In-place. Replaces each Channel-Mix `key`/`value` nn.Linear with QATLinear. Returns how
    many were replaced. Safe to call whether the model was just constructed or already has
    weights loaded (from_pretrained) -- it wraps whatever nn.Linear is currently there, so loaded
    weights are preserved."""
    n = 0
    for cmix in _iter_cmix_modules(model):
        for name in _CMIX_LINEAR_NAMES:
            mod = getattr(cmix, name)
            if isinstance(mod, nn.Linear):
                setattr(cmix, name, QATLinear(mod))
                n += 1
    return n


def convert_qat(model: RWKVXModel) -> int:
    """In-place. Replaces each QATLinear with a converted QuantizedLinear (real packed int3
    weights). Call this after QAT fine-tuning is done. Returns how many were converted."""
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
    """Runs a handful of forward passes over a representative slice of data (calib_texts: an
    iterable of raw strings, e.g. the first N lines pulled from the training set) so the
    activation/weight observers' moving min/max ranges settle before QAT fine-tuning starts.
    No gradients, no optimizer step -- pure calibration."""
    was_training = model.training
    model.eval()
    buf, n_batches = [], 0
    for text in calib_texts:
        if n_batches >= max_batches:
            break
        buf.extend(tokenizer.encode(text))
        buf.append(tokenizer.eos_token_id)
        while len(buf) >= ctx_len + 1 and n_batches < max_batches:
            chunk = buf[: ctx_len + 1]
            del buf[:ctx_len]
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            model(x)
            n_batches += 1
    model.train(was_training)
    return n_batches

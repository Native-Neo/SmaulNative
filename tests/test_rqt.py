import torch

from rqt import RQTLinear, _pack, _unpack, _levels


def test_fp6_packing_roundtrip():
    codes = torch.arange(16, dtype=torch.uint8) % 64
    packed = _pack(codes, 6)
    assert packed.numel() == 12
    assert torch.equal(_unpack(packed, 6, codes.numel()), codes)


def test_fp4_packing_roundtrip():
    codes = torch.arange(16, dtype=torch.uint8) % 16
    assert torch.equal(_unpack(_pack(codes, 4), 4, codes.numel()), codes)


def test_rqt_linear_updates_packed_weight():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    before = linear.packed.clone()
    x = torch.randn(2, 8)
    y = linear(x).sum()
    y.backward()
    assert linear._grad is not None
    linear.step(linear._grad * 1e-4, 0.0)
    assert linear.packed.numel() == before.numel()
    assert torch.isfinite(linear.unpack()).all()


def test_fp6_levels_are_finite():
    levels = _levels(6, torch.device("cpu"), torch.float32)
    assert levels.numel() == 64
    assert torch.isfinite(levels).all()

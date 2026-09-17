import torch

from mixed_precision import MixedPrecisionLinear, quantize


def test_fp4_fp6_are_finite():
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    for bits in (4, 6):
        q = quantize(x, bits)
        assert torch.isfinite(q).all()
        assert q.shape == x.shape


def test_fp8_gradient_quantization():
    x = torch.randn(8, 16, requires_grad=True)
    q = quantize(x, 8)
    q.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_linear_uses_fp32_accumulation():
    linear = MixedPrecisionLinear(torch.nn.Linear(8, 4, bias=False), 4, 6)
    x = torch.randn(2, 8, requires_grad=True)
    y = linear(x)
    assert y.dtype == torch.float32
    y.sum().backward()
    assert linear.weight.grad is not None

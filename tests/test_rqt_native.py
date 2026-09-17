import torch
import torch.nn as nn

from rqt import FP4, FP6, RQTLinear


def _check(bits):
    torch.manual_seed(1234)
    base = nn.Linear(17, 11, bias=True)
    layer = RQTLinear(base, bits)
    x = torch.randn(5, 17, requires_grad=True)

    out = layer(x)
    ref = torch.nn.functional.linear(x, layer.unpack(), layer.bias)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    loss = out.square().mean()
    loss.backward()
    assert layer._grad is not None
    assert layer._grad.shape == (11, 17)
    assert torch.isfinite(layer._grad).all()

    old = layer.unpack().clone()
    update = torch.full_like(old, 1e-5)
    layer.step(update, 0.0)
    new = layer.unpack()
    assert not torch.equal(old, new)


def test_fp4_native_rqt_linear():
    _check(FP4)


def test_fp6_native_rqt_linear():
    _check(FP6)

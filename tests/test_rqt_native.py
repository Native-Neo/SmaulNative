import torch
import torch.nn as nn

from rqt import FP4, FP6, RQTLinear


def _check(bits):
    torch.manual_seed(1234)
    base = nn.Linear(17, 11, bias=True)
    layer = RQTLinear(base, bits)

    x = torch.randn(2, 3, 17, requires_grad=True)
    out = layer(x)
    weight = layer.unpack().detach().requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True)
    ref = torch.nn.functional.linear(ref_x, weight, ref_bias)
    torch.testing.assert_close(out, ref.detach(), rtol=1e-5, atol=1e-5)

    grad_output = torch.randn_like(out)
    native_x = x.detach().clone().requires_grad_(True)
    layer.zero_grad()
    layer(native_x).backward(grad_output)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_weight = layer.unpack().detach().requires_grad_(True)
    ref_bias = layer.bias.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(ref_x, ref_weight, ref_bias).backward(grad_output)

    torch.testing.assert_close(native_x.grad, ref_x.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer._grad, ref_weight.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(layer.bias.grad, ref_bias.grad, rtol=1e-5, atol=1e-5)

    old = layer.unpack().clone()
    update = torch.full_like(old, 0.1)
    layer.step(update, 0.0)
    new = layer.unpack()
    assert not torch.equal(old, new)


def test_fp4_native_rqt_linear():
    _check(FP4)


def test_fp6_native_rqt_linear():
    _check(FP6)

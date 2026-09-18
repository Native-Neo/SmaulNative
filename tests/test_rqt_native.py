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

def test_fused_rqt_lion_matches_reference():
    from rqt import _native_rqt

    for bits in (FP4, FP6):
        torch.manual_seed(77 + bits)
        base = nn.Linear(17, 11, bias=False)
        fused = RQTLinear(base, bits)
        reference = RQTLinear(nn.Linear(17, 11, bias=False), bits)
        reference.packed.copy_(fused.packed)
        reference.scale.copy_(fused.scale)

        grad = torch.randn(11, 17)
        avg_fused = torch.randn_like(grad)
        avg_reference = avg_fused.clone()
        lr, b1, b2, decay = 1e-3, 0.9, 0.99, 2e-5

        ext = _native_rqt()
        assert ext is not None and ext is not False
        ext.rqt_lion_step(fused.packed, fused.scale, grad, avg_fused, 17, 11, bits, lr, b1, b2, decay)

        avg_reference.mul_(b1).add_(grad, alpha=1 - b1)
        update = avg_reference.sign().mul(lr)
        avg_reference.mul_(b2).add_(grad, alpha=1 - b2)
        reference.step(update, decay)

        torch.testing.assert_close(avg_fused, avg_reference, rtol=0, atol=0)
        torch.testing.assert_close(fused.unpack(), reference.unpack(), rtol=0, atol=0)

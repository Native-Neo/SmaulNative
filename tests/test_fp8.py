import torch
import torch.nn as nn

from rqt import FP8, FP8SGD, RQTLinear, _native_rqt


def test_fp8_linear_forward_backward():
    linear = nn.Linear(8, 4, bias=True)
    module = RQTLinear(linear, FP8)
    x = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    out = module(x)
    assert out.shape == (3, 4)
    assert out.dtype == torch.float32
    assert module.packed.dtype == torch.float8_e4m3fn
    loss = out.square().mean()
    loss.backward()
    assert module._grad is not None
    assert module._grad.shape == (4, 8)
    assert torch.isfinite(module._grad).all()
    assert torch.isfinite(x.grad).all()


def test_fp8_sgd_updates_packed_weights():
    linear = nn.Linear(8, 4, bias=False)
    module = RQTLinear(linear, FP8)
    model = nn.Module()
    model.add_module("linear", module)
    x = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    before = module.packed.clone()
    module(x).sum().backward()
    optimizer = FP8SGD(model, lr=0.01)
    optimizer.step()
    assert not torch.equal(before, module.packed)
    assert module._grad is None
    assert torch.isfinite(module.packed.float()).all()


def test_fp8_native_extension_if_available():
    ext = _native_rqt()
    if ext is False:
        return
    assert hasattr(ext, "fp8_sgd_step")
    assert hasattr(ext, "rqt_linear_forward")

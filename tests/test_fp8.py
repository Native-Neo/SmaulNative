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


def test_fp8_config_metadata_roundtrip():
    from rwkv_x_core import RWKVXConfig

    cfg = RWKVXConfig(tokenizer_sha256="abc", dataset_fingerprint="dataset")
    assert RWKVXConfig.load(_write_config(cfg)).tokenizer_sha256 == "abc"


def _write_config(cfg):
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "config.json"
    cfg.save(path)
    return path


def test_fp8_checkpoint_roundtrip(tmp_path):
    from rwkv_x_core import RWKVXConfig, RWKVXModel
    from rqt import prepare_fp8

    cfg = RWKVXConfig(vocab_size=32, n_embd=16, n_layer=2, head_size=4, n_moba_layer=1,
                      checkpoint_ffn=False, tokenizer_sha256="abc", dataset_fingerprint="dataset")
    model = RWKVXModel(cfg)
    prepare_fp8(model)
    model.save_pretrained(tmp_path, dtype="fp32", include_upstream=False)
    loaded = RWKVXModel.from_pretrained(tmp_path)
    assert loaded.cfg.fp8_training
    assert loaded.cfg.rqt_bits == 8
    assert loaded.cfg.tokenizer_sha256 == "abc"
    assert loaded.cfg.dataset_fingerprint == "dataset"
    original = {name: module.packed.detach().clone() for name, module in model.named_modules() if hasattr(module, "packed")}
    restored = {name: module.packed.detach().clone() for name, module in loaded.named_modules() if hasattr(module, "packed")}
    assert original.keys() == restored.keys()
    for name in original:
        assert torch.equal(original[name], restored[name])


def test_fp8_prepare_is_idempotent():
    from rwkv_x_core import RWKVXConfig, RWKVXModel
    from rqt import prepare_fp8

    model = RWKVXModel(RWKVXConfig(vocab_size=32, n_embd=16, n_layer=2, head_size=4, n_moba_layer=1,
                                    checkpoint_ffn=False))
    prepare_fp8(model)
    before = {name: module.packed.detach().clone() for name, module in model.named_modules() if hasattr(module, "packed")}
    prepare_fp8(model)
    after = {name: module.packed.detach().clone() for name, module in model.named_modules() if hasattr(module, "packed")}
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name])


def test_fp8_native_sgd_handles_exponent_boundary():
    ext = _native_rqt()
    if ext is False:
        return
    packed = torch.tensor([2.0], dtype=torch.float8_e4m3fn)
    grad = torch.tensor([[1.0]], dtype=torch.float32)
    ext.fp8_sgd_step(packed, grad, 1, 1, 0.01, 0.0)
    value = packed.float().item()
    assert 1.875 <= value <= 2.0
    assert torch.isfinite(packed.float()).all()



def test_fp8_native_sgd_preserves_e4m3_boundaries():
    ext = _native_rqt()
    if ext is False:
        return
    boundaries = [
        (0.25, 0.234375, 0.25),
        (0.5, 0.46875, 0.5),
        (1.0, 0.9375, 1.0),
        (2.0, 1.875, 2.0),
        (4.0, 3.75, 4.0),
        (8.0, 7.5, 8.0),
    ]
    for value, lower, upper in boundaries:
        packed = torch.tensor([value], dtype=torch.float8_e4m3fn)
        grad = torch.tensor([[0.01]], dtype=torch.float32)
        ext.fp8_sgd_step(packed, grad, 1, 1, 1.0, 0.0)
        updated = packed.float().item()
        assert lower <= updated <= upper
        assert torch.isfinite(packed.float()).all()


def test_fp8_native_sgd_saturates_without_nan():
    ext = _native_rqt()
    if ext is False:
        return
    packed = torch.tensor([448.0, -448.0], dtype=torch.float8_e4m3fn)
    grad = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    ext.fp8_sgd_step(packed, grad, 2, 1, 100.0, 0.0)
    assert torch.isfinite(packed.float()).all()
    assert packed.float().abs().max().item() <= 448.0

def test_fp8_native_forward_and_backward_input_match_reference():
    ext = _native_rqt()
    if ext is False:
        return
    linear = nn.Linear(8, 4, bias=False)
    module = RQTLinear(linear, FP8)
    x = torch.randn(5, 8, dtype=torch.float32)
    grad = torch.randn(5, 4, dtype=torch.float32)
    weight = module.packed.float()

    native_out = ext.rqt_linear_forward(x, module.packed, module.scale, 8, 4, FP8)
    native_grad_x = ext.rqt_linear_backward_input(grad, module.packed, module.scale, 8, 4, FP8)
    reference_out = x @ weight.t()
    reference_grad_x = grad @ weight

    assert torch.allclose(native_out, reference_out, atol=0.0, rtol=0.0)
    assert torch.allclose(native_grad_x, reference_grad_x, atol=0.0, rtol=0.0)

def test_frozen_fp8_backward_does_not_capture_weight_grad():
    linear = nn.Linear(8, 4, bias=False)
    module = RQTLinear(linear, FP8)
    module.trainable = False
    x = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    out = module(x)
    out.square().mean().backward()
    assert module._grad is None
    assert torch.isfinite(x.grad).all()

def test_fp8_backward_without_input_grad_still_captures_weight_grad():
    linear = nn.Linear(8, 4, bias=False)
    module = RQTLinear(linear, FP8)
    x = torch.randn(3, 8, dtype=torch.float32)
    module(x).square().mean().backward()
    assert module._grad is not None
    assert module._grad.shape == (4, 8)

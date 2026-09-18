import torch
import torch.nn as nn

from rqt import RQTLinear, RQTLion, _pack, _unpack, _levels, prepare_mixed_rqt, prepare_rqt


def test_fp6_packing_roundtrip():
    codes = torch.arange(16, dtype=torch.uint8) % 64
    packed = _pack(codes, 6)
    assert packed.numel() == 12
    assert torch.equal(_unpack(packed, 6, codes.numel()), codes)


def test_fp6_packing_handles_partial_group():
    codes = torch.tensor([1, 7, 18, 31, 42], dtype=torch.uint8)
    packed = _pack(codes, 6)
    assert packed.numel() == 6
    assert torch.equal(_unpack(packed, 6, codes.numel()), codes)


def test_fp4_packing_roundtrip():
    codes = torch.arange(16, dtype=torch.uint8) % 16
    assert torch.equal(_unpack(_pack(codes, 4), 4, codes.numel()), codes)


def test_fp4_packing_handles_odd_count():
    codes = torch.tensor([0, 3, 8, 15, 4], dtype=torch.uint8)
    packed = _pack(codes, 4)
    assert packed.numel() == 3
    assert torch.equal(_unpack(packed, 4, codes.numel()), codes)


def test_fp8_storage_roundtrip():
    weight = torch.tensor([[-1.0, -0.5, 0.5, 1.0]], dtype=torch.float32)
    linear = RQTLinear(torch.nn.Linear(4, 1, bias=False), 8)
    linear._replace_weight(weight)
    assert linear.packed.dtype == torch.float8_e4m3fn
    assert torch.isfinite(linear.unpack()).all()
    assert torch.allclose(linear.unpack(), weight, atol=0.125)


def test_rqt_linear_updates_packed_weight():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    before = linear.packed.clone()
    linear(torch.randn(2, 8)).sum().backward()
    assert linear._grad is not None
    linear.step(linear._grad * 1e-4, 0.0)
    assert linear.packed.numel() == before.numel()
    assert torch.isfinite(linear.unpack()).all()


def test_rqt_lion_steps_without_master_weight():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    opt = RQTLion(linear, lr=1e-3, weight_decay=0.0)
    assert not any(name == "weight" for name, _ in linear.named_parameters())
    packed = linear.packed.clone()
    linear(torch.randn(2, 8)).sum().backward()
    opt.step()
    assert linear.packed.dtype == torch.uint8
    assert linear.packed.shape == packed.shape
    assert linear._grad is None


def test_rqt_lion_state_uses_stable_names():
    model = torch.nn.Sequential(
        RQTLinear(torch.nn.Linear(8, 4, bias=False), 6),
        RQTLinear(torch.nn.Linear(4, 2, bias=False), 6),
    )
    opt = RQTLion(model, lr=1e-3, weight_decay=0.0)
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    state = opt.state_dict()
    assert set(state["rqt_state"]) == {"0", "1"}
    assert state["version"] == 5
    assert state["lr"] == 1e-3
    assert state["betas"] == (0.9, 0.99)

    clone = torch.nn.Sequential(
        RQTLinear(torch.nn.Linear(8, 4, bias=False), 6),
        RQTLinear(torch.nn.Linear(4, 2, bias=False), 6),
    )
    clone_opt = RQTLion(clone, lr=9e-4, betas=(0.8, 0.95), weight_decay=0.1)
    clone_opt.load_state_dict(state)
    assert clone_opt.lr == 1e-3
    assert clone_opt.betas == (0.9, 0.99)
    assert clone_opt.weight_decay == 0.0
    assert len(clone_opt.rqt_state) == 2
    assert all(state.shape == module.unpack().shape for module, state in clone_opt.rqt_state.items())


def test_rqt_optimizer_supports_reduced_precision_state():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    opt = RQTLion(linear, lr=1e-3, weight_decay=0.0, state_dtype=torch.bfloat16)
    linear(torch.randn(2, 8)).sum().backward()
    opt.step()
    state = opt.state_dict()
    assert state["state_dtype"] == "bfloat16"
    assert next(iter(state["rqt_state"].values())).dtype == torch.bfloat16

    clone = RQTLion(RQTLinear(torch.nn.Linear(8, 4, bias=False), 6))
    clone.load_state_dict(state)
    assert clone.state_dtype is torch.bfloat16


def test_rqt_rejects_bad_optimizer_state_shape():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    opt = RQTLion(linear, lr=1e-3, weight_decay=0.0)
    state = {"rqt_state": {"": torch.zeros(1)}, "param_state": {}}
    try:
        opt.load_state_dict(state)
    except ValueError as exc:
        assert "invalid RQT optimizer state shape" in str(exc)
    else:
        raise AssertionError("bad optimizer state was accepted")


def test_rqt_gradient_matches_quantized_weight_shape():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 4)
    x = torch.randn(3, 8)
    linear(x).square().mean().backward()
    assert linear._grad.shape == (4, 8)
    assert linear._grad.dtype == torch.float32
    assert not any(name == "weight" for name, _ in linear.named_parameters())


def test_rqt_forward_matches_input_dtype():
    for dtype in (torch.float16, torch.bfloat16):
        linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
        output = linear(torch.randn(2, 8, dtype=dtype))
        assert output.dtype == dtype


def test_rqt_shared_module_accumulates_weight_gradients():
    linear = RQTLinear(torch.nn.Linear(4, 3, bias=False), 6)
    x1, x2 = torch.randn(2, 4), torch.randn(2, 4)
    (linear(x1) + linear(x2)).sum().backward()
    expected = torch.ones(2, 3).t() @ (x1 + x2)
    torch.testing.assert_close(linear._grad, expected, rtol=1e-5, atol=1e-5)


def test_rqt_weight_is_requantized_after_every_step():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    opt = RQTLion(linear, lr=0.1, weight_decay=0.0)
    packed_before = linear.packed.clone()
    linear(torch.ones(2, 8)).sum().backward()
    opt.step()
    assert linear._grad is None
    assert linear.packed.dtype == torch.uint8
    assert linear.packed.shape == packed_before.shape
    assert torch.isfinite(linear.unpack()).all()


def test_rqt_optimizer_rejects_unknown_module():
    linear = RQTLinear(torch.nn.Linear(8, 4, bias=False), 6)
    opt = RQTLion(linear)
    try:
        opt.load_state_dict({"rqt_state": {"missing": torch.zeros(4, 8)}, "param_state": {}})
    except ValueError as exc:
        assert "unknown RQT optimizer module" in str(exc)
    else:
        raise AssertionError("unknown RQT module was accepted")


def test_prepare_rqt_replaces_all_linear_layers():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8)
            self.b = nn.Sequential(nn.Linear(8, 4), nn.ReLU())
            self.cfg = type("Cfg", (), {})()

    model = Model()
    assert prepare_rqt(model, 6) == 2
    assert isinstance(model.a, RQTLinear)
    assert isinstance(model.b[0], RQTLinear)
    assert model.cfg.rqt_bits == 6


def test_prepare_mixed_rqt_assigns_layer_precisions():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.att = nn.Linear(8, 8)
            self.ffn = nn.Linear(8, 8)
            self.other = nn.Linear(8, 8)
            self.head = nn.Linear(8, 4)
            self.cfg = type("Cfg", (), {})()

    model = Model()
    assert prepare_mixed_rqt(model) == 4
    assert model.att.bits == 8
    assert model.ffn.bits == 4
    assert model.other.bits == 6
    assert model.head.bits == 8
    assert model.cfg.rqt_mixed is True


def test_fp4_and_fp6_levels_are_finite_and_distinct():
    fp4 = _levels(4, torch.device("cpu"), torch.float32)
    fp6 = _levels(6, torch.device("cpu"), torch.float32)
    assert fp4.numel() == 16
    assert fp6.numel() == 64
    assert torch.isfinite(fp4).all()
    assert torch.isfinite(fp6).all()
    assert fp4.abs().max() == 6
    assert fp6.abs().max() == 28

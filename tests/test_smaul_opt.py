import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pytest
import torch

from kernel.fp8_tile import fp8_modules
from smaul_linear import LinearConfig, SmaulLinear
from train import Lion, SmaulOpt, _load_optimizer, _save_optimizer


def _dummy_model():
    return torch.nn.Module()


# ----------------------------------------------------------------------
# v2: reduced-precision state storage (FP32 arithmetic, narrower buffers)
# ----------------------------------------------------------------------
_STATE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


def test_state_dtype_defaults_to_bf16():
    """bf16 is the default: 2 bytes/element of state, FP32 math."""
    p = torch.nn.Parameter(torch.randn(4))
    opt = SmaulOpt([p])
    assert opt.state_dtype == "bf16"
    assert "bf16" in SmaulOpt._STATE_DTYPES
    # The removed 1-byte integer path must not creep back in.
    assert "int8" not in SmaulOpt._STATE_DTYPES
    assert not hasattr(opt, "m_scale"), "int8 scale machinery should be gone"
    assert not hasattr(opt, "v_scale")


def test_bf16_default_halves_the_state():
    p = torch.nn.Parameter(torch.randn(64))
    sizes = {}
    for sdt in ("fp32", "bf16"):
        opt = SmaulOpt([p], lr=1e-3, state_dtype=sdt, clip=1e9)
        p.grad = torch.randn(64)
        opt.step(_dummy_model())
        sizes[sdt] = opt.m[p].element_size()
    assert sizes["bf16"] == 2 and sizes["fp32"] == 4, sizes
    assert sizes["bf16"] * 2 == sizes["fp32"], sizes


def test_bf16_tracks_fp32_closely():
    """The measured reason bf16 is the default: it stays within ~0.1% of the
    lossless fp32 state over ordinary gradients."""
    n, steps = 256, 30
    theta0 = torch.randn(n) * 0.1
    torch.manual_seed(0)
    grads = [torch.randn(n) * (0.1 ** (i % 4)) for i in range(steps)]
    out = {}
    for sdt in ("bf16", "fp32"):
        p = torch.nn.Parameter(theta0.clone())
        opt = SmaulOpt([p], lr=1e-3, state_dtype=sdt, clip=1e9)
        for g in grads:
            p.grad = g.clone()
            opt.step(_dummy_model())
        out[sdt] = p.detach().clone()
    rel = float((out["bf16"] - out["fp32"]).norm() / out["fp32"].norm())
    assert rel < 0.02, rel


def test_state_dtype_rejects_unknown():
    p = torch.nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="state_dtype"):
        SmaulOpt([p], state_dtype="int4")
    with pytest.raises(ValueError, match="state_dtype"):
        SmaulOpt([p], state_dtype="fp8")


def test_update_clip_validation():
    p = torch.nn.Parameter(torch.randn(4))
    assert SmaulOpt([p]).update_clip == 10.0
    with pytest.raises(ValueError, match="update_clip"):
        SmaulOpt([p], update_clip=0)
    with pytest.raises(ValueError, match="update_clip"):
        SmaulOpt([p], update_clip=-1.0)


def _heavy_tail_case(sdt, update_clip=10.0, n=256, steps=12):
    """One huge gradient element among small ones. fp16 is the case that needs
    the update bound here; bf16 tracks fp32 without help."""
    torch.manual_seed(0)
    theta0 = torch.ones(n) * 0.1
    p = torch.nn.Parameter(theta0.clone())
    opt = SmaulOpt([p], lr=1e-3, weight_decay=0.0, state_dtype=sdt,
                   clip=1e9, update_clip=update_clip)
    grads = [torch.randn(n) * 0.01 for _ in range(steps)]
    grads[5][0] = 1e3
    for g in grads:
        p.grad = g.clone()
        opt.step(_dummy_model())
    return p.detach().clone()


def test_fp32_state_is_never_clamped():
    """fp32 has no quantization error, so the invariant |u| <= ~1 already holds
    and the guard must not alter the reference path."""
    a = _heavy_tail_case("fp32")
    b = _heavy_tail_case("fp32", update_clip=1.0001)  # absurdly tight
    assert torch.equal(a, b), "update_clip must not affect fp32 state"


@pytest.mark.parametrize("sdt", ["bf16", "fp16"])
def test_heavy_tail_gradients_do_not_explode(sdt):
    """A heavy-tailed gradient must not blow the weights up. Now that the 1-byte
    integer path is gone, neither narrow-float width actually diverged here --
    this test is a regression guard that they keep tracking fp32."""
    ref = _heavy_tail_case("fp32")
    got = _heavy_tail_case(sdt)
    assert torch.isfinite(got).all(), sdt
    assert float(got.abs().max()) < 10 * float(ref.abs().max()), (sdt, float(got.abs().max()))


@pytest.mark.parametrize("sdt", ["bf16", "fp16"])
def test_update_clip_is_defensive_not_load_bearing(sdt):
    """The update bound is a guard, not a crutch: turning it off must not change
    the outcome materially. It does bind marginally -- |u| can sit just above 1
    because m_hat and v_hat use different bias corrections early on -- but the
    catastrophic |u| ~ 1e10 mode belonged to the removed 1-byte integer path."""
    ref = _heavy_tail_case("fp32")
    default = _heavy_tail_case(sdt)                # update_clip = 10.0
    off = _heavy_tail_case(sdt, update_clip=1e30)   # effectively disabled
    # Neither setting may diverge.
    for name, t in (("default", default), ("off", off)):
        assert torch.isfinite(t).all(), (sdt, name)
        assert float(t.abs().max()) < 10 * float(ref.abs().max()), (sdt, name, float(t.abs().max()))
    # And the two must agree closely: the bound is not deciding the outcome.
    rel = float((default - off).norm() / max(float(off.norm()), 1e-12))
    assert rel < 0.05, (sdt, rel)


def test_update_clip_never_binds_on_normal_gradients():
    """On well-scaled gradients |u| stays ~1, far below the default 10.0, so the
    reduced dtypes must match fp32 closely rather than being clipped."""
    def run(sdt, n=256, steps=40, seed=0):
        torch.manual_seed(seed)
        p = torch.nn.Parameter(torch.randn(n) * 0.1)
        opt = SmaulOpt([p], lr=1e-3, state_dtype=sdt, clip=1e9)
        for i in range(steps):
            torch.manual_seed(100 + i)
            p.grad = torch.randn(n) * (0.1 ** (i % 4))
            opt.step(_dummy_model())
        return p.detach().clone()
    ref = run("fp32")
    for sdt, tol in (("bf16", 0.02), ("fp16", 0.02)):
        rel = float((run(sdt) - ref).norm() / ref.norm())
        assert rel < tol, (sdt, rel)


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_state_storage_width(sdt):
    p = torch.nn.Parameter(torch.randn(64))
    opt = SmaulOpt([p], lr=1e-3, state_dtype=sdt, clip=1e9)
    p.grad = torch.randn(64)
    assert opt.step(_dummy_model()) != float("inf")
    m, v = opt.m[p], opt.v[p]
    expect = _STATE_BYTES[sdt]
    assert m.element_size() == expect, (sdt, m.dtype)
    assert v.element_size() == expect, (sdt, v.dtype)
    assert bool(torch.isfinite(m.float()).all()) and bool(torch.isfinite(v.float()).all())


@pytest.mark.parametrize("sdt", ["bf16", "fp16"])
def test_reduced_state_matches_fp32_closely(sdt):
    """The math is FP32 for every dtype, so results must track the baseline."""
    n, steps = 256, 12
    torch.manual_seed(0)
    grads = [torch.randn(n) * (0.1 ** (i % 4)) for i in range(steps)]
    theta0 = torch.randn(n) * 0.1
    out = {}
    for d in ("fp32", sdt):
        p = torch.nn.Parameter(theta0.clone())
        opt = SmaulOpt([p], lr=1e-3, state_dtype=d, clip=1e9)
        for g in grads:
            p.grad = g.clone()
            opt.step(_dummy_model())
        out[d] = p.detach().clone()
    rel = float((out[sdt] - out["fp32"]).norm() / out["fp32"].norm())
    assert rel < 0.02, (sdt, rel)


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_reduced_state_zero_gradients(sdt):
    theta0 = torch.tensor([1.5, -2.5], dtype=torch.float32)
    p = torch.nn.Parameter(theta0.clone())
    opt = SmaulOpt([p], lr=1e-4, weight_decay=0.01, state_dtype=sdt, clip=1e9)
    p.grad = torch.zeros(2)
    assert opt.step(_dummy_model()) != float("inf")
    # m=v=0 -> u=0 -> only decay, exactly as in fp32.
    assert torch.allclose(opt.m[p].float(), torch.zeros(2))
    assert torch.allclose(opt.v[p].float(), torch.zeros(2))
    assert torch.allclose(p.detach(), theta0 * (1 - 1e-4 * 0.01))


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_reduced_state_nonfinite_grads_skipped(sdt):
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    opt = SmaulOpt([p], lr=1e-4, state_dtype=sdt, clip=1e9)
    p.grad = torch.tensor([0.1, 0.1])
    opt.step(_dummy_model())
    before, m_b, v_b = p.detach().clone(), opt.m[p].clone(), opt.v[p].clone()
    p.grad = torch.tensor([float("nan"), 0.1])
    assert opt.step(_dummy_model()) == float("inf")
    assert opt.step_count == 1
    assert torch.equal(p.detach(), before)
    assert torch.equal(opt.m[p], m_b) and torch.equal(opt.v[p], v_b)


@pytest.mark.parametrize("sdt", ["bf16", "fp16"])
def test_reduced_state_fp8_storage_untouched(sdt):
    """Narrower state must not change the FP8 weight format or add a master copy."""
    from kernel.fp8_tile import FP8Linear
    torch.manual_seed(5)
    mod = FP8Linear(64, 130, tile=32)
    mod.train()
    holder = torch.nn.Module()
    holder.add_module("lin", mod)
    opt = SmaulOpt([], lr=2e-4, state_dtype=sdt, clip=1e9)
    for _ in range(3):
        mod(torch.randn(6, 64, requires_grad=True)).backward(torch.randn(6, 130))
        assert opt.step(holder) != float("inf")
    assert mod.w8.dtype == torch.uint8
    assert not any(p.shape == mod.w8.shape and p.dtype == torch.float32
                   for p in mod.parameters(recurse=False))
    # No scale side-buffers: narrowing a float state needs only a cast.
    assert not hasattr(opt, "m_scale") and not hasattr(opt, "v_scale")
    assert opt.m[mod].shape == (130, 64)


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_reduced_state_checkpoint_roundtrip(tmp_path, sdt):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype=sdt)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    d = tmp_path / sdt
    d.mkdir()
    _save_optimizer(d, opt, model)
    data = json.loads((d / "optimizer.json").read_text())
    assert data["state_dtype"] == sdt
    opt2 = SmaulOpt(list(model.parameters()))
    _load_optimizer(d, opt2, model)
    assert opt2.state_dtype == sdt
    assert opt2.step_count == 2
    for k in opt.m:
        assert opt2.m[k].dtype == opt.m[k].dtype, (sdt, k)
        assert torch.equal(opt.m[k], opt2.m[k]), sdt
        assert torch.equal(opt.v[k], opt2.v[k]), sdt
    # The saved file must reflect the reduced width, not a forced fp32 copy.
    assert (d / "optimizer_state.safetensors").stat().st_size > 0


def test_state_file_size_shrinks_with_dtype(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    sizes = {}
    for sdt in ("bf16", "fp16", "fp32"):
        model = SmaulLinear(cfg)
        opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype=sdt)
        ids = torch.randint(0, 64, (2, 8))
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
        d = tmp_path / sdt
        d.mkdir()
        _save_optimizer(d, opt, model)
        sizes[sdt] = (d / "optimizer_state.safetensors").stat().st_size
    assert sizes["bf16"] < sizes["fp32"]
    # Both 2-byte widths are half of fp32.
    assert sizes["bf16"] == pytest.approx(sizes["fp32"] * 0.5, rel=0.05)
    assert sizes["fp16"] == pytest.approx(sizes["fp32"] * 0.5, rel=0.05)


def test_state_dtype_preserved_through_resume(tmp_path):
    torch.manual_seed(9)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype="bf16")
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    model.save_pretrained(tmp_path / "ckpt")
    _save_optimizer(tmp_path / "ckpt", opt, model)
    model2 = SmaulLinear.from_pretrained(tmp_path / "ckpt")
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(tmp_path / "ckpt", opt2, model2)
    assert opt2.state_dtype == "bf16"
    opt2.zero_grad(model2)
    _, loss = model2(ids, ids)
    assert torch.isfinite(loss)
    loss.backward()
    opt2.step(model2)
    _, after = model2(ids, ids)
    assert torch.isfinite(after)
    assert opt2.step_count == 3


def test_checkpoint_rejects_unknown_state_dtype(tmp_path):
    p = torch.nn.Parameter(torch.randn(2))
    opt = SmaulOpt([p])
    d = opt.state_dict()
    d["state_dtype"] = "int4"
    with pytest.raises(ValueError, match="state_dtype"):
        opt.load_state_dict(d)


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_fp8_state_survives_checkpoint(tmp_path, sdt):
    """FP8 module states (the fp8.* keys) must round-trip at their own width, and
    the file must contain no scale side-buffers now that int8 is gone."""
    from safetensors.torch import load_file
    torch.manual_seed(4)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    mods = fp8_modules(model)
    assert mods, "expected FP8 modules"
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype=sdt)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    d = tmp_path / sdt
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    blobs = load_file(str(d / "optimizer_state.safetensors"), device="cpu")
    assert any(k.startswith("m.fp8.") for k in blobs)
    assert not any("_scale" in k for k in blobs), sorted(blobs)[:5]
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)
    assert opt2.state_dtype == sdt and opt2.step_count == 2
    mods2 = dict(fp8_modules(model2))
    for name, m in mods:
        m2 = mods2[name]
        assert opt2.m[m2].dtype == opt.m[m].dtype
        assert torch.equal(opt.m[m], opt2.m[m2]), name
        assert torch.equal(opt.v[m], opt2.v[m2]), name
    # And the resumed optimizer must keep stepping without NaN.
    opt2.zero_grad(model2)
    _, loss = model2(ids, ids)
    loss.backward()
    assert opt2.step(model2) != float("inf")
    _, after = model2(ids, ids)
    assert torch.isfinite(after)


def test_constructor_defaults():
    p = torch.nn.Parameter(torch.randn(4))
    opt = SmaulOpt([p])
    assert opt.lr == 1e-4
    assert opt.beta_m == 0.9
    assert opt.beta_v == 0.999
    assert opt.epsilon == 1e-8
    assert opt.weight_decay == 0.01
    assert opt.step_count == 0
    assert opt.m == {} and opt.v == {}
    sd = opt.state_dict()
    assert sd["name"] == "smaul"
    assert sd["step"] == 0
    assert sd["beta_m"] == 0.9 and sd["beta_v"] == 0.999
    assert sd["epsilon"] == 1e-8
    assert sd["learning_rate"] == 1e-4 and sd["weight_decay"] == 0.01


def test_constructor_validation():
    p = torch.nn.Parameter(torch.randn(2))
    with pytest.raises(ValueError):
        SmaulOpt([p], lr=0)
    with pytest.raises(ValueError):
        SmaulOpt([p], lr=-1e-4)
    with pytest.raises(ValueError):
        SmaulOpt([p], beta_m=1.0)
    with pytest.raises(ValueError):
        SmaulOpt([p], beta_v=-0.1)
    with pytest.raises(ValueError):
        SmaulOpt([p], epsilon=0)
    with pytest.raises(ValueError):
        SmaulOpt([p], weight_decay=-0.1)
    with pytest.raises(ValueError):
        SmaulOpt([p], clip=0)


def test_configurable():
    p = torch.nn.Parameter(torch.randn(2))
    opt = SmaulOpt([p], lr=3e-4, beta_m=0.8, beta_v=0.99, epsilon=1e-6,
                   weight_decay=0.02, clip=2.0)
    assert opt.lr == 3e-4 and opt.beta_m == 0.8 and opt.beta_v == 0.99
    assert opt.epsilon == 1e-6 and opt.weight_decay == 0.02 and opt.clip == 2.0


def test_deterministic_first_update():
    torch.manual_seed(0)
    theta0 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    p = torch.nn.Parameter(theta0.clone())
    # fp32 state so the hand-computed reference is exactly checkable.
    opt = SmaulOpt([p], lr=1e-4, beta_m=0.9, beta_v=0.999, epsilon=1e-8,
                   weight_decay=0.01, clip=1e9, state_dtype="fp32")
    g = torch.tensor([0.1, -0.2], dtype=torch.float32)
    p.grad = g.clone()
    norm = opt.step(_dummy_model())
    assert norm != float("inf")
    assert opt.step_count == 1
    # Reference: m1=0.1*g, v1=0.001*|g|, m_hat=g, v_hat=|g|.
    m1 = 0.1 * g
    v1 = 0.001 * g.abs()
    m_hat = m1 / 0.1
    v_hat = v1 / 0.001
    u = m_hat / (v_hat + 1e-8)
    expected = theta0 * (1 - 1e-4 * 0.01) - 1e-4 * u
    assert torch.allclose(p.detach(), expected, atol=1e-7, rtol=1e-6), (p.detach(), expected)
    # States are FP32.
    assert opt.m[p].dtype == torch.float32 and opt.v[p].dtype == torch.float32
    assert torch.allclose(opt.m[p], m1, atol=1e-8)
    assert torch.allclose(opt.v[p], v1, atol=1e-10)


def test_deterministic_second_update_and_bias_correction():
    theta0 = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32)
    p = torch.nn.Parameter(theta0.clone())
    opt = SmaulOpt([p], lr=1e-4, beta_m=0.9, beta_v=0.99, epsilon=1e-8,
                   weight_decay=0.0, clip=1e9, state_dtype="fp32")
    g1 = torch.tensor([0.5, -0.25, 0.1], dtype=torch.float32)
    g2 = torch.tensor([0.25, 0.5, -0.3], dtype=torch.float32)
    m = torch.zeros(3)
    v = torch.zeros(3)
    th = theta0.clone()
    for t, g in enumerate([g1, g2], start=1):
        p.grad = g.clone()
        opt.step(_dummy_model())
        m = 0.9 * m + 0.1 * g
        v = 0.99 * v + 0.01 * g.abs()
        bc1 = 1 - 0.9 ** t
        bc2 = 1 - 0.99 ** t
        u = (m / bc1) / (v / bc2 + 1e-8)
        th = th - 1e-4 * u
        assert torch.allclose(p.detach(), th, atol=1e-7, rtol=1e-6), (t, p.detach(), th)
    assert opt.step_count == 2
    # Bias correction must actually change the result: without dividing by
    # (1 - beta**t) the second-step update would differ measurably.
    m_nc = 0.9 * (0.1 * g1) + 0.1 * g2
    v_nc = 0.99 * (0.01 * g1.abs()) + 0.01 * g2.abs()
    u_nc = m_nc / (v_nc + 1e-8)
    uncorrected = theta0 - 1e-4 * (g1 / (g1.abs() + 1e-8)) - 1e-4 * u_nc
    assert not torch.allclose(p.detach(), uncorrected, atol=1e-6)


def test_determinism_repeated():
    def run_once():
        torch.manual_seed(123)
        p = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
        opt = SmaulOpt([p], lr=2e-4, clip=1e9, state_dtype="fp32")
        for _ in range(3):
            g = torch.randn(8)
            p.grad = g.clone()
            opt.step(_dummy_model())
        return p.detach().clone(), opt.m[p].clone(), opt.v[p].clone(), opt.step_count
    a_p, a_m, a_v, a_s = run_once()
    b_p, b_m, b_v, b_s = run_once()
    assert a_s == b_s == 3
    assert torch.equal(a_p, b_p)
    assert torch.equal(a_m, b_m)
    assert torch.equal(a_v, b_v)


def test_weight_decay_decoupled():
    theta0 = torch.tensor([2.0, -3.0], dtype=torch.float32)
    g = torch.tensor([0.2, 0.1], dtype=torch.float32)
    p0 = torch.nn.Parameter(theta0.clone())
    o0 = SmaulOpt([p0], lr=1e-3, weight_decay=0.0, clip=1e9, state_dtype="fp32")
    p0.grad = g.clone()
    o0.step(_dummy_model())
    p1 = torch.nn.Parameter(theta0.clone())
    o1 = SmaulOpt([p1], lr=1e-3, weight_decay=0.05, clip=1e9, state_dtype="fp32")
    p1.grad = g.clone()
    o1.step(_dummy_model())
    # Same u, difference is exactly lr*wd*theta0 (float32 rounding).
    diff = p0.detach() - p1.detach()
    assert torch.allclose(diff, torch.tensor(1e-3 * 0.05) * theta0, atol=1e-6, rtol=1e-5)


def test_zero_gradients():
    theta0 = torch.tensor([1.5, -2.5], dtype=torch.float32)
    p = torch.nn.Parameter(theta0.clone())
    opt = SmaulOpt([p], lr=1e-4, weight_decay=0.01, clip=1e9, state_dtype="fp32")
    p.grad = torch.zeros(2)
    opt.step(_dummy_model())
    # m=v=0 -> u=0 -> only decay.
    assert torch.allclose(opt.m[p], torch.zeros(2))
    assert torch.allclose(opt.v[p], torch.zeros(2))
    assert torch.allclose(p.detach(), theta0 * (1 - 1e-4 * 0.01))
    assert opt.step_count == 1


def test_small_gradients_finite():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    opt = SmaulOpt([p], lr=1e-4, clip=1e9)
    p.grad = torch.tensor([1e-30, -1e-30])
    n = opt.step(_dummy_model())
    assert n != float("inf")
    assert torch.isfinite(p).all() and torch.isfinite(opt.m[p]).all()
    assert torch.isfinite(opt.v[p]).all()


def test_large_gradients_finite_no_clip():
    p = torch.nn.Parameter(torch.tensor([0.5, -0.5]))
    opt = SmaulOpt([p], lr=1e-4, weight_decay=0.0, clip=1e9)
    p.grad = torch.tensor([1e4, -1e4])
    n = opt.step(_dummy_model())
    assert n != float("inf")
    assert torch.isfinite(p).all()
    # u ~ sign(g) for large |g| >> eps.
    assert p[0].item() < 0.5 and p[1].item() > -0.5


def test_large_gradients_are_clipped_like_lion():
    p = torch.nn.Parameter(torch.ones(4))
    opt = SmaulOpt([p], lr=1e-4, clip=1.0)
    p.grad = torch.full((4,), 10.0)
    n = opt.step(_dummy_model())
    assert n > 1.0  # norm reported pre-clip
    assert torch.isfinite(p).all()


def test_nan_inf_safety_skips_and_keeps_counter():
    for bad in (float("nan"), float("inf"), float("-inf")):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        opt = SmaulOpt([p], lr=1e-4, clip=1e9)
        # One good step first.
        p.grad = torch.tensor([0.1, 0.1])
        opt.step(_dummy_model())
        assert opt.step_count == 1
        m_before, v_before = opt.m[p].clone(), opt.v[p].clone()
        p_before = p.detach().clone()
        p.grad = torch.tensor([bad, 0.1])
        n = opt.step(_dummy_model())
        assert n == float("inf")
        assert opt.step_count == 1  # not advanced
        assert torch.equal(p.detach(), p_before)
        assert torch.equal(opt.m[p], m_before)
        assert torch.equal(opt.v[p], v_before)
        assert p.grad is None  # cleared


def test_state_init_and_step_counter():
    p = torch.nn.Parameter(torch.randn(3))
    q = torch.nn.Parameter(torch.randn(3))
    opt = SmaulOpt([p, q], lr=1e-4, clip=1e9, state_dtype="fp32")
    assert opt.step_count == 0 and opt.m == {} and opt.v == {}
    p.grad = torch.randn(3)
    # q has no grad this step.
    opt.step(_dummy_model())
    assert opt.step_count == 1
    assert p in opt.m and p in opt.v
    assert q not in opt.m  # no grad -> no state
    assert opt.m[p].dtype == torch.float32
    q.grad = torch.randn(3)
    p.grad = torch.randn(3)
    opt.step(_dummy_model())
    assert opt.step_count == 2
    assert q in opt.m


def test_checkpoint_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    assert opt.step_count == 2
    _save_optimizer(tmp_path, opt, model)
    assert (tmp_path / "optimizer.json").exists()
    data = json.loads((tmp_path / "optimizer.json").read_text())
    assert data["name"] == "smaul" and data["step"] == 2
    assert "beta_m" in data and "beta_v" in data and "epsilon" in data
    assert "weight_decay" in data and "learning_rate" in data
    assert (tmp_path / "optimizer_state.safetensors").exists()
    # Load into fresh optimizer bound to the same model objects.
    opt2 = SmaulOpt(list(model.parameters()), lr=9e-5)
    _load_optimizer(tmp_path, opt2, model)
    assert opt2.step_count == 2 and opt2.lr == 1e-4
    for k in opt.m:
        assert k in opt2.m and torch.equal(opt.m[k], opt2.m[k])
        assert torch.equal(opt.v[k], opt2.v[k])


def test_checkpoint_resume_continues_identically(tmp_path):
    torch.manual_seed(7)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=2e-4)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    model.save_pretrained(tmp_path / "ckpt")
    _save_optimizer(tmp_path / "ckpt", opt, model)
    # Reference: one more step on the original (same ids + same weights =>
    # same gradients, so the restored run must land on the same parameters).
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    ref_params = [p.detach().clone() for p in model.parameters()]
    # Restore model + optimizer and replay the same step.
    model2 = SmaulLinear.from_pretrained(tmp_path / "ckpt")
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(tmp_path / "ckpt", opt2, model2)
    assert opt2.step_count == 2
    opt2.zero_grad(model2)
    _, loss2 = model2(ids, ids)
    loss2.backward()
    opt2.step(model2)
    for a, b in zip(ref_params, model2.parameters()):
        assert torch.allclose(a, b.detach(), atol=1e-7, rtol=1e-6)
    assert opt2.step_count == 3


def test_loading_other_optimizer_fails_clearly(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=32, d_model=16, n_layer=1, n_heads=2, tile=16,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    lion = Lion(list(model.parameters()), lr=1e-4)
    (tmp_path).mkdir(parents=True, exist_ok=True)
    from train import _save_optimizer as _save
    _save(tmp_path, lion, model)
    opt = SmaulOpt(list(model.parameters()))
    with pytest.raises(ValueError, match="smaul|Lion|name|m/v"):
        _load_optimizer(tmp_path, opt, model)


def test_lion_checkpoints_remain_loadable(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=32, d_model=16, n_layer=1, n_heads=2, tile=16,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    lion = Lion(list(model.parameters()), lr=2e-4)
    _save_optimizer(tmp_path, lion, model)
    lion2 = Lion(list(model.parameters()), lr=1e-5)
    _load_optimizer(tmp_path, lion2, model)
    assert lion2.lr == 2e-4


def test_fp8_integration_keeps_storage_and_state(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    mods = fp8_modules(model)
    assert mods, "expected FP8 modules"
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype="fp32")
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    assert torch.isfinite(loss)
    loss.backward()
    n = opt.step(model)
    assert n != float("inf")
    assert opt.step_count == 1
    for _, m in mods:
        assert m.w8.dtype == torch.uint8
        assert m._gw is None  # cleared after update
        key = m
        assert key in opt.m and key in opt.v
        assert opt.m[key].shape == (m.out_f, m.in_f)
        assert opt.m[key].dtype == torch.float32
    _, loss2 = model(ids, ids)
    assert torch.isfinite(loss2)
    # Overhead is ~2 FP32 per FP8 weight element.
    for _, m in mods:
        assert opt.m[m].numel() == m.out_f * m.in_f
        assert opt.v[m].numel() == m.out_f * m.in_f


def test_fp8_blockwise_matches_full_matrix_reference():
    from kernel.fp8_tile import FP8Linear
    torch.manual_seed(6)
    lr, wd, bm, bv, eps = 2e-4, 0.01, 0.9, 0.999, 1e-8
    m = FP8Linear(64, 130, tile=32)
    m.train()
    holder = torch.nn.Module()
    holder.add_module("lin", m)
    opt = SmaulOpt([], lr=lr, beta_m=bm, beta_v=bv, epsilon=eps, weight_decay=wd,
                   clip=1e9, state_dtype="fp32")
    # Attach FP8 grad manually by running a backward.
    m(torch.randn(6, 64, requires_grad=True)).backward(torch.randn(6, 130))
    gw = m._gw.clone()
    w8_before, sc_before = m.w8.clone(), m.sc.clone()
    opt.step(holder)
    assert m._gw is None
    # Reference full-matrix computation.
    st_m = torch.zeros_like(gw)
    st_v = torch.zeros_like(gw)
    st_m_ref = (st_m * bm + gw * (1 - bm))
    st_v_ref = (st_v * bv + gw.abs() * (1 - bv))
    bc1, bc2 = 1 - bm, 1 - bv
    u_ref = (st_m_ref / bc1) / (st_v_ref / bc2 + eps)
    upd_ref = u_ref * lr
    assert torch.equal(opt.m[m], st_m_ref)
    assert torch.equal(opt.v[m], st_v_ref)
    m_ref = FP8Linear(64, 130, tile=32)
    m_ref.w8.copy_(w8_before)
    m_ref.sc.copy_(sc_before)
    m_ref.requant(upd_ref, lr * wd)
    assert torch.equal(m.w8, m_ref.w8)
    assert torch.equal(m.sc, m_ref.sc)


def test_works_with_all_param_types_plain_and_rawr():
    for arch in ("plain", "rawr"):
        torch.manual_seed(0)
        cfg = LinearConfig(vocab_size=48, d_model=16, n_layer=1, n_heads=2,
                           precision="fp32", architecture=arch,
                           embedding_storage="ram")
        model = SmaulLinear(cfg)
        opt = SmaulOpt(list(model.parameters()), lr=1e-4)
        ids = torch.randint(0, 48, (2, 8))
        before = [p.detach().clone() for p in model.parameters()]
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
        assert torch.isfinite(loss)
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert changed, arch
        # Every trainable param with a grad got FP32 states.
        for p in model.parameters():
            if p.grad is not None:
                assert p in opt.m and p in opt.v, arch


def test_training_smoke_loss_sensible_and_resumable(tmp_path):
    torch.manual_seed(11)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    model.train()
    opt = SmaulOpt(list(model.parameters()), lr=1e-4)
    ids = torch.randint(0, 64, (2, 16))
    losses = []
    for _ in range(5):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        assert torch.isfinite(loss)
        loss.backward()
        n = opt.step(model)
        assert n != float("inf")
        losses.append(float(loss.detach()))
    assert all(v < 1e4 for v in losses)
    model.save_pretrained(tmp_path / "s")
    _save_optimizer(tmp_path / "s", opt, model)
    m2 = SmaulLinear.from_pretrained(tmp_path / "s")
    o2 = SmaulOpt(list(m2.parameters()))
    _load_optimizer(tmp_path / "s", o2, m2)
    assert o2.step_count == 5
    o2.zero_grad(m2)
    _, loss = m2(ids, ids)
    assert torch.isfinite(loss)
    loss.backward()
    o2.step(m2)
    _, loss_after = m2(ids, ids)
    assert torch.isfinite(loss_after)

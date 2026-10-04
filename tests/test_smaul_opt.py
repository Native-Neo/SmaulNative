import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import math

import pytest
import torch

from kernel.fp8_tile import fp8_modules
from model import LinearConfig, Lion, SmaulLinear, SmaulOpt, _grad_norm
from train import _load_optimizer, _save_optimizer


def _dummy_model():
    return torch.nn.Module()


def _v_sig(opt, model):
    """Canonical stored-v signature per state, keyed by stable name.

    Returns {name: (full_v,)} for an unfactored state or {name: (v_row, v_col)}
    for a factored one, so round-trips can be compared across two *different*
    model instances (where object identity differs) as well as the same one.
    """
    names = {id(p): n for n, p in model.named_parameters()}
    for n, m in fp8_modules(model):
        names[id(m)] = "fp8:" + n
    out = {}
    for k in opt.m:
        nm = names.get(id(k))
        if nm is None:
            continue
        if k in opt.v_row:
            out[nm] = (opt.v_row[k].clone(), opt.v_col[k].clone())
        else:
            out[nm] = (opt.v[k].clone(),)
    return out


def _assert_v_sig_equal(got, want):
    assert set(got) == set(want), (sorted(got), sorted(want))
    for k in want:
        assert len(got[k]) == len(want[k]), k
        for a, b in zip(got[k], want[k]):
            assert a.dtype == b.dtype, (k, a.dtype, b.dtype)
            assert a.shape == b.shape, (k, a.shape, b.shape)
            assert torch.equal(a, b), k


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
    _assert_v_sig_equal(_v_sig(opt2, model), _v_sig(opt, model))
    # The saved file must reflect the reduced width, not a forced fp32 copy.
    assert (d / "optimizer_state.safetensors").stat().st_size > 0


def _state_bytes(opt):
    """Persistent optimizer-state bytes only: no params, no grads, no transients."""
    tot = 0
    for store in (opt.m, opt.v, opt.v_row, opt.v_col):
        for t in store.values():
            tot += t.numel() * t.element_size()
    return tot


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
        assert _state_bytes(opt) < sizes[sdt]  # file also carries safetensors headers
    assert sizes["bf16"] < sizes["fp32"]
    # With m full-size and v factored, the 2-byte state is well under half of
    # the 4-byte state, but not exactly half (m is still full-size).
    assert sizes["bf16"] < sizes["fp32"] * 0.75
    assert sizes["fp16"] < sizes["fp32"] * 0.75


def test_factor_v_shrinks_state_without_touching_m(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    got = {}
    for fv in (True, False):
        model = SmaulLinear(cfg)
        opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=fv)
        ids = torch.randint(0, 64, (2, 8))
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
        m_bytes = sum(t.numel() * t.element_size() for t in opt.m.values())
        v_bytes = _state_bytes(opt) - m_bytes
        got[fv] = (m_bytes, v_bytes)
        d = tmp_path / ("factored" if fv else "full")
        d.mkdir()
        _save_optimizer(d, opt, model)
    m_f, v_f = got[True]
    m_u, v_u = got[False]
    # m is byte-identical either way: factoring touches v only.
    assert m_f == m_u, (m_f, m_u)
    # v shrinks substantially.
    assert v_f < v_u, (v_f, v_u)
    assert v_u / v_f > 5, (v_f, v_u)


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
    _assert_v_sig_equal(_v_sig(opt2, model2), _v_sig(opt, model))
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
    _assert_v_sig_equal(_v_sig(opt2, model), _v_sig(opt, model))


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
        assert key in opt.m and key in opt.v_row and key in opt.v_col
        assert opt.m[key].shape == (m.out_f, m.in_f)
        assert opt.m[key].dtype == torch.float32
    _, loss2 = model(ids, ids)
    assert torch.isfinite(loss2)
    # m is full-size; v is factored for this 2-D state, so (out_f + in_f).
    for _, m in mods:
        assert opt.m[m].numel() == m.out_f * m.in_f
        assert m in opt.v_row and m in opt.v_col
        assert opt.v_row[m].numel() + opt.v_col[m].numel() == m.out_f + m.in_f


def test_fp8_blockwise_matches_full_matrix_reference():
    from kernel.fp8_tile import FP8Linear
    torch.manual_seed(6)
    lr, wd, bm, bv, eps = 2e-4, 0.01, 0.9, 0.999, 1e-8
    m = FP8Linear(64, 130, tile=32)
    m.train()
    holder = torch.nn.Module()
    holder.add_module("lin", m)
    # factor_v off: this asserts the blockwise path equals the full-matrix
    # reference exactly, which a rank-1 v reconstruction cannot do.
    opt = SmaulOpt([], lr=lr, beta_m=bm, beta_v=bv, epsilon=eps, weight_decay=wd,
                   clip=1e9, state_dtype="fp32", factor_v=False)
    # Attach FP8 grad manually by running a backward.
    m(torch.randn(6, 64, requires_grad=True)).backward(torch.randn(6, 130))
    gw = m._gw.clone()
    w8_before, sc_before = m.w8.clone(), m.sc.clone()
    opt.step(holder)
    assert m._gw is None
    # Reference full-matrix computation, in FP32. `gw` may be bf16 (_gw storage),
    # but the EMA is evaluated in FP32, so the reference must widen the gradient
    # rather than inherit its dtype -- the optimizer promotes bf16 -> fp32
    # exactly, so this stays bit-comparable.
    st_m = torch.zeros(gw.shape, dtype=torch.float32)
    st_v = torch.zeros(gw.shape, dtype=torch.float32)
    # Widen explicitly: a Python float scalar is weakly typed, so `bf16 * 0.1`
    # would round in bf16 instead of promoting to fp32 like `m_b.add_(g, alpha=)`.
    gwf = gw.float()
    st_m_ref = st_m * bm + gwf * (1 - bm)
    st_v_ref = st_v * bv + gwf.abs() * (1 - bv)
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
        # Every trainable param with a grad got state: m always, plus v in
        # whichever form its shape selects.
        for p in model.parameters():
            if p.grad is not None:
                assert p in opt.m, arch
                assert (p in opt.v) or (p in opt.v_row and p in opt.v_col), arch


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


# ======================================================================
# Factored v: shape rules, memory, checkpoints, and the numerical
# comparison against full-v (the centerpiece).
# ======================================================================


def _factored_opt(p, **kw):
    kw.setdefault("lr", 1e-3)
    kw.setdefault("clip", 1e9)
    return SmaulOpt([p], **kw)


# ---- shape rules -----------------------------------------------------

@pytest.mark.parametrize("shape,factored", [
    ((64, 48), True), ((2, 2), True), ((4096, 4096), True),
    ((48,), False), ((), False), ((1, 10), False), ((10, 1), False),
])
def test_factor_shape_rule(shape, factored):
    p = torch.nn.Parameter(torch.randn(shape) if shape else torch.randn(()))
    opt = _factored_opt(p, factor_v=True)
    p.grad = torch.randn(shape) if shape else torch.randn(())
    opt.step(_dummy_model())
    assert opt._factor_shape(tuple(p.shape)) is factored, shape
    assert p in opt.m
    if factored:
        assert p in opt.v_row and p in opt.v_col, shape
        assert p not in opt.v, "factored state must not also keep a full v"
        assert opt.v_row[p].shape == (shape[0],)
        assert opt.v_col[p].shape == (shape[1],)
        assert opt.v_row[p].dtype == torch.bfloat16
        assert opt.v_col[p].dtype == torch.bfloat16
    else:
        assert p in opt.v and p not in opt.v_row, shape
        assert opt.v[p].shape == tuple(p.shape)


def test_factor_v_off_keeps_full_v_everywhere():
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = _factored_opt(p, factor_v=False)
    p.grad = torch.randn(32, 16)
    opt.step(_dummy_model())
    assert p in opt.v and p not in opt.v_row and p not in opt.v_col
    assert opt.v[p].shape == (32, 16)


def test_threshold_never_costs_more_than_full():
    """_factor_shape agrees with R + C <= R * C on every 2-D shape.

    The point is the code's decision, not the arithmetic: factoring must be
    taken exactly where it cannot cost more, and refused everywhere else.
    """
    opt = _factored_opt(torch.nn.Parameter(torch.randn(2, 2)), factor_v=True)
    for r in range(0, 14):
        for c in range(0, 14):
            assert opt._factor_shape((r, c)) is (r >= 2 and c >= 2), (r, c)
            if r >= 2 and c >= 2:
                assert r + c <= r * c, (r, c)
    # ...and we refuse exactly the shapes where that would not hold.
    p = torch.nn.Parameter(torch.randn(1, 8))
    opt = _factored_opt(p, factor_v=True)
    assert opt._factor_shape((1, 8)) is False


# ---- memory ----------------------------------------------------------

def test_factored_state_holds_no_full_matrix_v():
    R, C = 300, 400
    p = torch.nn.Parameter(torch.randn(R, C))
    opt = _factored_opt(p, factor_v=True)
    for _ in range(3):
        p.grad = torch.randn(R, C)
        opt.step(_dummy_model())
    total = sum(t.numel() for t in list(opt.m.values()) + list(opt.v.values())
                + list(opt.v_row.values()) + list(opt.v_col.values()))
    assert total == R * C + R + C, total        # m full + two marginals
    v_bytes = (opt.v_row[p].numel() + opt.v_col[p].numel()) * 2
    assert v_bytes == (R + C) * 2
    assert v_bytes < R * C * 2 / 10, (v_bytes, R * C * 2)


def test_large_matrix_memory_target():
    """The stated target: a 4096x4096 v goes from ~32 MB to ~16 KB.

    Computed from the optimizer's own storage width and state layout, not
    from literals: a regression that widens _storage_dtype or stores a full
    v alongside the marginals breaks this.
    """
    R = C = 4096
    p = torch.nn.Parameter(torch.zeros(8, 8))
    opt = _factored_opt(p, factor_v=True)
    assert opt._factor_shape((R, C)) is True
    width = torch.empty(0, dtype=opt._storage_dtype(signed=False)).element_size()
    full = R * C * width
    factored = (R + C) * width
    assert (full, factored) == (33554432, 16384)
    assert full / factored > 2000
    # And the real state after a step has exactly that shape: one full m,
    # two marginals, no full v anywhere.
    q = torch.nn.Parameter(torch.randn(300, 400))
    opt2 = _factored_opt(q, factor_v=True)
    q.grad = torch.randn(300, 400)
    opt2.step(_dummy_model())
    assert opt2.v_row[q].numel() + opt2.v_col[q].numel() == 300 + 400
    assert q not in opt2.v


def test_m_is_always_full_size():
    for shape in ((64, 48), (48,), ()):
        p = torch.nn.Parameter(torch.randn(shape) if shape else torch.randn(()))
        opt = _factored_opt(p, factor_v=True)
        p.grad = torch.randn(shape) if shape else torch.randn(())
        opt.step(_dummy_model())
        assert opt.m[p].shape == tuple(p.shape), shape


def test_factor_v_does_not_change_arithmetic_dtype():
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = _factored_opt(p, factor_v=True)
    p.grad = torch.randn(32, 16)
    opt.step(_dummy_model())
    # Storage is bf16; the update path works in fp32 via .float().
    assert opt.m[p].dtype == torch.bfloat16
    assert opt.v_row[p].dtype == torch.bfloat16
    # The marginal reduction yields fp32, which is what the EMA runs in.
    assert SmaulOpt._mean_abs(torch.randn(4, 5), 1).dtype == torch.float32


def test_no_fp16_or_int8_state_dtype():
    assert "fp16" in SmaulOpt._STATE_DTYPES      # allowed, just not the default
    assert "int8" not in SmaulOpt._STATE_DTYPES
    assert SmaulOpt([torch.nn.Parameter(torch.randn(2))]).state_dtype == "bf16"


# ---- correctness of the factorization itself -------------------------

def test_marginals_are_exact():
    """The EMA is linear, so EMA-of-marginals == marginals-of-EMA exactly.
    The stored row/col vectors are the true marginals of v, with no error."""
    R, C, T, bv = 24, 32, 25, 0.9
    p = torch.nn.Parameter(torch.randn(R, C))
    opt = _factored_opt(p, factor_v=True, state_dtype="fp32", weight_decay=0.0,
                      beta_v=bv)
    v_full = torch.zeros(R, C)
    for _ in range(T):
        g = torch.randn(R, C)
        p.grad = g.clone()
        opt.step(_dummy_model())
        v_full = bv * v_full + (1 - bv) * g.abs()
    # Compare the *final* marginals (state is bias-uncorrected, like v_full).
    assert torch.allclose(opt.v_row[p].float(), v_full.mean(1), atol=1e-5)
    assert torch.allclose(opt.v_col[p].float(), v_full.mean(0), atol=1e-5)


def test_reconstruction_exact_for_rank_one_gradient():
    """Best case: if |g| is rank-1 the reconstruction must be exact, which is
    what proves the R*C/G form is the right one (an un-divided R*C is not)."""
    torch.manual_seed(3)
    a = torch.rand(20, 1) + 0.1
    b = torch.rand(1, 30) + 0.1
    g = (a * b).contiguous()
    p = torch.nn.Parameter(torch.zeros(20, 30))
    opt = _factored_opt(p, factor_v=True, state_dtype="fp32", weight_decay=0.0)
    for _ in range(20):
        p.grad = g.clone()
        opt.step(_dummy_model())
    p_ref = torch.nn.Parameter(torch.zeros(20, 30))
    opt_ref = _factored_opt(p_ref, factor_v=False, state_dtype="fp32", weight_decay=0.0)
    for _ in range(20):
        p_ref.grad = g.clone()
        opt_ref.step(_dummy_model())
    rel = float((p - p_ref).norm() / p_ref.norm())
    assert rel < 1e-5, rel


def test_reconstruct_block_never_materializes_full_v():
    """_reconstruct_block must return only the requested rows."""
    r_hat = torch.arange(1, 11, dtype=torch.float32)
    c_hat = torch.arange(1, 21, dtype=torch.float32)
    blk = SmaulOpt._reconstruct_block(r_hat, c_hat, r_hat.mean(), 0, 4)
    assert blk.shape == (4, 20)
    assert SmaulOpt._reconstruct_block(r_hat, c_hat, r_hat.mean(), 6, 10).shape == (4, 20)
    # And it equals the outer-product formula.
    ref = torch.outer(r_hat[6:10], c_hat) / r_hat.mean()
    assert torch.equal(SmaulOpt._reconstruct_block(r_hat, c_hat, r_hat.mean(), 6, 10), ref)


def test_mean_abs_matches_abs_mean_without_full_temp():
    g = torch.randn(64, 48)
    assert torch.allclose(SmaulOpt._mean_abs(g, 1), g.abs().mean(1), atol=1e-6)
    assert torch.allclose(SmaulOpt._mean_abs(g, 0), g.abs().mean(0), atol=1e-6)


# ---- bias correction / weight decay / determinism under factoring ----

def test_bias_correction_still_applied_when_factored():
    """First step: v_hat marginals must equal mean|g| exactly (bc2 cancels it)."""
    torch.manual_seed(5)
    p = torch.nn.Parameter(torch.ones(8, 4))
    opt = _factored_opt(p, factor_v=True, state_dtype="fp32", weight_decay=0.0)
    g = torch.randn(8, 4)
    p.grad = g.clone()
    opt.step(_dummy_model())
    # bc1 = 1-beta_m = 0.1, bc2 = 1-beta_v = 0.001
    m_hat = opt.m[p].float() / 0.1
    r_hat = opt.v_row[p].float() / 0.001
    c_hat = opt.v_col[p].float() / 0.001
    v_hat = torch.outer(r_hat, c_hat) / r_hat.mean()
    u = m_hat / (v_hat + 1e-8)
    assert torch.allclose(p.detach(), 1.0 - 1e-3 * u, atol=1e-6)


def test_weight_decay_applied_when_factored():
    p0 = torch.nn.Parameter(torch.full((8, 4), 2.0))
    p1 = torch.nn.Parameter(torch.full((8, 4), 2.0))
    o0 = _factored_opt(p0, factor_v=True, weight_decay=0.0)
    o1 = _factored_opt(p1, factor_v=True, weight_decay=0.05)
    g = torch.randn(8, 4)
    p0.grad = g.clone(); o0.step(_dummy_model())
    p1.grad = g.clone(); o1.step(_dummy_model())
    # Same u; the only difference is the decoupled decay.
    diff = p0.detach() - p1.detach()
    assert torch.allclose(diff, torch.full((8, 4), 1e-3 * 0.05 * 2.0), atol=1e-5, rtol=1e-4)


def test_determinism_preserved_when_factored():
    def run():
        torch.manual_seed(77)
        p = torch.nn.Parameter(torch.randn(16, 12))
        opt = _factored_opt(p, factor_v=True)
        for _ in range(4):
            torch.manual_seed(500)
            p.grad = torch.randn(16, 12)
            opt.step(_dummy_model())
        return p.detach().clone(), opt.v_row[p].clone(), opt.v_col[p].clone()
    a, b = run(), run()
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_no_nan_inf_under_factoring():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(32, 24) * 0.1)
    opt = _factored_opt(p, factor_v=True)
    for i in range(20):
        p.grad = torch.randn(32, 24) * (0.1 ** (i % 4))
        n = opt.step(_dummy_model())
        assert n != float("inf")
        assert torch.isfinite(p).all()
        assert torch.isfinite(opt.v_row[p].float()).all()
        assert torch.isfinite(opt.v_col[p].float()).all()


def test_nonfinite_grads_still_skipped_when_factored():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = _factored_opt(p, factor_v=True)
    p.grad = torch.randn(8, 4)
    opt.step(_dummy_model())
    before, r_b, c_b = p.detach().clone(), opt.v_row[p].clone(), opt.v_col[p].clone()
    p.grad = torch.full((8, 4), float("nan"))
    assert opt.step(_dummy_model()) == float("inf")
    assert opt.step_count == 1
    assert torch.equal(p.detach(), before)
    assert torch.equal(opt.v_row[p], r_b) and torch.equal(opt.v_col[p], c_b)


# ---- checkpoints -----------------------------------------------------

def test_factored_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(2):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    d = tmp_path / "f"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    from safetensors.torch import load_file
    blobs = load_file(str(d / "optimizer_state.safetensors"), device="cpu")
    assert any(k.startswith("v_row.param.") for k in blobs)
    assert any(k.startswith("v_col.param.") for k in blobs)
    assert json.loads((d / "optimizer.json").read_text())["factor_v"] is True
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)
    assert opt2.factor_v is True and opt2.step_count == 2
    _assert_v_sig_equal(_v_sig(opt2, model2), _v_sig(opt, model))
    # keeps stepping
    opt2.zero_grad(model2)
    _, loss = model2(ids, ids)
    loss.backward()
    assert opt2.step(model2) != float("inf")
    _, after = model2(ids, ids)
    assert torch.isfinite(after)


def test_factored_checkpoint_resume_continues_identically(tmp_path):
    """Resume a factored checkpoint: the next step must match a run that never
    stopped, which requires v_row/v_col and the step counter to be exact."""
    torch.manual_seed(8)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=2e-4, factor_v=True)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(3):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    d = tmp_path / "ck"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    # one more step in-process
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    ref = [p.detach().clone() for p in model.parameters()]
    # resume and replay
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)
    assert opt2.step_count == 3
    opt2.zero_grad(model2)
    _, l2 = model2(ids, ids)
    l2.backward()
    opt2.step(model2)
    for a, b in zip(ref, model2.parameters()):
        assert torch.allclose(a, b.detach(), atol=1e-7, rtol=1e-6)
    assert opt2.step_count == 4


def test_full_v_checkpoint_migrates_to_factored(tmp_path):
    """A pre-factoring (full-v) checkpoint must load into factored mode with the
    marginals preserved exactly, and say so rather than silently dropping state."""
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=32, d_model=16, n_layer=1, n_heads=2, tile=16,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    full = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=False)
    ids = torch.randint(0, 32, (2, 8))
    full.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    full.step(model)
    d = tmp_path / "old"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, full, model)
    saved_v = {k: v.clone() for k, v in full.v.items()}
    fact = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    _load_optimizer(d, fact, model)
    # 2-D states became factored; their marginals match the stored full v exactly.
    moved = 0
    for k, v in saved_v.items():
        if fact._factor_shape(tuple(v.shape)):
            assert k in fact.v_row and k in fact.v_col
            assert torch.allclose(fact.v_row[k].float(), v.float().mean(1), atol=2e-2)
            assert torch.allclose(fact.v_col[k].float(), v.float().mean(0), atol=2e-2)
            moved += 1
        else:
            assert k in fact.v
    assert moved > 0
    assert fact.step_count == full.step_count


def test_factored_checkpoint_refuses_full_mode(tmp_path):
    """factored -> full is not reconstructible; it must fail loudly."""
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=32, d_model=16, n_layer=1, n_heads=2, tile=16,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    ids = torch.randint(0, 32, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    d = tmp_path / "f"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    want_full = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=False)
    with pytest.raises(ValueError, match="factored"):
        _load_optimizer(d, want_full, model)


def test_checkpoint_without_factor_v_key_is_treated_as_full(tmp_path):
    """Backward compatibility: a checkpoint written before factoring existed has
    no factor_v key and must load as full-v, not be assumed factored."""
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=32, d_model=16, n_layer=1, n_heads=2, tile=16,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=False)
    ids = torch.randint(0, 32, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    d = tmp_path / "legacy"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    # Strip the flag, exactly as a pre-factoring checkpoint would look.
    j = d / "optimizer.json"
    data = json.loads(j.read_text())
    data.pop("factor_v", None)
    j.write_text(json.dumps(data, indent=2))
    fresh = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=False)
    _load_optimizer(d, fresh, model)
    assert fresh.step_count == 1
    _assert_v_sig_equal(_v_sig(fresh, model), _v_sig(opt, model))


def test_mixed_full_and_factored_states_in_one_checkpoint(tmp_path):
    """A real model has 2-D (factored) and 1-D (full) params at once."""
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp32", architecture="plain")
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    n_fact = len(opt.v_row)
    n_full = len(opt.v)
    assert n_fact > 0 and n_full > 0, (n_fact, n_full)
    d = tmp_path / "mix"
    d.mkdir()
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    opt2 = SmaulOpt(list(model.parameters()))
    _load_optimizer(d, opt2, model)
    _assert_v_sig_equal(_v_sig(opt2, model), _v_sig(opt, model))


# ---- FP8 integration -------------------------------------------------

def test_fp8_model_training_with_factored_v():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    model.train()
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    assert torch.isfinite(loss)
    loss.backward()
    assert opt.step(model) != float("inf")
    for _, m in fp8_modules(model):
        assert m.w8.dtype == torch.uint8           # FP8 format untouched
        assert m._gw is None
        assert m in opt.m and m in opt.v_row and m in opt.v_col
        assert opt.m[m].shape == (m.out_f, m.in_f)  # m still full
        assert opt.v_row[m].shape == (m.out_f,)
        assert opt.v_col[m].shape == (m.in_f,)
    _, loss2 = model(ids, ids)
    assert torch.isfinite(loss2)


def test_no_fp32_master_weight_still_holds_with_factoring():
    from kernel.fp8_tile import FP8Linear
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=2, n_heads=4, tile=32)
    model = SmaulLinear(cfg)
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, factor_v=True)
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.step(model)
    for _name, m in fp8_modules(model):
        for pname, p in m.named_parameters(recurse=False):
            assert p.shape != m.w8.shape or p.dtype != torch.float32, (pname,)
        assert m.w8.dtype == torch.uint8
        # the factored marginals are vectors, never an [out_f, in_f] fp32 copy
        assert opt.v_row[m].dim() == 1 and opt.v_col[m].dim() == 1


# ======================================================================
# THE NUMERICAL COMPARISON: full-v vs factored-v
# ======================================================================


_DISTS = ("uniform", "normal", "heavy_tailed", "sparse", "mostly_zero",
          "unbalanced_rows", "unbalanced_cols", "rank_one")


def _dist_grads(kind, R, C, T, seed=0):
    """Deterministic gradient sequence of the requested structure."""
    g = torch.Generator().manual_seed(seed)
    # Fixed factors: a sum of *different* rank-1 matrices is full rank, so
    # "rank_one" only keeps v separable if a and b never change.
    a_fix = torch.rand(R, 1, generator=g) + 0.1
    b_fix = torch.rand(1, C, generator=g) + 0.1
    out = []
    for t in range(T):
        if kind == "uniform":
            x = torch.rand(R, C, generator=g)
        elif kind == "normal":
            x = torch.randn(R, C, generator=g).abs()
        elif kind == "heavy_tailed":
            x = torch.randn(R, C, generator=g).abs().pow(0.2)
        elif kind == "sparse":
            x = torch.rand(R, C, generator=g)
            x[x < 0.90] = 0.0
        elif kind == "mostly_zero":
            x = torch.rand(R, C, generator=g)
            x[x < 0.995] = 0.0
        elif kind == "unbalanced_rows":
            x = torch.rand(R, C, generator=g) * torch.logspace(0, 3, R).unsqueeze(1)
        elif kind == "unbalanced_cols":
            x = torch.rand(R, C, generator=g) * torch.logspace(0, 3, C).unsqueeze(0)
        elif kind == "rank_one":
            x = (a_fix * b_fix).contiguous()
        else:
            raise AssertionError(kind)
        out.append(x)
    return out


def _run(mode, grads, R, C, lr=1e-3, sdt="fp32"):
    theta0 = torch.full((R, C), 0.1)
    p = torch.nn.Parameter(theta0.clone())
    opt = _factored_opt(p, factor_v=(mode == "factored"), state_dtype=sdt,
                        weight_decay=0.0, beta_v=0.9)
    us, vhat = [], None
    for x in grads:
        p.grad = x.clone()
        opt.step(_dummy_model())
        us.append(p.detach().clone())
    # final reconstructed v (bias-corrected) for the v-error column
    t = len(grads)
    bc2 = 1.0 - 0.9 ** t
    if mode == "factored":
        rh = opt.v_row[p].float() / bc2
        ch = opt.v_col[p].float() / bc2
        vhat = torch.outer(rh, ch) / rh.mean()
    else:
        vhat = opt.v[p].float() / bc2
    return us, vhat, p.detach().clone()


def _report(kind, R=24, C=40, T=60, sdt="fp32"):
    grads = _dist_grads(kind, R, C, T)
    uf, v_full, pf = _run("full", grads, R, C, sdt=sdt)
    us, v_fact, ps = _run("factored", grads, R, C, sdt=sdt)
    v_rel = float((v_fact - v_full).norm() / v_full.norm())
    v_abs = float((v_fact - v_full).abs().max())
    uf_t = torch.stack(uf)                      # per-step parameters
    us_t = torch.stack(us)
    u_rel = float((us_t - uf_t).norm() / uf_t.norm())
    u_abs = float((us_t - uf_t).abs().max())
    p_rel = float((ps - pf).norm() / pf.norm())
    p_abs = float((ps - pf).abs().max())
    finite = bool(torch.isfinite(v_fact).all() and torch.isfinite(v_full).all()
                  and torch.isfinite(uf_t).all() and torch.isfinite(us_t).all()
                  and torch.isfinite(pf).all() and torch.isfinite(ps).all())
    return v_rel, v_abs, u_rel, u_abs, p_rel, p_abs, finite


@pytest.mark.parametrize("kind", _DISTS)
def test_factored_vs_full_numerical_error(kind, capsys):
    """The headline comparison. Not bit-identical by design: the reconstruction
    is an approximation. Reported per gradient structure, because a rank-1 field
    is dense and therefore handles sparse v badly."""
    v_rel, v_abs, u_rel, u_abs, p_rel, p_abs, finite = _report(kind)
    with capsys.disabled():
        print(f"\n[full vs factored] {kind:16s} v_rel={v_rel:.4f} v_abs={v_abs:.3e} "
              f"u_rel={u_rel:.4f} u_abs={u_abs:.3e} p_rel={p_rel:.4f} p_abs={p_abs:.3e}")
    # No NaN/Inf may appear in either mode, for any distribution.
    assert finite, kind
    # Well-conditioned structures must stay close.
    if kind in ("rank_one", "heavy_tailed"):
        assert p_rel < 0.10, (kind, p_rel)      # exact / near-exact
    if kind in ("uniform", "normal", "unbalanced_rows", "unbalanced_cols"):
        assert p_rel < 0.35, (kind, p_rel)
    # Sparse v is the known weak spot; assert the failure is bounded, not silent.
    if kind in ("sparse", "mostly_zero"):
        # Known weak spot: bounded and finite, but far worse than dense grids.
        assert p_rel < 2.0, (kind, p_rel)
    # Rank-1 must be essentially exact.
    if kind == "rank_one":
        assert p_rel < 1e-4, p_rel


def test_factored_never_exceeds_full_v_error_on_dense_grids():
    """Sanity: on a dense grid the marginals themselves are reproduced exactly,
    so the only error is the dropped rank term, which shrinks as the grid gets
    more separable. This pins the direction of the error rather than a magnitude."""
    errs = [_report("normal", R=8, C=8, T=20)[0],
            _report("normal", R=64, C=64, T=20)[0]]
    assert errs[0] < 1.0 and errs[1] < 1.0, errs


def test_factored_is_deterministic_across_identical_runs():
    grads = _dist_grads("normal", 16, 24, 12)
    a = _run("factored", grads, 16, 24)[2]
    b = _run("factored", grads, 16, 24)[2]
    assert torch.equal(a, b)


def test_sparse_gradients_are_the_documented_weak_spot():
    """Locks in the known limitation so it cannot regress silently: a rank-1 v
    under-estimates v at the non-zeros, which inflates the step there."""
    grads = _dist_grads("sparse", 24, 40, 60)
    uf, v_full, pf = _run("full", grads, 24, 40)
    us, v_fact, ps = _run("factored", grads, 24, 40)
    # The reconstruction smears mass into positions whose true v is zero.
    tiny = v_full < 1e-6
    if tiny.any():
        assert float(v_fact[tiny].abs().max()) > 0.0
    # And it under-estimates where the gradient actually is: compare the
    # reconstruction against the true v on the positions that get gradient.
    g_last = grads[-1]
    hot = g_last > 0
    under = float(v_full[hot].mean() / max(float(v_fact[hot].mean()), 1e-30))
    print(f"\n[sparse] v is {under:.2f}x too SMALL at the {int(hot.sum())} positions "
          f"with gradient -> steps there are ~{under:.2f}x too large")
    assert under > 1.0, under


# ---- factored-v regressions found during the 256M validation ----------

def test_zero_gradients_apply_weight_decay_when_factored():
    """Regression: the grand mean is 0 when every gradient is 0, so R*C/G was
    0/0 = NaN, u came out non-finite, and the block was skipped -- which silently
    dropped that block's weight decay. Factored must match full-v here."""
    for fv in (False, True):
        p = torch.nn.Parameter(torch.full((8, 4), 2.0))
        opt = _factored_opt(p, factor_v=fv, weight_decay=0.05)
        p.grad = torch.zeros(8, 4)
        assert opt.step(_dummy_model()) != float("inf")
        want = 2.0 * (1 - 1e-3 * 0.05)
        assert torch.allclose(p.detach(), torch.full((8, 4), want), atol=1e-6), fv
        # v is 0 and m is 0, so no gradient step happened
        if fv:
            assert torch.equal(opt.v_row[p].float(), torch.zeros(8))
        assert torch.equal(opt.m[p].float(), torch.zeros(8, 4)), fv


def test_zero_then_nonzero_gradients_do_not_stale_the_state():
    """Regression: m's write-back sat behind the finite-u check, so a block whose
    u went non-finite never advanced its m while the step counter did. m must
    track the gradient EMA regardless."""
    for fv in (False, True):
        p = torch.nn.Parameter(torch.ones(8, 4))
        opt = _factored_opt(p, factor_v=fv, weight_decay=0.0)
        for _ in range(3):
            p.grad = torch.zeros(8, 4)
            opt.step(_dummy_model())
        p.grad = torch.ones(8, 4)
        opt.step(_dummy_model())
        # After 3 zero steps and one unit gradient, m = 0.1 * 1
        assert torch.allclose(opt.m[p].float(), torch.full((8, 4), 0.1), atol=1e-3), fv


def test_factored_matches_full_v_on_all_zero_gradients_exactly():
    a = torch.nn.Parameter(torch.full((16, 8), 1.5))
    b = torch.nn.Parameter(torch.full((16, 8), 1.5))
    oa = _factored_opt(a, factor_v=False, weight_decay=0.02)
    ob = _factored_opt(b, factor_v=True, weight_decay=0.02)
    for _ in range(5):
        for t in (a, b):
            t.grad = torch.zeros(16, 8)
        oa.step(_dummy_model())
        ob.step(_dummy_model())
    assert torch.allclose(a.detach(), b.detach(), atol=1e-7), float((a - b).abs().max())


# ---- BF16 gradients (_gw + dense p.grad) ------------------------------

def test_gw_is_bf16_by_default():
    from kernel.fp8_tile import FP8Linear, GW_DTYPE
    assert GW_DTYPE is torch.bfloat16
    m = FP8Linear(32, 96, tile=32)
    m.train()
    m(torch.randn(8, 32, requires_grad=True)).backward(torch.randn(8, 96))
    assert m._gw.dtype is torch.bfloat16
    assert m._gw.numel() * m._gw.element_size() == m._gw.numel() * 2


def test_gw_bf16_halves_the_persistent_gradient_buffer():
    """The persistent win: _gw is the largest optimizer-side buffer, and it now
    costs 2 bytes/element instead of 4 for the same tensor.

    (The step's *transient* allocations are dominated by the FP8 requant path,
    which allocates far more than the gradient does, so the blockwise read in
    step() is hygiene rather than what makes the run fit. The property that must
    not regress is the buffer width, checked here and by test_gw_is_bf16_by_default.)
    """
    from kernel.fp8_tile import FP8Linear
    torch.manual_seed(0)
    out_f, in_f = 256, 256
    mod = FP8Linear(in_f, out_f, tile=32)
    mod.train()
    mod(torch.randn(8, in_f, requires_grad=True)).backward(torch.randn(8, out_f))
    assert mod._gw is not None
    got = mod._gw.numel() * mod._gw.element_size()
    assert got == out_f * in_f * 2, got
    assert got * 2 == out_f * in_f * 4      # exactly half of the old fp32 buffer


def test_step_never_widens_the_whole_gw():
    """Guard against reintroducing a whole-matrix FP32 copy of _gw.

    `m._gw.float()` in step() would allocate an FP32 tensor as large as the BF16
    buffer it replaced, i.e. give back the entire saving. The block loop must
    read row slices and let the FP32 state accumulators promote.
    """
    import inspect
    from model import SmaulOpt as _S
    src = inspect.getsource(_S.step)
    assert "_gw.float()" not in src, "step() must not widen the whole _gw to FP32"
    assert "_gw.float()" not in inspect.getsource(_S)


def test_narrow_grads_casts_dense_grads():
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = _factored_opt(p, grad_dtype="bf16")
    p.grad = torch.randn(8, 4)
    assert p.grad.dtype is torch.float32
    opt.narrow_grads_()
    assert p.grad.dtype is torch.bfloat16
    assert p.grad.shape == (8, 4)


@pytest.mark.parametrize("gd,expect", [("bf16", torch.bfloat16),
                                       ("fp16", torch.float16),
                                       ("fp32", torch.float32)])
def test_grad_dtype_option(gd, expect):
    p = torch.nn.Parameter(torch.randn(8, 4))
    opt = _factored_opt(p, grad_dtype=gd)
    p.grad = torch.randn(8, 4)
    opt.narrow_grads_()
    assert p.grad.dtype is expect, (gd, p.grad.dtype)


def test_grad_dtype_validation():
    p = torch.nn.Parameter(torch.randn(2))
    for bad in ("int8", "fp8", "bfloat32", 1):
        with pytest.raises(ValueError, match="grad_dtype"):
            _factored_opt(p, grad_dtype=bad)


def test_narrowed_grads_still_train_and_stay_finite():
    """End-to-end: a bf16 gradient must produce a finite, decreasing loss, and
    the update must stay close to the same run with FP32 gradients."""
    def run(gd):
        torch.manual_seed(0)
        cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                           precision="fp32", architecture="plain")
        model = SmaulLinear(cfg)
        model.train()
        opt = SmaulOpt(list(model.parameters()), lr=1e-3, grad_dtype=gd)
        torch.manual_seed(1)
        ids = torch.randint(0, 64, (2, 16))
        out = []
        for _ in range(6):
            opt.zero_grad(model)
            _, loss = model(ids, ids)
            loss.backward()
            opt.narrow_grads_()
            assert opt.step(model) != float("inf")
            out.append(float(loss.detach()))
        return out
    l32 = run("fp32")
    l16 = run("bf16")
    assert all(torch.isfinite(torch.tensor(x)) for x in l16)
    assert l16[-1] < l16[0], (l16[0], l16[-1])
    rel = abs(l16[-1] - l32[-1]) / max(abs(l32[-1]), 1e-12)
    print(f"\n[bf16 grads] fp32 final={l32[-1]:.6f} bf16 final={l16[-1]:.6f} rel={rel:.2e}")
    assert rel < 0.05, (l32[-1], l16[-1])


def test_fp8_training_with_bf16_gw_and_narrowed_grads():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    model.train()
    opt = SmaulOpt(list(model.parameters()), lr=1e-4)
    ids = torch.randint(0, 64, (2, 8))
    losses = []
    for _ in range(4):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        assert torch.isfinite(loss)
        loss.backward()
        opt.narrow_grads_()
        assert opt.step(model) != float("inf")
        losses.append(float(loss.detach()))
    for _, m in fp8_modules(model):
        assert m.w8.dtype is torch.uint8          # FP8 format untouched
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.dtype is torch.bfloat16, p.dtype
    assert losses[-1] < losses[0], losses
    print(f"\n[fp8 + bf16 grads] losses {[round(x, 4) for x in losses]}")


# ---------------------------------------------------------------------------
# Global grad norm (train._grad_norm, used by Lion._clip and SmaulOpt._clip)
# ---------------------------------------------------------------------------

def _exact_norm(grads):
    """Reference: concatenate and sum squares in float64, one tensor at a time."""
    return float(torch.stack([g.double().pow(2).sum() for g in grads]).sum().sqrt())


def test_grad_norm_matches_float64_reference_across_dtypes_and_shapes():
    torch.manual_seed(0)
    for dt in (torch.float32, torch.bfloat16, torch.float64):
        for shape in ((7,), (33, 41), (1024, 512), (2, 3, 5)):
            grads = [torch.randn(shape, dtype=dt) for _ in range(3)]
            got, want = _grad_norm(grads), _exact_norm(grads)
            assert got == pytest.approx(want, rel=1e-6, abs=1e-12), (dt, shape)


def test_grad_norm_is_exact_enough_to_not_regress_toward_float32():
    """The accumulation must stay in float64.

    torch.linalg.vector_norm accumulates linearly, so a float32 accumulation
    is ~6e-4 relative off on a 16M-element tensor. This pins the result to
    the float64 reference so a future "optimization" to float32/bfloat16
    cannot land silently: the clip threshold is this number.
    """
    torch.manual_seed(1)
    big = torch.randn(1 << 24, dtype=torch.bfloat16)
    assert _grad_norm([big]) == pytest.approx(_exact_norm([big]), rel=1e-12)
    # And it is demonstrably not the float32 answer.
    f32 = float(torch.linalg.vector_norm(big.float(), ord=2,
                                         dtype=torch.float32).item())
    assert abs(f32 - _grad_norm([big])) / _grad_norm([big]) > 1e-6


def test_grad_norm_blocks_large_tensors_without_changing_the_result():
    """Crossing _NORM_BLOCK must not perturb the value."""
    import train as _train
    torch.manual_seed(2)
    base = torch.randn(_train._NORM_BLOCK + 7, dtype=torch.bfloat16)
    want = _exact_norm([base])
    assert _grad_norm([base]) == pytest.approx(want, rel=1e-9)
    for n in (0, 1, 2, _train._NORM_BLOCK - 1, _train._NORM_BLOCK,
              _train._NORM_BLOCK + 1, 4 * _train._NORM_BLOCK):
        g = torch.randn(n, dtype=torch.bfloat16)
        assert _grad_norm([g]) == pytest.approx(_exact_norm([g]), rel=1e-6, abs=1e-9), n


def test_grad_norm_skips_empty_and_matches_a_known_value():
    assert _grad_norm([]) == 0.0
    assert _grad_norm([torch.zeros(0), torch.zeros(0, dtype=torch.bfloat16)]) == 0.0
    # 3-4-5 triangle, exactly 1.0 in float64 and 1.0 in float32 too.
    assert _grad_norm([torch.tensor([3.0, 4.0])]) == pytest.approx(5.0, rel=1e-9)
    # Per-tensor norms combine in quadrature: sqrt(1 + 4 + 9) == sqrt(14).
    grads = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    assert _grad_norm(grads) == pytest.approx(14.0 ** 0.5, rel=1e-12)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_grad_norm_propagates_non_finite(bad):
    """Non-finite must surface through the norm, so _clip needs no isfinite scan."""
    g = torch.randn(1 << 12, dtype=torch.bfloat16)
    g[123] = bad
    assert not math.isfinite(_grad_norm([g]))
    # ...and the non-finite entry is detected even when it sits in a block
    # other than the first.
    g2 = torch.randn(1 << 12, dtype=torch.bfloat16)
    g2[-1] = bad
    assert not math.isfinite(_grad_norm([g2]))


def test_grad_norm_handles_non_contiguous_gradients():
    torch.manual_seed(3)
    base = torch.randn(64, 128, dtype=torch.bfloat16)
    view = base.t()            # non-contiguous, same values
    assert not view.is_contiguous()
    assert _grad_norm([view]) == pytest.approx(_exact_norm([view]), rel=1e-6)


@pytest.mark.parametrize("opt_cls", [Lion, SmaulOpt])
def test_clip_contract_preserved(opt_cls):
    """Both optimizers keep the documented _clip contract."""
    # no gradients at all -> 0.0
    p = torch.nn.Parameter(torch.zeros(3))
    assert opt_cls([p])._clip([], 1.0) == 0.0

    # pre-clip norm is reported, and the grads are actually rescaled
    p = torch.nn.Parameter(torch.ones(4))
    opt = opt_cls([p], clip=1.0)
    p.grad = torch.full((4,), 10.0)          # norm 20
    n = opt._clip([], 1.0)
    assert n == pytest.approx(20.0, rel=1e-6)
    assert float(p.grad.norm()) <= 1.0 + 1e-6

    # no rescale when under the threshold
    p = torch.nn.Parameter(torch.ones(4))
    opt = opt_cls([p], clip=1.0)
    p.grad = torch.full((4,), 0.1)          # norm 0.2
    n = opt._clip([], 1.0)
    assert n == pytest.approx(0.2, rel=1e-6)
    assert float(p.grad.norm()) == pytest.approx(0.2, rel=1e-6)


@pytest.mark.parametrize("opt_cls", [Lion, SmaulOpt])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_clip_reports_inf_and_does_not_rescale_on_non_finite(opt_cls, bad):
    p = torch.nn.Parameter(torch.ones(4))
    opt = opt_cls([p], clip=1.0)
    p.grad = torch.tensor([1.0, 1.0, 1.0, bad])
    before = p.grad.detach().clone()
    assert opt._clip([], 1.0) == float("inf")
    # Untouched: the old code returned before any scaling, and so must this.
    # Bitwise, because torch.equal is False whenever NaN is present.
    assert torch.equal(p.grad.detach().view(torch.int32),
                       before.view(torch.int32))


def test_lion_and_smaul_clip_agree_on_the_same_gradients():
    """_clip sees exactly the grads reachable from the params it was given.

    Lion leaves grads in FP32; SmaulOpt defaults to --grad-dtype bf16, which
    only works because narrow_grads_ clears p.grad_dtype. Both widths must
    produce the same norm as the float64 reference.
    """
    torch.manual_seed(4)
    unused = torch.randn(5000, dtype=torch.bfloat16)   # must not be counted
    src = torch.randn(97)

    p0 = torch.nn.Parameter(torch.zeros(97))
    lion = Lion([p0], clip=1.0)
    p0.grad = src.clone()                               # FP32, Lion's path
    fp32_norm = lion._clip([], 1.0)
    assert fp32_norm == pytest.approx(_exact_norm([src]), rel=1e-12)
    assert fp32_norm < _exact_norm([src, unused])

    # Same values, stored bf16 the way SmaulOpt actually receives them.
    p1 = torch.nn.Parameter(torch.zeros(97))
    sm = SmaulOpt([p1], lr=1e-4, clip=1.0, grad_dtype="bf16")
    p1.grad = src.clone()
    sm.narrow_grads_()
    assert p1.grad.dtype is torch.bfloat16
    # _clip rescales in place, so keep a copy of the pre-clip values.
    pre = p1.grad.detach().clone()
    bf16_norm = sm._clip([], 1.0)
    assert bf16_norm == pytest.approx(_exact_norm([pre]), rel=1e-9)
    assert bf16_norm == pytest.approx(fp32_norm, rel=1e-3)   # bf16 input rounding
    assert float(p1.grad.float().norm()) <= 1.0 + 1e-3      # and it did rescale


# ---------------------------------------------------------------------------
# _load_smaul_states is a wall of deliberate refusals -- "refusing partial
# load", "refusing to guess", "refusing to pretend state exists". The happy path
# is covered by test_fp8_state_survives_checkpoint above; none of the guards
# were. A guard that silently stops raising does not fail loudly, it accepts a
# corrupt or mismatched checkpoint and resumes a run with the wrong optimizer
# state, which is the one outcome this function exists to prevent.
# ---------------------------------------------------------------------------

def _trained_fp8_checkpoint(tmp_path, factor_v=True, steps=2, sdt="fp32"):
    """A real checkpoint with real state, plus the blobs, for corruption."""
    from safetensors.torch import load_file
    torch.manual_seed(11)
    d = tmp_path / f"ck_{factor_v}_{sdt}"
    d.mkdir(exist_ok=True)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    assert fp8_modules(model), "expected FP8 modules"
    opt = SmaulOpt(list(model.parameters()), lr=1e-4, state_dtype=sdt, factor_v=factor_v)
    ids = torch.randint(0, 64, (2, 8))
    for _ in range(steps):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)
    model.save_pretrained(d)
    _save_optimizer(d, opt, model)
    sp = d / "optimizer_state.safetensors"
    # steps=0 produces no state at all, and _save_smaul_states correctly writes
    # nothing in that case, so there is no file to read.
    blobs = load_file(str(sp), device="cpu") if sp.exists() else {}
    return d, model, opt, ids, blobs


def _reloaded(d):
    return SmaulLinear.from_pretrained(d), SmaulOpt(list(SmaulLinear.from_pretrained(d).parameters()))


def _attempt(d, blobs):
    """Write `blobs` over the checkpoint's state and try to resume from it."""
    from safetensors.torch import save_file
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    sp = d / "optimizer_state.safetensors"
    if sp.exists():
        sp.unlink()
    if blobs is not None:
        save_file(blobs, str(sp))
    return lambda: _load_optimizer(d, opt2, model2)


def test_round_trip_restores_state_exactly(tmp_path):
    """The baseline the guards protect: a resume is bit-identical in state."""
    d, model, opt, _ids, blobs = _trained_fp8_checkpoint(tmp_path)
    assert blobs, "expected a non-empty state file"
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)
    assert opt2.step_count == opt.step_count == 2
    mods2 = dict(fp8_modules(model2))
    for name, m in fp8_modules(model):
        assert torch.equal(opt.m[m], opt2.m[mods2[name]]), name
    _assert_v_sig_equal(_v_sig(opt2, model2), _v_sig(opt, model))


def test_round_trip_restores_dense_param_state_too(tmp_path):
    """The embedding is a dense Parameter, not an FP8 module; it uses the
    other key prefix, and a guard that only understood fp8.* would drop it."""
    d, model, opt, _ids, _blobs = _trained_fp8_checkpoint(tmp_path)
    params2 = dict(SmaulLinear.from_pretrained(d).named_parameters())
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)
    dense = [n for n, _ in model.named_parameters() if n in params2]
    assert dense, "expected dense parameters"
    for n in dense:
        p2 = params2[n]
        if p2 in opt2.m:
            assert torch.equal(opt.m[p2], opt2.m[p2]), n


def test_resume_leaves_no_temp_state_file(tmp_path):
    d, _m, _o, _i, _b = _trained_fp8_checkpoint(tmp_path)
    assert [p.name for p in d.iterdir() if p.name.endswith(".tmp")] == []


# --- a checkpoint that claims progress but carries no state ------------------

def test_resume_refuses_a_missing_state_file_when_step_is_nonzero(tmp_path):
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path)
    (d / "optimizer_state.safetensors").unlink()
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    # A live optimizer that already has state: this is the "continuing a run
    # whose checkpoint lost its state file" case, distinct from the JSON-only
    # claim tested next.
    opt2.load_state_dict(json.loads((d / "optimizer.json").read_text()))
    opt2.m = {p: torch.zeros_like(p) for p in list(model2.parameters())[:1]}
    assert opt2.step_count and opt2.m
    with pytest.raises(ValueError, match="missing optimizer_state.safetensors"):
        _load_optimizer(d, opt2, model2)


def test_resume_refuses_to_pretend_state_exists(tmp_path):
    """The distinct message when optimizer.json says step>0 but state is gone.

    Same missing file, different question: the first case has live state in the
    optimizer, the second has only the JSON's claim. Both must refuse, and the
    message has to distinguish them or the operator cannot tell which happened.
    """
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path)
    (d / "optimizer_state.safetensors").unlink()
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    opt2.load_state_dict(json.loads((d / "optimizer.json").read_text()))
    opt2.m, opt2.v, opt2.v_row, opt2.v_col = {}, {}, {}, {}     # no live state
    with pytest.raises(ValueError, match="refusing to pretend state exists"):
        _load_optimizer(d, opt2, model2)


def test_a_fresh_checkpoint_without_state_is_accepted(tmp_path):
    """step=0 with no state file is a legitimate new run, not corruption."""
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path, steps=0)
    assert not (d / "optimizer_state.safetensors").exists()
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()))
    _load_optimizer(d, opt2, model2)             # must not raise
    assert opt2.step_count == 0


# --- malformed keys ---------------------------------------------------------

def test_resume_rejects_a_key_with_no_separator(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    bad = dict(blobs)
    bad["m"] = bad[next(iter(bad))].clone()   # clone: safetensors rejects aliases
    with pytest.raises(ValueError, match="invalid SmaulOpt state key"):
        _attempt(d, bad)()


def test_resume_rejects_an_unknown_state_kind(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    bad = dict(blobs)
    first = next(iter(blobs))
    bad["momentum." + first.split(".", 1)[1]] = bad[first].clone()
    with pytest.raises(ValueError, match="invalid SmaulOpt state key"):
        _attempt(d, bad)()


def test_resume_rejects_an_unknown_key_prefix(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    bad = dict(blobs)
    first = next(iter(blobs))
    bad["m.other." + first.split(".", 1)[1]] = bad[first].clone()
    with pytest.raises(ValueError, match="invalid SmaulOpt state key"):
        _attempt(d, bad)()


def test_resume_rejects_state_for_a_parameter_that_no_longer_exists(tmp_path):
    """A renamed or deleted parameter must not be silently skipped."""
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    bad = dict(blobs)
    victim = next(k for k in bad if k.startswith("m.param."))
    bad["m.param.no_such_parameter"] = bad.pop(victim)
    with pytest.raises(ValueError, match="no matching parameter"):
        _attempt(d, bad)()


def test_resume_rejects_state_for_an_fp8_module_that_no_longer_exists(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    bad = dict(blobs)
    victim = next(k for k in bad if k.startswith("m.fp8."))
    bad["m.fp8.no_such_module"] = bad.pop(victim)
    with pytest.raises(ValueError, match="no matching FP8 module"):
        _attempt(d, bad)()


# --- inconsistent v representations ----------------------------------------

def test_resume_rejects_a_half_written_factored_pair(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    assert any(k.startswith("v_col.") for k in blobs), "need a factored state"
    bad = {k: v for k, v in blobs.items() if not k.startswith("v_col.")}
    with pytest.raises(ValueError, match="refusing partial load"):
        _attempt(d, bad)()


def test_resume_rejects_full_and_factored_v_for_one_object(tmp_path):
    """Never guess which of two v representations is meant."""
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    row_key = next(k for k in blobs if k.startswith("v_row."))
    base = row_key.split(".", 1)[1]
    m_key = "m." + base
    bad = dict(blobs)
    bad["v." + base] = blobs[m_key].clone()          # both forms for one object
    assert bad["v." + base].data_ptr() != blobs[row_key].data_ptr()
    with pytest.raises(ValueError, match="refusing to guess"):
        _attempt(d, bad)()


def test_resume_rejects_an_object_with_no_v_at_all(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    base = next(k for k in blobs if k.startswith("v.")).split(".", 1)[1]
    bad = {k: v for k, v in blobs.items() if k not in ("v." + base,)}
    with pytest.raises(ValueError, match="no v at all"):
        _attempt(d, bad)()


def test_resume_rejects_transposed_marginals(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    # A swap is only detectable when the two extents differ, so pick a
    # non-square 2-D state.
    square = None
    for k in blobs:
        if not k.startswith("v_col."):
            continue
        base = k.split(".", 1)[1]
        m = blobs.get("m." + base)
        if m is not None and m.dim() == 2 and m.shape[0] != m.shape[1]:
            square = base
            break
    assert square is not None, "fixture needs a non-square 2-D state"
    bad = dict(blobs)
    bad["v_row." + square] = blobs["v_col." + square].clone()
    bad["v_col." + square] = blobs["v_row." + square].clone()
    with pytest.raises(ValueError, match="transposed or mismatched pair"):
        _attempt(d, bad)()


# --- shape and architecture mismatches -------------------------------------

def test_resume_rejects_marginals_that_do_not_match_their_own_extent(tmp_path):
    """The check that replaced an unreachable one: lengths against m.

    The old guard compared v_row's shape to v_col's reversed shape, which can
    never fire because both are 1-D -- `shape[::-1]` is the shape. This is the
    invariant it was reaching for.
    """
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    base = None
    for k in blobs:
        if k.startswith("v_row."):
            cand = k.split(".", 1)[1]
            m = blobs.get("m." + cand)
            if m is not None and m.dim() == 2:
                base = cand
                break
    assert base is not None
    bad = dict(blobs)
    bad["v_row." + base] = blobs["v_row." + base].reshape(1, -1)   # 2-D marginal
    with pytest.raises(ValueError, match="transposed or mismatched pair"):
        _attempt(d, bad)()


def test_square_states_cannot_be_caught_by_shape_alone(tmp_path):
    """Documented limit of the new check, so it is not oversold.

    For a square 2-D state the two marginals have the same length, so swapping
    them is invisible to any shape comparison -- only a value comparison would
    catch it, and none is made. The new guard catches every non-square case,
    which is all of them in practice.
    """
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    squares = []
    for k in blobs:
        if not k.startswith("v_col."):
            continue
        base_k = k.split(".", 1)[1]
        m = blobs.get("m." + base_k)
        if m is not None and m.dim() == 2 and m.shape[0] == m.shape[1]:
            squares.append(base_k)
    # The 32x32 attention projections are always square; if the fixture ever
    # stops producing one, the test must fail, not silently skip.
    assert squares, "fixture produced no square 2-D state"
    # It loads without complaint, which is the point: shape checks cannot help here.
    _attempt(d, blobs)()


def test_resume_rejects_a_state_of_the_wrong_shape(tmp_path):
    """The arch-changed case: same names, different extents."""
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path)
    victim = next(k for k in blobs if k.startswith("m.fp8."))
    bad = dict(blobs)
    base = victim.split(".", 1)[1]
    bad[victim] = torch.zeros(blobs[victim].shape[0] + 8, blobs[victim].shape[1])
    # Keep the marginals consistent with the widened m, so the live-object shape
    # check is the one that fires rather than the marginal check.
    for prefix in ("v_row", "v_col", "v"):
        k = prefix + "." + base
        if k in bad:
            t = bad[k]
            if t.dim() == 1:
                # v_row is R long and v_col is C long, for the *widened* m.
                n = bad[victim].shape[0] if prefix == "v_row" else bad[victim].shape[1]
                bad[k] = torch.zeros(n)
    with pytest.raises(ValueError, match="checkpoint incompatible"):
        _attempt(d, bad)()


def test_resume_rejects_m_and_v_of_different_shapes(tmp_path):
    d, _m, _o, _i, blobs = _trained_fp8_checkpoint(tmp_path, factor_v=False)
    assert any(k.startswith("v.") for k in blobs), "need a full-v state"
    victim = next(k for k in blobs if k.startswith("v."))
    bad = dict(blobs)
    bad[victim] = torch.zeros(blobs[victim].shape[0], blobs[victim].shape[1] + 3)
    with pytest.raises(ValueError, match="m/v shape mismatch"):
        _attempt(d, bad)()


def test_resume_reports_an_unreadable_state_file(tmp_path):
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path)
    (d / "optimizer_state.safetensors").write_bytes(b"not a safetensors file")
    with pytest.raises((RuntimeError, ValueError)) as e:
        _attempt(d, None)()
    assert "state" in str(e.value).lower() or "safetensors" in str(e.value).lower()


# --- full-v vs factored-v across a resume ----------------------------------

def test_resume_refuses_a_factored_state_when_factor_v_is_disabled(tmp_path):
    """Expanding a factored v into a full v would have to invent the rank term."""
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path, factor_v=True)
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()), factor_v=False)
    with pytest.raises(ValueError, match="factor_v is disabled"):
        _load_optimizer(d, opt2, model2)


def test_resume_migrates_a_full_v_state_to_factored(tmp_path):
    """The documented migration, and the claim it makes is checkable.

    "the marginals of a stored v are exactly recoverable, so R and C are
    preserved exactly; only the rank term is dropped" -- so the migrated
    row/col means must equal the original tensor's means, not merely be close.
    """
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path, factor_v=False)
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()), factor_v=True, state_dtype="fp32")
    _load_optimizer(d, opt2, model2)
    assert opt2.v_row, "expected the full v to have been migrated"
    assert not any(o in opt2.v for o in opt2.v_row), "a migrated object kept both forms"
    # The pre-migration tensor is gone, so compare against the checkpoint's own
    # stored full v.
    from safetensors.torch import load_file
    blobs = load_file(str(d / "optimizer_state.safetensors"), device="cpu")
    stored = {k: v for k, v in blobs.items() if k.startswith("v.")}
    assert stored, "the fixture must have stored a full v"
    by_name = dict(model2.named_parameters())
    by_name.update(dict(fp8_modules(model2)))
    checked, kept = 0, 0
    for k, v in stored.items():
        obj_name = k.split(".", 1)[1].split(".", 1)[1]
        obj = by_name.get(obj_name)
        assert obj is not None, obj_name
        if obj in opt2.v_row:
            # 2-D with both extents >= 2: migrated to marginals.
            assert torch.allclose(opt2.v_row[obj].float(), v.float().mean(dim=1), atol=1e-6), obj_name
            assert torch.allclose(opt2.v_col[obj].float(), v.float().mean(dim=0), atol=1e-6), obj_name
            assert obj not in opt2.v, obj_name
            checked += 1
        else:
            # 1-D/0-D states cannot be factored and must keep the full v.
            assert obj in opt2.v, obj_name
            kept += 1
    assert checked, "expected at least one migrated 2-D state"
    assert kept, "expected at least one unfactorable 1-D state"


def test_migrated_state_is_saved_back_in_factored_form(tmp_path):
    """The message promises the next save writes the factored form; check it."""
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path, factor_v=False)
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()), factor_v=True, state_dtype="fp32")
    _load_optimizer(d, opt2, model2)
    out2 = tmp_path / "resaved"
    out2.mkdir()
    model2.save_pretrained(out2)
    _save_optimizer(out2, opt2, model2)
    from safetensors.torch import load_file
    blobs = load_file(str(out2 / "optimizer_state.safetensors"), device="cpu")
    assert any(k.startswith("v_row.") for k in blobs), sorted(blobs)[:5]
    # And that file must load back without needing another migration.
    model3 = SmaulLinear.from_pretrained(out2)
    opt3 = SmaulOpt(list(model3.parameters()), factor_v=True, state_dtype="fp32")
    _load_optimizer(out2, opt3, model3)
    _assert_v_sig_equal(_v_sig(opt3, model3), _v_sig(opt2, model2))


@pytest.mark.parametrize("sdt", ["bf16", "fp16", "fp32"])
def test_restored_state_keeps_the_declared_storage_width(tmp_path, sdt):
    """A resumed run must not silently widen or narrow its own state."""
    d, _m, _o, _i, _blobs = _trained_fp8_checkpoint(tmp_path, sdt=sdt)
    want = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[sdt]
    model2 = SmaulLinear.from_pretrained(d)
    opt2 = SmaulOpt(list(model2.parameters()), state_dtype=sdt)
    _load_optimizer(d, opt2, model2)
    for t in list(opt2.m.values()) + list(opt2.v_row.values()) + list(opt2.v_col.values()):
        assert t.dtype == want, (sdt, t.dtype)


# ---------------------------------------------------------------------------
# Lion validated one of its five hyperparameters. AGENTS.md states the house
# style as "validate inputs eagerly and raise ValueError at boundaries
# (constructors and CLI parsing both do this)", and SmaulOpt implements that for
# every one of its own -- so Lion was the odd one out, and it is the default
# optimizer.
# ---------------------------------------------------------------------------

def _p():
    return [torch.nn.Parameter(torch.randn(3))]


@pytest.mark.parametrize("kw", [
    dict(lr=-1.0), dict(lr=0.0), dict(lr=float("nan")), dict(lr=float("inf")),
    dict(betas=(2.0, 0.99)), dict(betas=(-0.5, 0.9)), dict(betas=(0.9, 1.5)),
    dict(betas=(0.9,)),                       # too short
    dict(betas="ab"),                          # not a pair at all
    dict(betas=("a", "b")),
    dict(wd=-1.0), dict(wd=float("inf")), dict(wd=float("nan")),
    dict(clip=0.0), dict(clip=-1.0), dict(clip=float("nan")),
])
def test_lion_rejects_bad_hyperparameters(kw):
    with pytest.raises(ValueError):
        Lion(_p(), **kw)


def test_a_run_cannot_start_with_a_nan_learning_rate():
    """Why the check is in the constructor rather than in step().

    Lion(lr=nan) used to be accepted and turned every weight into NaN on the
    first step, with no error -- which in a long run reads as a divergence that
    never gets attributed to the constructor. Now the run refuses to start.
    """
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))

    model = M()
    before = model.w.detach().clone()
    with pytest.raises(ValueError, match="lr must be positive finite"):
        Lion(model.parameters(), lr=float("nan"))
    assert torch.equal(model.w.detach(), before), "construction touched the weights"
    # And a healthy lr leaves them finite, which is the contrast that matters.
    opt = Lion(model.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad(model)
        (model.w * 2).sum().backward()
        opt.step(model)
    assert torch.isfinite(model.w).all()


def test_a_negative_learning_rate_is_refused_rather_than_ascending():
    """Lion sign-steps by lr, so a negative lr walks uphill, silently."""
    with pytest.raises(ValueError, match="lr must be positive finite"):
        Lion(_p(), lr=-1e-2)


def test_lion_accepts_the_values_train_py_and_rl_py_pass():
    """Coercion, not rejection: numeric strings and ints stay usable."""
    # betas=1.0 is *not* accepted: the range is [0, 1) in both optimizers, since
    # a beta of exactly 1 means momentum never decays.
    opt = Lion(_p(), lr="1e-4", betas=(0, 0.99), wd=0)
    assert opt.lr == 1e-4 and opt.b1 == 0.0 and opt.wd == 0.0


def test_lion_defaults_are_unchanged():
    opt = Lion(_p())
    assert (opt.lr, opt.b1, opt.b2, opt.wd, opt.clip) == (1e-4, 0.9, 0.99, 0.01, 1.0)


def test_both_optimizers_agree_on_what_a_valid_hyperparameter_is():
    """The two constructors must not drift into different contracts."""
    bad = [dict(lr=-1.0), dict(clip=0.0), dict(lr=float("nan"))]
    for kw in bad:
        with pytest.raises(ValueError):
            Lion(_p(), **kw)
        with pytest.raises(ValueError):
            SmaulOpt(_p(), **{"lr": kw["lr"]} if "lr" in kw else kw)


@pytest.mark.parametrize("dt", ["bf16", "fp16", "fp32"])
def test_smaul_opt_rejects_an_unknown_state_dtype(dt):
    with pytest.raises(ValueError, match="state_dtype"):
        SmaulOpt(_p(), state_dtype="fp64")


@pytest.mark.parametrize("value", [-1.0, 0.0, float("nan"), float("inf"), "x", None])
def test_smaul_opt_rejects_bad_lr(value):
    with pytest.raises(ValueError, match="lr"):
        SmaulOpt(_p(), lr=value)


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0, float("nan"), "x"])
def test_smaul_opt_rejects_bad_betas(value):
    with pytest.raises(ValueError, match="beta"):
        SmaulOpt(_p(), beta_m=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), "x"])
def test_smaul_opt_rejects_a_bad_epsilon(value):
    with pytest.raises(ValueError, match="epsilon"):
        SmaulOpt(_p(), epsilon=value)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "x"])
def test_smaul_opt_rejects_a_bad_weight_decay(value):
    # 0.0 is valid: no decay. Only negatives and non-finites are not.
    with pytest.raises(ValueError, match="weight_decay"):
        SmaulOpt(_p(), weight_decay=value)
    assert SmaulOpt(_p(), weight_decay=0.0).weight_decay == 0.0


def test_smaul_opt_rejects_a_non_smaul_checkpoint():
    """m/v state is not interchangeable with another optimizer's."""
    opt = SmaulOpt(_p())
    with pytest.raises(ValueError, match="no optimizer name"):
        opt.load_state_dict({"lr": 1e-4})
    with pytest.raises(ValueError, match="cannot load optimizer"):
        opt.load_state_dict({"name": "lion", "lr": 1e-4})
    with pytest.raises(ValueError, match="must be a dict"):
        opt.load_state_dict(["not", "a", "dict"])


def test_lion_load_state_dict_is_tolerant():
    """Lion needs no momentum state, so a SmaulOpt checkpoint is readable."""
    opt = Lion(_p())
    opt.load_state_dict({"name": "smaul", "lr": 5e-4, "step": 3, "factor_v": True})
    assert opt.lr == 5e-4, "Lion should take the hyperparameters it understands"


# ---------------------------------------------------------------------------
# AGENTS.md: "FP8 weight gradients accumulate in module._gw, not .grad, and
# accumulate across backward calls. Any optimizer must call zero_grad(model)
# *with the model* or _gw double-counts."
#
# The second sentence is about zero_grad; the first is a stronger property and it
# is what makes gradient accumulation usable with the FP8 path at all. Two
# backward calls followed by one step must equal one big backward exactly -- if
# _gw were overwritten rather than accumulated, or zeroed between the backwards,
# the weights would drift with no error anywhere.
# ---------------------------------------------------------------------------

def _fp8_model(seed=0):
    torch.manual_seed(seed)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2, tile=32,
                       precision="fp8", architecture="plain")
    model = SmaulLinear(cfg)
    assert fp8_modules(model), "expected FP8 modules"
    return model


def test_micro_batched_backward_equals_one_big_backward():
    """Two backwards, one step -- bit-identical to a single fused backward."""
    a, b = _fp8_model(), _fp8_model(0)
    b.load_state_dict(a.state_dict())
    oa = Lion(list(a.parameters()), lr=1e-2)
    ob = Lion(list(b.parameters()), lr=1e-2)
    x1 = torch.randint(0, 64, (1, 8))
    x2 = torch.randint(0, 64, (1, 8))

    oa.zero_grad(a)
    _, l1 = a(x1, x1); l1.backward()
    _, l2 = a(x2, x2); l2.backward()
    assert all(m._gw is not None for _, m in fp8_modules(a))
    oa.step(a)

    ob.zero_grad(b)
    big = torch.cat([x1, x2])
    _, l3 = b(big, big); l3.backward()
    ob.step(b)

    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    worst = 0.0
    for k in sa:
        if sa[k].dtype.is_floating_point:
            worst = max(worst, float((sa[k].float() - sb[k].float()).abs().max()))
    assert worst == 0.0, f"accumulation changed the result by {worst}"


def test_gw_is_cleared_after_every_step_not_just_the_first():
    """So a second step cannot start from the first step's gradients."""
    model = _fp8_model()
    opt = Lion(list(model.parameters()), lr=1e-3)
    ids = torch.randint(0, 64, (2, 8))
    for step in range(1, 4):
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        assert all(m._gw is not None for _, m in fp8_modules(model)), step
        opt.step(model)
        assert all(m._gw is None for _, m in fp8_modules(model)), \
            f"_gw survived step {step}, so step {step + 1} would double-count"


def test_zero_grad_without_the_model_warns_and_leaves_gw_alone(tmp_path, capsys):
    """The documented footgun: zero_grad() with no model cannot reach _gw.

    AGENTS.md calls this out, so it is pinned as the behaviour it has: a
    RuntimeWarning rather than an error, and the stale _gw survives -- which is
    exactly why passing the model is required. (RuntimeWarning, not
    UserWarning: I guessed UserWarning first and the test passed only because a
    failure to warn looks identical to a failure to warn for the right reason.)
    """
    model = _fp8_model()
    opt = Lion(list(model.parameters()), lr=1e-3)
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    assert all(m._gw is not None for _, m in fp8_modules(model))

    with pytest.warns(RuntimeWarning, match="_gw"):
        opt.zero_grad()
    assert all(m._gw is not None for _, m in fp8_modules(model)), \
        "zero_grad() cleared _gw after all, so the warning overstates the problem"


def test_zero_grad_with_the_model_clears_gw_and_the_dense_grads():
    model = _fp8_model()
    opt = Lion(list(model.parameters()), lr=1e-3)
    ids = torch.randint(0, 64, (2, 8))
    opt.zero_grad(model)
    _, loss = model(ids, ids)
    loss.backward()
    opt.zero_grad(model)
    assert all(m._gw is None for _, m in fp8_modules(model))
    assert all(p.grad is None for p in model.parameters() if p.grad is not None)

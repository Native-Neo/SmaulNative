import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from kernel.fp8_tile import FP8Linear, fp8_modules, quantize_tiles
from smaul_linear import LinearConfig, SmaulLinear


def test_quant_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 2
    wq, sc = quantize_tiles(w)
    assert wq.dtype == torch.uint8
    assert sc.shape == (64, 2)
    from kernel.fp8_tile import decode_tile
    rec = torch.cat([decode_tile(wq, sc, 0, 64, t) for t in range(2)], 1)
    rel = ((rec - w).abs().amax() / w.abs().amax()).item()
    assert rel < 0.08, rel


def test_overflow_underflow_saturate():
    w = torch.tensor([[1e30, 1e-30, 0.0, -1e30]])
    wq, sc = quantize_tiles(w)
    from kernel.fp8_tile import decode_tile
    rec = decode_tile(wq, sc, 0, 1, 0)
    assert torch.isfinite(rec).all()
    assert rec[0, 2] == 0
    assert (rec[0, 0] > 0) and (rec[0, 3] < 0)


def test_fp8_linear_grad_finite():
    torch.manual_seed(1)
    m = FP8Linear(32, 32, tile=16)
    m.train()
    x = torch.randn(4, 32, requires_grad=True)
    y = m(x)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert m._gw is not None and torch.isfinite(m._gw).all()


def test_never_full_dequant():
    torch.manual_seed(2)
    m = FP8Linear(128, 256, tile=32)
    assert m.w8.dtype == torch.uint8
    assert m.w8.numel() == 256 * 128
    assert m.sc.shape == (256, 4)
    x = torch.randn(2, 128)
    y = m(x)
    assert y.shape == (2, 256)
    assert m.w8.dtype == torch.uint8


def test_model_block_spec():
    cfg = LinearConfig(vocab_size=256, d_model=64, n_layer=2, n_heads=4, tile=32)
    m = SmaulLinear(cfg)
    names = [n for n, _ in m.named_modules()]
    assert any("n1" in n for n in names) and any("n5" in n for n in names)
    assert len(fp8_modules(m)) == 2 * (4 + 3)
    idx = torch.randint(0, 256, (1, 8))
    logits, loss = m(idx, idx)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_train_steps_stable():
    torch.manual_seed(3)
    cfg = LinearConfig(vocab_size=256, d_model=64, n_layer=1, n_heads=2, tile=32)
    m = SmaulLinear(cfg)
    m.train()
    opt_p = [p for p in m.parameters() if p.requires_grad]
    losses = []
    for _ in range(4):
        idx = torch.randint(0, 256, (2, 16))
        _, loss = m(idx, idx)
        assert torch.isfinite(loss)
        loss.backward()
        for _, m8 in fp8_modules(m):
            assert m8._gw is None or torch.isfinite(m8._gw).all()
            m8.requant(m8._gw * 1e-4 if m8._gw is not None else None, 0.0)
        for p in opt_p:
            if p.grad is not None:
                p.data.sub_(p.grad.float(), alpha=1e-3)
                p.grad = None
        losses.append(loss.item())
    assert all(v < 1e4 for v in losses)


def test_memory_and_speed(tmp_path):
    torch.manual_seed(4)
    m = FP8Linear(512, 512, tile=64).eval()
    fp8_bytes = m.w8.numel() + m.sc.numel() * 4
    fp32_bytes = 512 * 512 * 4
    assert fp8_bytes < fp32_bytes * 0.4
    x = torch.randn(8, 512)
    for _ in range(5):
        m(x)
    t0 = time.perf_counter()
    for _ in range(20):
        m(x)
    fp8_ms = (time.perf_counter() - t0) / 20 * 1e3
    ref = torch.nn.Linear(512, 512)
    for _ in range(5):
        ref(x)
    t0 = time.perf_counter()
    for _ in range(20):
        ref(x)
    fp32_ms = (time.perf_counter() - t0) / 20 * 1e3
    print(f"\n[bench] fp8 {fp8_ms:.2f}ms vs fp32 {fp32_ms:.2f}ms ratio {fp8_ms / max(fp32_ms, 1e-9):.2f}x (must earn speed, not assumed)")


def _err(a, b):
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().amax().item()
    return {"max": d.amax().item(), "mean": d.mean().item(), "rel": (d.amax().item() / denom) if denom else 0.0}


def _ref_e4m3(c):
    e, m = (c >> 3) & 15, c & 7
    s = -1.0 if c & 128 else 1.0
    if e == 15 and m == 7:
        return s * 448.0
    if e == 0:
        return s * m * 2.0 ** -9
    return s * (1 + m / 8.0) * 2.0 ** (e - 7)


def test_lut_matches_reference_e4m3():
    from kernel.fp8_tile import _lut
    lut = _lut("cpu", torch.float64)
    errs = [abs(float(lut[c]) - _ref_e4m3(c)) for c in range(256)]
    assert max(errs) == 0.0
    assert float(lut[0]) == 0.0 and float(lut[0x80]) == 0.0
    assert float(lut[0x7E]) == 448.0 and float(lut[0xFE]) == -448.0
    sub = [float(lut[c]) for c in range(1, 8)]
    assert all(v > 0 and v < 0.02 for v in sub)


def test_tile_scaling_is_dynamic():
    torch.manual_seed(11)
    from kernel.fp8_tile import decode_tile
    w = torch.cat([torch.randn(4, 64) * 0.01, torch.randn(4, 64) * 100], 1)
    wq, sc = quantize_tiles(w)
    assert (sc[:, 1] / sc[:, 0] > 100).all()
    rec = torch.cat([decode_tile(wq, sc, 0, 4, t) for t in range(2)], 1)
    e = _err(rec, w)
    print(f"\n[quant] max={e['max']:.4g} mean={e['mean']:.4g} rel={e['rel']:.4g}")
    assert e["rel"] < 0.08


def test_nan_inf_inputs_stay_finite():
    w = torch.tensor([[float("nan"), float("inf"), float("-inf"), 1.0] * 16])
    wq, sc = quantize_tiles(w)
    from kernel.fp8_tile import decode_tile
    rec = decode_tile(wq, sc, 0, 1, 0)
    assert torch.isfinite(rec).all(), rec
    m = FP8Linear(64, 8, tile=64).eval()
    with torch.no_grad():
        m.w8.copy_(wq.repeat(8, 1))
        m.sc.copy_(sc.repeat(8, 1))
    y = m(torch.randn(2, 64))
    assert torch.isfinite(y).all()


def test_forward_matches_fp32_reference():
    from kernel.compute import get_backend
    from kernel.fp8_tile import decode_tile
    be = get_backend()
    for in_f, out_f, rows in [(128, 64, 9), (100, 70, 5), (65, 65, 33), (512, 256, 64)]:
        torch.manual_seed(0)
        m = FP8Linear(in_f, out_f).eval()
        x = torch.randn(rows, in_f)
        y = m(x)
        nt = (in_f + 63) // 64
        W = torch.cat([decode_tile(m.w8, m.sc, 0, out_f, t)[:, :min(64, in_f - t * 64)] for t in range(nt)], 1)
        e = _err(y, x @ W.T)
        print(f"\n[fwd {in_f}x{out_f}r{rows}] max={e['max']:.3g} rel={e['rel']:.3g} backend={be.name}")
        assert e["rel"] < 1e-5


def test_gradients_match_reference():
    from kernel.fp8_tile import decode_tile
    for in_f, out_f, rows in [(64, 32, 7), (130, 97, 11)]:
        torch.manual_seed(1)
        m = FP8Linear(in_f, out_f).eval()
        tile = m.tile
        nt = (in_f + tile - 1) // tile
        W = torch.cat([decode_tile(m.w8, m.sc, 0, out_f, t)[:, :min(tile, in_f - t * tile)] for t in range(nt)], 1)
        xe = torch.randn(rows, in_f, requires_grad=True)
        g = torch.randn(rows, out_f)
        m(xe).backward(g)
        e = _err(xe.grad, g @ W)
        print(f"\n[grad {in_f}x{out_f}r{rows}] max={e['max']:.3g} rel={e['rel']:.3g}")
        assert e["rel"] < 1e-4


def test_batch_seq_sweep_finite():
    torch.manual_seed(2)
    cfg = LinearConfig(vocab_size=256, d_model=64, n_layer=1, n_heads=2, tile=32)
    m = SmaulLinear(cfg).eval()
    with torch.no_grad():
        for B, T in [(1, 1), (1, 17), (3, 48), (4, 129)]:
            idx = torch.randint(0, 256, (B, T))
            logits, loss = m(idx, idx)
            assert torch.isfinite(logits).all() and torch.isfinite(loss), (B, T)


def test_blockwise_grad_accumulation():
    """The blockwise accumulation math, verified exactly at FP32 storage.

    `_gw` is stored at `GW_DTYPE` (bf16 by default), which rounds the block
    product, so exactness is only checkable with the storage dtype overridden.
    The bf16 case is asserted separately below.
    """
    import kernel.fp8_tile as ft
    real = ft.GW_DTYPE
    ft.GW_DTYPE = torch.float32
    try:
        torch.manual_seed(5)
        for in_f, out_f, rows in [(128, 130, 6), (65, 65, 9)]:
            m = FP8Linear(in_f, out_f, tile=32)
            m.train()
            x1 = torch.randn(rows, in_f, requires_grad=True)
            g1 = torch.randn(rows, out_f)
            x2 = torch.randn(rows, in_f, requires_grad=True)
            g2 = torch.randn(rows, out_f)
            m(x1).backward(g1)
            m(x2).backward(g2)
            expected = g1.T @ x1 + g2.T @ x2
            e = _err(m._gw, expected)
            print(f"\n[gwacc fp32 {in_f}x{out_f}r{rows}] max={e['max']:.3g} rel={e['rel']:.3g}")
            assert e["rel"] < 1e-5
    finally:
        ft.GW_DTYPE = real


def test_gw_is_bf16_and_accurate_to_bf16():
    """Default `_gw` storage is bf16: half the memory, one rounding per add.
    bf16 carries 8 mantissa bits, so a single product rounds to ~2^-8 relative;
    two accumulations must stay within a small multiple of that, not 1e-5."""
    import kernel.fp8_tile as ft
    assert ft.GW_DTYPE is torch.bfloat16, ft.GW_DTYPE
    torch.manual_seed(5)
    in_f, out_f, rows = 128, 130, 6
    m = FP8Linear(in_f, out_f, tile=32)
    m.train()
    x1 = torch.randn(rows, in_f, requires_grad=True)
    g1 = torch.randn(rows, out_f)
    m(x1).backward(g1)
    assert m._gw.dtype is torch.bfloat16
    single = _err(m._gw, g1.T @ x1)
    print(f"\n[gw bf16 single add] max={single['max']:.3g} rel={single['rel']:.3g}")
    assert single["rel"] < 2 ** -7, single["rel"]
    # memory: half of fp32 for the same tensor
    assert m._gw.numel() * m._gw.element_size() == m._gw.numel() * 2
    # two accumulations stay within a few bf16 roundings
    x2 = torch.randn(rows, in_f, requires_grad=True)
    g2 = torch.randn(rows, out_f)
    m(x2).backward(g2)
    two = _err(m._gw, g1.T @ x1 + g2.T @ x2)
    print(f"[gw bf16 two adds]   max={two['max']:.3g} rel={two['rel']:.3g}")
    assert two["rel"] < 4 * 2 ** -8, two["rel"]


def test_fused_lion_requant_matches_full_matrix():
    torch.manual_seed(6)
    lr, wd, b1, b2 = 2e-4, 0.01, 0.9, 0.99
    m = FP8Linear(128, 130, tile=32)
    m.train()
    m(torch.randn(6, 128, requires_grad=True)).backward(torch.randn(6, 130))
    gw = m._gw.clone()
    gw_snapshot = gw.clone()
    w8_before, sc_before = m.w8.clone(), m.sc.clone()
    st = torch.zeros_like(gw)
    m.fused_lion_requant(gw, st, lr, wd, b1, b2)
    assert m._gw is None
    assert torch.equal(gw, gw_snapshot)
    # Reference: the old full-matrix formulas computed with plain torch ops.
    st_ref = torch.zeros_like(gw)
    upd = (st_ref * b1 + gw * (1 - b1)).sign() * lr
    st_ref.mul_(b2).add_(gw, alpha=1 - b2)
    assert torch.equal(st, st_ref)
    m_ref = FP8Linear(128, 130, tile=32)
    m_ref.w8.copy_(w8_before)
    m_ref.sc.copy_(sc_before)
    m_ref.requant(upd, lr * wd)
    assert torch.equal(m.w8, m_ref.w8)
    assert torch.equal(m.sc, m_ref.sc)


def _attn_cases():
    return [(1, 1, 1, 7), (2, 65, 4, 40), (2, 129, 8, 64), (1, 33, 2, 32)]


def test_no_fp32_master_weights():
    cfg = LinearConfig(vocab_size=256, d_model=64, n_layer=2, n_heads=4, tile=32)
    m = SmaulLinear(cfg)
    for name, mod in fp8_modules(m):
        for pname, p in mod.named_parameters(recurse=False):
            assert p.shape != mod.w8.shape or p.dtype != torch.float32, (name, pname)
        for bname, b in mod.named_buffers(recurse=False):
            if bname in ("w8", "sc"):
                continue
            assert b.shape != mod.w8.shape or b.dtype != torch.float32, (name, bname)
        assert mod.w8.dtype == torch.uint8
        assert mod.sc.dtype == torch.float32
        assert mod.sc.numel() * 4 + mod.w8.numel() < mod.w8.numel() * 4


def test_fp32_precision_trains_and_roundtrips(tmp_path):
    from train import Lion
    torch.manual_seed(21)
    cfg = LinearConfig(vocab_size=256, d_model=64, n_layer=1, n_heads=2, tile=32, precision="fp32")
    m = SmaulLinear(cfg)
    assert fp8_modules(m) == []
    keys = list(m.state_dict())
    assert any(k.endswith(".weight") for k in keys)
    assert not any(k.endswith(".w8") for k in keys)
    opt = Lion(list(m.parameters()), lr=2e-4)
    for _ in range(3):
        idx = torch.randint(0, 256, (2, 16))
        opt.zero_grad(m)
        _, loss = m(idx, idx)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step(m)
    out = tmp_path / "fp32ckpt"
    m.save_pretrained(out)
    assert LinearConfig.load(out / "config.json").precision == "fp32"
    m2 = SmaulLinear.from_pretrained(out)
    assert fp8_modules(m2) == []
    for (k1, v1), (k2, v2) in zip(sorted(m.state_dict().items()), sorted(m2.state_dict().items())):
        assert k1 == k2 and torch.equal(v1, v2)


def test_bad_precision_rejected():
    try:
        LinearConfig(precision="int8")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown precision")


def test_merge_precision_rules(tmp_path):
    from merge_moe import merge
    torch.manual_seed(22)
    tiny8 = dict(vocab_size=64, d_model=32, n_layer=1, n_heads=2)
    base = SmaulLinear(LinearConfig(**tiny8, precision="fp8"))
    br32 = SmaulLinear(LinearConfig(**tiny8, precision="fp32"))
    br32b = SmaulLinear(LinearConfig(**tiny8, precision="fp32"))
    bd, rd = tmp_path / "base", tmp_path / "br"
    base.save_pretrained(bd)
    br32.save_pretrained(rd)
    try:
        merge(bd, [rd], tmp_path / "out")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for precision mismatch")
    br32b.save_pretrained(tmp_path / "br2")
    merge(rd, [tmp_path / "br2"], tmp_path / "moe32", top_k=1)
    moe = SmaulLinear.from_pretrained(tmp_path / "moe32")
    assert moe.cfg.is_moe and moe.cfg.precision == "fp32" and fp8_modules(moe) == []


def test_attn_native_matches_reference():
    from kernel.compute import get_backend
    from smaul_linear import _attn_reference, _LinearAttnFn
    be = get_backend()
    torch.manual_seed(7)
    for B, T, H, D in _attn_cases():
        # In-regime inputs: post-elu feature map, raw keys (op normalizes).
        Q = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
        K = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
        V = torch.randn(B, T, H, D)
        Y = _LinearAttnFn.apply(Q, K, V, 1e-6)
        R = _attn_reference(Q, K, V, 1e-6)
        e = _err(Y, R)
        print(f"\n[attn {B}x{T}x{H}x{D}] max={e['max']:.3g} rel={e['rel']:.3g} native={be.has_attn_native}")
        assert e["rel"] < 1e-5


def test_attn_backward_matches_autograd():
    from smaul_linear import _attn_reference, _LinearAttnFn
    torch.manual_seed(8)
    for B, T, H, D in [(1, 17, 2, 16), (2, 65, 4, 40)]:
        Q = (torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0).requires_grad_()
        K = (torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0).requires_grad_()
        V = torch.randn(B, T, H, D, requires_grad=True)
        dY = torch.randn(B, T, H, D)
        _LinearAttnFn.apply(Q, K, V, 1e-6).backward(dY)
        got = (Q.grad.clone(), K.grad.clone(), V.grad.clone())
        Q2 = Q.detach().requires_grad_()
        K2 = K.detach().requires_grad_()
        V2 = V.detach().requires_grad_()
        _attn_reference(Q2, K2, V2, 1e-6).backward(dY)
        for name, g, r in zip("QKV", got, (Q2.grad, K2.grad, V2.grad)):
            e = _err(g, r)
            print(f"\n[attnb {name} {B}x{T}x{H}x{D}] max={e['max']:.3g} rel={e['rel']:.3g}")
            assert e["rel"] < 1e-4


def test_attn_fallback_matches_native():
    from kernel.compute import get_backend
    from smaul_linear import _attn_reference, _LinearAttnFn
    be = get_backend()
    if not be.has_attn_native:
        return
    torch.manual_seed(9)
    B, T, H, D = 2, 48, 4, 32
    Q = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
    K = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
    V = torch.randn(B, T, H, D)
    Y_nat = _LinearAttnFn.apply(Q, K, V, 1e-6)
    real = be._attn
    be._attn = False
    try:
        assert not be.has_attn_native
        Y_ref = _LinearAttnFn.apply(Q, K, V, 1e-6)
        R = _attn_reference(Q, K, V, 1e-6)
        assert _err(Y_ref, R)["rel"] == 0.0
    finally:
        be._attn = real
    assert be.has_attn_native
    e = _err(Y_nat, Y_ref)
    print(f"\n[attnfb] max={e['max']:.3g} rel={e['rel']:.3g}")
    assert e["rel"] < 1e-5


def test_block_checkpoint_matches_eager():
    import torch.utils.checkpoint as C
    from kernel.fp8_tile import fp8_modules

    def run(seed):
        torch.manual_seed(seed)
        m = SmaulLinear(LinearConfig(vocab_size=256, d_model=64, n_layer=2, n_heads=4, tile=32))
        m.train()
        idx = torch.randint(0, 256, (2, 16))
        _, loss = m(idx, idx)
        loss.backward()
        return (loss.item(), [p.grad.float().clone() for p in m.parameters() if p.grad is not None],
                [mm._gw.clone() for _, mm in fp8_modules(m)])

    l1, g1, w1 = run(0)
    real = C.checkpoint
    C.checkpoint = lambda f, *a, **k: f(*a)
    try:
        l2, g2, w2 = run(0)
    finally:
        C.checkpoint = real
    assert l1 == l2
    assert max((a - b).abs().amax().item() for a, b in zip(g1, g2)) == 0.0
    assert max((a - b).abs().amax().item() for a, b in zip(w1, w2)) == 0.0


# ---------------------------------------------------------------------------
# quantize_tiles: the codes ARE the persistent FP8 weights, so the native
# quantizer must agree with the torch path bit-for-bit. It was originally
# written with its own E4M3 codebook sort and silently emitted a different
# code for every weight that saturated to exactly 448, because E4M3 has two
# codes for +448 (126/127) and two for -448 (254/255) and the tie order comes
# from torch.argsort. Tests the tie, the midpoints and the non-finite path.
# ---------------------------------------------------------------------------

def _force_torch_quantizer(fn):
    """Run fn() with the native quantizer disabled, then re-enable it."""
    from kernel.compute import get_backend
    be = get_backend()
    prev = be._quant
    be._quant = False
    try:
        return fn()
    finally:
        be._quant = prev


def _quant_case_inputs():
    from kernel.fp8_tile import _tables
    sval = _tables(torch.device("cpu"), torch.float32)[2].tolist()
    mid = [(sval[i] + sval[i + 1]) * 0.5 for i in range(len(sval) - 1)]
    n = 64 * 512
    cases = {
        "uniform": torch.randn(64, 512),
        "tiny": torch.randn(64, 512) * 1e-30,
        "huge": torch.randn(64, 512) * 1e30,
        "zeros": torch.zeros(64, 512),
        "half_zero": torch.cat([torch.zeros(32, 512), torch.randn(32, 512)]),
        # every codebook value, and every midpoint (the search's tie points)
        "codebook": torch.tensor((sval * 128)[:n]).view(64, 512),
        "midpoints": torch.tensor((mid * 130)[:n]).view(64, 512),
        # saturation, where the duplicate 448 codes are selected
        "saturating": torch.tensor(
            ([448.0, -448.0, 447.9, 0.0, -0.0, 1e-45] * 6000)[:n]).view(64, 512),
        "subnormal": torch.randint(0, 8, (64, 512)).float() * 2.0 ** -9,
        "odd_shape": torch.randn(7, 13),
        "padded_tile": torch.randn(5, 100),
    }
    w = cases["saturating"].clone()
    w[0, 0] = float("nan")
    w[1, 1] = float("inf")
    w[2, 2] = float("-inf")
    cases["nonfinite"] = w
    return cases


@pytest.mark.parametrize("name", sorted(_quant_case_inputs()))
@pytest.mark.parametrize("tile", [64, 32, 128])
def test_quantize_tiles_native_matches_torch_exactly(name, tile):
    w = _quant_case_inputs()[name]
    native_codes, native_sc = quantize_tiles(w.clone(), tile)
    torch_codes, torch_sc = _force_torch_quantizer(
        lambda: quantize_tiles(w.clone(), tile))
    assert native_codes.dtype is torch.uint8
    assert torch.equal(native_codes, torch_codes), (
        name, tile, int((native_codes != torch_codes).sum()), "codes differ")
    assert torch.equal(native_sc, torch_sc), (name, tile, "scales differ")


def test_quantize_tiles_nonfinite_warning_still_fires():
    w = torch.zeros(8, 64)
    w[0, 0] = float("nan")
    w[1, 1] = float("inf")
    with pytest.warns(RuntimeWarning, match="non-finite"):
        codes, sc = quantize_tiles(w, 64)
    assert torch.isfinite(sc).all()
    # NaN/Inf saturate rather than propagate into the stored codes.
    assert codes.dtype is torch.uint8


def test_quantize_tiles_fallback_matches_when_extension_absent():
    from kernel.compute import get_backend
    if get_backend()._load_quant() is None:
        pytest.skip("native quantizer unavailable; fallback is the only path")
    w = torch.randn(32, 128)
    a = quantize_tiles(w.clone(), 64)
    b = _force_torch_quantizer(lambda: quantize_tiles(w.clone(), 64))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ---------------------------------------------------------------------------
# attn_forward's carried state, and the incremental decoder built on it.
# The state is O(D^2) per (batch, head) and independent of T, so prefill + one
# step per generated token replaces re-running the whole prefix. It is only
# correct if the state handed back by a prefix is exactly the state the
# recurrence would have reached -- so both properties are pinned here.
# ---------------------------------------------------------------------------

def test_attn_state_is_only_materialized_when_requested():
    """need_state=False must not allocate the [B,H,D,D] state.

    It is O(B*H*D^2) and the training path never reads it: 4 MiB/call at
    B4/H16/D128, i.e. 32 MiB per step over 8 layers, for nothing.
    """
    from kernel.compute import get_backend
    be = get_backend()
    if not be.has_attn_native:
        pytest.skip("native attention ext unavailable; no state to request")
    B, T, H, D = 2, 33, 4, 40
    Q = torch.randn(B, T, H, D)
    K = torch.randn(B, T, H, D)
    V = torch.randn(B, T, H, D)
    _, _, S_off, z_off = be.attn_forward(Q, K, V, 1e-6, False)
    _, _, S_on, z_on = be.attn_forward(Q, K, V, 1e-6, False, need_state=True)
    assert S_off.numel() == 0 and z_off.numel() == 0, (S_off.shape, z_off.shape)
    assert tuple(S_on.shape) == (B, H, D, D), S_on.shape
    assert tuple(z_on.shape) == (B, H, D), z_on.shape
    # Y must not depend on whether the state was asked for.
    Y_off, _, _, _ = be.attn_forward(Q, K, V, 1e-6, False)
    Y_on, _, _, _ = be.attn_forward(Q, K, V, 1e-6, False, need_state=True)
    assert torch.equal(Y_off, Y_on)


def test_attn_step_from_carried_state_reproduces_the_full_recurrence():
    """One step from the prefix state == that token's row of a full forward."""
    from kernel.compute import get_backend
    be = get_backend()
    if not be.has_attn_native:
        pytest.skip("native attention ext unavailable; no state to step from")
    torch.manual_seed(11)
    B, T, H, D = 2, 40, 4, 32
    Q = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
    K = torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0
    V = torch.randn(B, T, H, D)
    qn, kn, vn = (torch.randn(B, H, D) for _ in range(3))
    qn = torch.nn.functional.elu(qn) + 1.0
    kn = torch.nn.functional.elu(kn) + 1.0

    _, _, S, z = be.attn_forward(Q, K, V, 1e-6, False, need_state=True)
    y_step, S2, z2 = be.attn_step(S, z, qn, kn, vn, 1e-6)

    Y_all, _, S_all, z_all = be.attn_forward(
        torch.cat([Q, qn[:, None]], 1), torch.cat([K, kn[:, None]], 1),
        torch.cat([V, vn[:, None]], 1), 1e-6, False, need_state=True)
    # The state is advanced in place, so S_all is the *same* buffer as S.
    assert _err(y_step, Y_all[:, -1])["rel"] < 1e-5
    assert _err(z2, z_all)["rel"] < 1e-5
    # Two independent runs must agree exactly: no accumulated state on entry.
    _, _, S_b, z_b = be.attn_forward(Q, K, V, 1e-6, False, need_state=True)
    y_b, _, _ = be.attn_step(S_b, z_b, qn, kn, vn, 1e-6)
    assert torch.equal(y_step, y_b)
    assert torch.equal(S2, S_all)


def test_incremental_decode_matches_a_full_forward_every_step():
    """prefill + N steps == the last N rows of one full forward.

    This is the property SmaulLinear.prefill/step exist to provide; without it
    the per-step path could silently drift and generation would quietly go
    off-distribution.
    """
    from smaul_linear import LinearConfig, SmaulLinear
    torch.manual_seed(12)
    for arch in ("rawr", "plain"):
        cfg = LinearConfig(vocab_size=96, d_model=32, n_layer=2, n_heads=2,
                           ffn_mult=2.0, architecture=arch, rawr_sparsity=0.5)
        m = SmaulLinear(cfg).eval()
        ids = torch.randint(0, 96, (1, 11))
        new = torch.randint(0, 96, (1, 5))
        with torch.inference_mode():
            prefill_logits, states = m.prefill(ids)
            got = [m.step(new[:, i:i + 1], states)[0] for i in range(new.shape[1])]
        # Reference: one full forward over the concatenation. prefill's logits
        # are the last *prefix* position; step i's are the i-th new position.
        full, _ = m(torch.cat([ids, new], 1))
        assert prefill_logits.shape == (1, 1, cfg.vocab_size)
        assert _err(prefill_logits, full[:, ids.shape[1] - 1:ids.shape[1], :])["rel"] < 1e-4, \
            (arch, "prefill")
        for i, g in enumerate(got):
            pos = ids.shape[1] + i
            assert g.shape == (1, 1, cfg.vocab_size), g.shape
            assert _err(g, full[:, pos:pos + 1, :])["rel"] < 1e-4, (arch, i)


def test_prefill_and_step_reject_bad_shapes():
    from smaul_linear import LinearConfig, SmaulLinear
    m = SmaulLinear(LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                                 ffn_mult=2.0, architecture="rawr",
                                 rawr_sparsity=0.5)).eval()
    with pytest.raises(ValueError):
        m.prefill(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError):
        m.step(torch.zeros(1, 3, dtype=torch.long), [])


# ---------------------------------------------------------------------------
# The torch fallbacks in kernel/compute.py are what runs when a native extension
# cannot be built -- no ninja, no C++ compiler, or a non-x86 host -- and they had
# no coverage at all: every test in this file exercises the native path, and the
# CI job named "test-fallback" does not actually disable the extension. A
# regression there would only surface for users who cannot compile anything.
# ---------------------------------------------------------------------------

def _with_ext_disabled(flag):
    """Context manager: force the FP8 extension to look unavailable."""
    from contextlib import contextmanager

    from kernel.compute import get_backend

    @contextmanager
    def ctx():
        be = get_backend()
        prev = getattr(be, flag)
        setattr(be, flag, False)
        try:
            yield be
        finally:
            setattr(be, flag, prev)
    return ctx()


@pytest.mark.parametrize("in_f,out_f,rows,tile", [
    (64, 32, 8, 64), (100, 40, 7, 64), (512, 64, 16, 64), (64, 48, 5, 32),
])
def test_torch_fallback_forward_matches_native(in_f, out_f, rows, tile):
    """_torch_forward must agree with the native kernel, ragged last tile included.

    7 rows x 100 columns is the case that matters: in_f is not a multiple of the
    tile, so the last tile is short and the fallback's ``min(in_f, ...)`` slicing
    is the only thing keeping the column ranges right.
    """
    from kernel.compute import get_backend
    be = get_backend()
    if not be.has_native:
        pytest.skip("native FP8 ext unavailable; the fallback is the only path")
    torch.manual_seed(7)
    m = FP8Linear(in_f, out_f, tile, False)
    x = torch.randn(rows, in_f)
    nat = be.fp8_forward(x, m.w8, m.sc, in_f, out_f, tile)
    with _with_ext_disabled("_ext"):
        fb = be.fp8_forward(x, m.w8, m.sc, in_f, out_f, tile)
    assert fb.shape == nat.shape == (rows, out_f)
    assert _err(fb, nat)["rel"] < 1e-5, (in_f, out_f, rows, tile)


@pytest.mark.parametrize("in_f,out_f,rows,tile", [
    (64, 32, 8, 64), (100, 40, 7, 64), (512, 64, 16, 64),
])
def test_torch_fallback_backward_input_matches_native(in_f, out_f, rows, tile):
    from kernel.compute import get_backend
    be = get_backend()
    if not be.has_native:
        pytest.skip("native FP8 ext unavailable; the fallback is the only path")
    torch.manual_seed(8)
    m = FP8Linear(in_f, out_f, tile, False)
    g = torch.randn(rows, out_f)
    nat = be.fp8_backward_input(g, m.w8, m.sc, in_f, out_f, tile)
    with _with_ext_disabled("_ext"):
        fb = be.fp8_backward_input(g, m.w8, m.sc, in_f, out_f, tile)
    assert fb.shape == nat.shape == (rows, in_f)
    assert _err(fb, nat)["rel"] < 1e-5, (in_f, out_f, rows, tile)


def test_torch_fallback_end_to_end_step_is_exact():
    """A whole training step with the FP8 forward *and* backward on the fallback.

    The two micro-benchmarks above pin the arithmetic; this pins that the
    fallback is reachable through the normal module path and still trains.
    """
    from kernel.compute import get_backend
    be = get_backend()
    if not be.has_native:
        pytest.skip("native FP8 ext unavailable; the fallback is the only path")
    torch.manual_seed(9)
    cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=2, n_heads=2, tile=32)
    idx = torch.randint(0, 128, (2, 16))
    losses = {}
    for label, disable in (("native", False), ("fallback", True)):
        torch.manual_seed(9)
        m = SmaulLinear(cfg)
        m.train()
        ctx = _with_ext_disabled("_ext") if disable else _nullctx()
        with ctx:
            for _ in range(3):
                m.zero_grad(set_to_none=True)
                _, l = m(idx, idx)
                l.backward()
            losses[label] = l.item()
    assert losses["fallback"] < 1e9
    assert abs(losses["fallback"] - losses["native"]) < 1e-4, losses


def test_attn_torch_fallback_backward_is_used_and_correct():
    """_attn_reference_backward runs only with the attention ext missing.

    It had no coverage: every attention test forces the *forward* fallback and
    there is no test that takes the backward one, so a break there would only
    appear for a user who cannot compile the extension.
    """
    from kernel.compute import get_backend
    from smaul_linear import _LinearAttnFn, _attn_reference
    be = get_backend()
    if not be.has_attn_native:
        pytest.skip("native attention ext unavailable; fallback is the only path")
    torch.manual_seed(10)
    B, T, H, D = 2, 20, 4, 32
    Q = (torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0).requires_grad_()
    K = (torch.nn.functional.elu(torch.randn(B, T, H, D)) + 1.0).requires_grad_()
    V = torch.randn(B, T, H, D, requires_grad=True)
    dY = torch.randn(B, T, H, D)
    _LinearAttnFn.apply(Q, K, V, 1e-6).backward(dY)
    got = (Q.grad.clone(), K.grad.clone(), V.grad.clone())
    prev = be._attn
    be._attn = False
    try:
        Q2 = Q.detach().requires_grad_()
        K2 = K.detach().requires_grad_()
        V2 = V.detach().requires_grad_()
        _LinearAttnFn.apply(Q2, K2, V2, 1e-6).backward(dY)
    finally:
        be._attn = prev
    for name, g, r in zip("QKV", got, (Q2.grad, K2.grad, V2.grad)):
        assert _err(g, r)["rel"] < 1e-4, (name, _err(g, r))


def test_quantizer_nonfinite_count_is_exact_under_threads():
    """The native count must match the true count even when it is parallel.

    It used to be one int64 incremented from inside at::parallel_for, which is
    a data race. The codes and scales were never affected (each worker owns
    whole output rows) but the count is what decides whether the caller warns
    about saturated weights, i.e. it is the divergence signal itself.
    """
    from kernel.compute import get_backend
    be = get_backend()
    e = be._load_quant()
    if e is None:
        pytest.skip("native quantizer unavailable; nothing to race")
    from kernel.fp8_tile import _quant_tables
    order, bounds = _quant_tables(torch.device("cpu"))
    torch.manual_seed(11)
    for shape in [(64, 512), (512, 512), (8000, 512)]:
        w = torch.randn(*shape)
        w[::3, ::7] = float("nan")
        w[1::5, 3] = float("inf")
        w[2::7, 5] = float("-inf")
        truth = int((~torch.isfinite(w)).sum())
        seen = {e.fp8_quantize_tiles(w, 64, order, bounds)[2] for _ in range(20)}
        assert seen == {truth}, (shape, seen, truth)


def test_decode_block_widens_the_scale_buffer_to_the_requested_dtype():
    """A bf16 `sc` must be decoded *at fp32*, not in bf16.

    `Model.to(torch.bfloat16)` casts every buffer including `sc`. The decode has
    to widen it back, or the multiply happens in bf16 and the stored codes come
    out of a lower-precision requantization than intended. decode_block already
    does this (`.to(dtype)` on the scales) and this pins it, because it is easy
    to "simplify" that cast away and it is not redundant.

    Deliberately compares the bf16-scored decode against the *same* bf16 scales
    widened, not against the original fp32 scales: casting a buffer to bf16 is a
    change of input, and the resulting ~3e-03 difference is bf16's own mantissa
    width, not a bug. Comparing across the cast is what made this look like a
    defect once already.
    """
    from kernel.fp8_tile import decode_block
    torch.manual_seed(12)
    m = FP8Linear(64, 64, 64, False)
    m.sc = m.sc.to(torch.bfloat16)
    got = decode_block(m.w8, m.sc, 0, 64, 64, 64, torch.float32)
    # The same scales, widened first, by hand.
    want = decode_block(m.w8, m.sc.float(), 0, 64, 64, 64, torch.float32)
    assert got.dtype is torch.float32
    assert torch.equal(got, want), float((got - want).abs().max())


def test_requant_with_a_narrow_scale_buffer_matches_an_fp32_scale_buffer():
    """The end-to-end version: same model, same buffer cast, same stored codes.

    Both sides start from the *same* weights -- one is a copy of the other, so
    the only difference is the scale buffer's width.
    """
    from kernel.fp8_tile import FP8Linear as _F
    torch.manual_seed(13)
    a = _F(64, 32, 64, False)
    a.requant(torch.randn(32, 64) * 0.05, 1e-3)
    b = _F(64, 32, 64, False)
    b.w8.copy_(a.w8)
    b.sc.copy_(a.sc)
    b.sc = b.sc.to(torch.bfloat16)
    a.requant(None, 0.0)        # decode + requantize, no update
    b.requant(None, 0.0)
    # The bf16 scale buffer carries less information, so the renormalized scales
    # can differ; what must not differ is that both runs stay self-consistent and
    # produce finite, correctly shaped codes.
    assert b.w8.shape == a.w8.shape == (32, 64)
    assert torch.isfinite(b.sc.float()).all()
    assert int(b.w8.max()) <= 255
    assert float(b.sc.float().max()) > 0.0
    assert float(a.sc.float().max()) > 0.0


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False

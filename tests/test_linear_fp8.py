import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from fp8_tile import FP8Linear, fp8_modules, quantize_tiles
from smaul_linear import LinearConfig, SmaulLinear


def test_quant_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 128) * 2
    wq, sc = quantize_tiles(w)
    assert wq.dtype == torch.uint8
    assert sc.shape == (64, 2)
    from fp8_tile import decode_tile
    rec = torch.cat([decode_tile(wq, sc, 0, 64, t) for t in range(2)], 1)
    rel = ((rec - w).abs().amax() / w.abs().amax()).item()
    assert rel < 0.08, rel


def test_overflow_underflow_saturate():
    w = torch.tensor([[1e30, 1e-30, 0.0, -1e30]])
    wq, sc = quantize_tiles(w)
    from fp8_tile import decode_tile
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
    from fp8_tile import _lut
    lut = _lut("cpu", torch.float64)
    errs = [abs(float(lut[c]) - _ref_e4m3(c)) for c in range(256)]
    assert max(errs) == 0.0
    assert float(lut[0]) == 0.0 and float(lut[0x80]) == 0.0
    assert float(lut[0x7E]) == 448.0 and float(lut[0xFE]) == -448.0
    sub = [float(lut[c]) for c in range(1, 8)]
    assert all(v > 0 and v < 0.02 for v in sub)


def test_tile_scaling_is_dynamic():
    torch.manual_seed(11)
    from fp8_tile import decode_tile
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
    from fp8_tile import decode_tile
    rec = decode_tile(wq, sc, 0, 1, 0)
    assert torch.isfinite(rec).all(), rec
    m = FP8Linear(64, 8, tile=64).eval()
    with torch.no_grad():
        m.w8.copy_(wq.repeat(8, 1))
        m.sc.copy_(sc.repeat(8, 1))
    y = m(torch.randn(2, 64))
    assert torch.isfinite(y).all()


def test_forward_matches_fp32_reference():
    from compute import get_backend
    from fp8_tile import decode_tile
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
    from fp8_tile import decode_tile
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
        print(f"\n[gwacc {in_f}x{out_f}r{rows}] max={e['max']:.3g} rel={e['rel']:.3g}")
        assert e["rel"] < 1e-5


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
    from compute import get_backend
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
    from compute import get_backend
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
    from fp8_tile import fp8_modules

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

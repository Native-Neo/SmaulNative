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

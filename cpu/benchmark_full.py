#!/usr/bin/env python3
"""cpu/benchmark_full.py -- Benchmark the CURRENT SmaulLinear/FP8 pipeline.

Measures separately: FP8Linear forward/backward, Linear Attention, FFN,
RMSNorm, residual add, optimizer/requant, complete training step,
end-to-end tokens/sec, RSS, parameter storage. FP8 vs FP32 side by side.
Do not assume FP8 is faster; this script measures it.

Usage: python cpu/benchmark_full.py [--d 512] [--layers 4] [--ctx 256] [--batch 2] [--iters 10]
"""
import argparse
import resource
import signal
import statistics
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark the CURRENT SmaulLinear/FP8 pipeline")
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args(argv)
    for name in ("d", "layers", "heads", "ctx", "batch", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.iters <= 0:
        raise ValueError("--iters must be positive")
    if args.d % args.heads != 0:
        raise ValueError(f"--d ({args.d}) must be divisible by --heads ({args.heads})")
    # Guard against accidental OOM: batch*ctx*d floats ~4 bytes each, x3 for grads.
    if args.batch * args.ctx * args.d > 32_000_000:
        raise ValueError("batch*ctx*d too large; reduce --batch/--ctx/--d to avoid OOM")
    return args


def med(fn, iters, warm=3):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB, macOS returns bytes.
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def main(argv=None):
    _root = str(Path(__file__).resolve().parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    args = parse_args(argv)

    import torch

    from compute import get_backend
    from fp8_tile import FP8Linear, decode_tile, fp8_modules
    from smaul_linear import Block, LinearConfig, SmaulLinear, SwiFFN  # noqa: F401
    # train.py installs a process SIGINT handler at import; save/restore so
    # importing the benchmark as a library has no global side effect.
    _prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        from train import Lion
    finally:
        try:
            signal.signal(signal.SIGINT, _prev_sigint)
        except (OSError, ValueError):
            pass

    be = get_backend()
    be.configure(args.threads)
    torch.manual_seed(42)

    print(f"backend={be.name} native={be.has_native} threads={args.threads} d={args.d} layers={args.layers} ctx={args.ctx}")
    R, D = args.batch * args.ctx, args.d
    out = {}

    torch.manual_seed(0)
    m8 = FP8Linear(D, D)
    m8.train()
    ref = torch.nn.Linear(D, D, bias=False)
    with torch.no_grad():
        nt = (D + 63) // 64
        ref.weight.copy_(torch.cat([decode_tile(m8.w8, m8.sc, 0, D, t) for t in range(nt)], 1))
    x = torch.randn(R, D)
    g = torch.randn(R, D)
    out["fp8_fwd"] = med(lambda: m8(x), args.iters)
    out["fp32_fwd"] = med(lambda: ref(x), args.iters)

    def fp8_bwd():
        xx = x.clone().requires_grad_(True)
        m8(xx).backward(g)

    out["fp8_bwd"] = med(fp8_bwd, args.iters)

    def ref_bwd():
        xx = x.clone().requires_grad_(True)
        ref(xx).backward(g)

    out["fp32_bwd"] = med(ref_bwd, args.iters)

    cfg = LinearConfig(vocab_size=2000, d_model=D, n_layer=1, n_heads=args.heads)
    blk = Block(cfg).eval()
    xb = torch.randn(args.batch, args.ctx, D).to(torch.bfloat16)
    out["attention"] = med(lambda: blk.att(blk.n1(xb)), args.iters)
    out["ffn"] = med(lambda: blk.ffn(xb), args.iters)
    out["rmsnorms"] = med(lambda: (blk.n1(xb), blk.n2(xb.float()), blk.n3(xb), blk.n4(xb.float()), blk.n5(xb)), args.iters)
    a = torch.randn_like(xb)
    out["residual"] = med(lambda: (xb.float() + a.float()).to(xb.dtype), args.iters)

    model = SmaulLinear(LinearConfig(vocab_size=2000, d_model=D, n_layer=args.layers, n_heads=args.heads))
    model.train()
    opt = Lion(list(model.parameters()), lr=2e-4)
    ids = torch.randint(0, 2000, (args.batch, args.ctx))

    def full_step():
        opt.zero_grad(model)
        _, loss = model(ids, ids)
        loss.backward()
        opt.step(model)

    # Time the full step BEFORE the requant micro-bench mutates weights,
    # and use median (not mean) consistently with the other ops.
    def _one_full_step_ms():
        t0 = time.perf_counter()
        full_step()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(3):
        full_step()
    step_ms = statistics.median(_one_full_step_ms() for _ in range(args.iters))
    tps = ids.numel() / (step_ms / 1e3) if step_ms > 0 else 0.0

    # Requant bench on a detached clone so the timed model above is untouched.
    import copy
    m8r = copy.deepcopy(fp8_modules(model)[0][1])
    upd = torch.randn(m8r.out_f, m8r.in_f) * 1e-4
    out["requant"] = med(lambda: m8r.requant(upd, 0.0), args.iters)

    # Stored bytes: FP8 weights (u8) + scales (fp32) + norms/embed/head (fp32/bf16 actual).
    fp8b = sum(m.w8.numel() + m.sc.numel() * 4 for _, m in fp8_modules(model))
    fpb = sum(m.w8.numel() * 4 for _, m in fp8_modules(model))
    other = sum(p.numel() * p.element_size() for p in model.parameters()) - fpb
    # fpb already counts FP8 weights as fp32; other adds the rest.
    print(f"\n{'op':12s} {'FP8 ms':>9s} {'FP32 ms':>9s} {'ratio':>6s}")
    fwd_ratio = out['fp8_fwd'] / out['fp32_fwd'] if out['fp32_fwd'] else float('nan')
    bwd_ratio = out['fp8_bwd'] / out['fp32_bwd'] if out['fp32_bwd'] else float('nan')
    print(f"{'linear_fwd':12s} {out['fp8_fwd']:9.2f} {out['fp32_fwd']:9.2f} {fwd_ratio:6.2f}x")
    print(f"{'linear_bwd':12s} {out['fp8_bwd']:9.2f} {out['fp32_bwd']:9.2f} {bwd_ratio:6.2f}x")
    for k in ("attention", "ffn", "rmsnorms", "residual", "requant"):
        print(f"{k:12s} {out[k]:9.2f}")
    print(f"\nfull step {step_ms:.0f}ms | {tps:.0f} tok/s | RSS {rss_mb():.0f}MB | "
          f"stored FP8 {fp8b/1048576:.1f}MiB vs FP32 {fpb/1048576:.1f}MiB (+other {other/1048576:.1f}MiB)")


if __name__ == "__main__":
    main()

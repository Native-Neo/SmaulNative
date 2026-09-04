#!/usr/bin/env python3
"""cpu/benchmark_full.py -- Reproducible CPU training benchmark for SmaulNative.

Measures: forward time, backward time, optimizer time, total step time, tokens/sec,
CPU utilization (wall-clock estimate), peak RAM.

Usage:
    python cpu/benchmark_full.py [--ctx_len 512] [--steps 3] [--threads 2]

Compilation is enabled by default. Pass --no-compile to measure eager PyTorch.
"""
import argparse
import os
import resource
import sys
import time
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
_cpu = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
if _cpu not in sys.path:
    sys.path.insert(0, _cpu)

p = argparse.ArgumentParser()
p.add_argument("--ctx_len", type=int, default=512)
p.add_argument("--steps", type=int, default=3)
p.add_argument("--threads", type=int, default=None)
p.add_argument("--compile", dest="compile", action="store_true", default=True)
p.add_argument("--no-compile", dest="compile", action="store_false")
args = p.parse_args()

_default_threads = str(os.environ.get("SMAUL_CPU_THREADS") or max(1, (os.cpu_count() or 2) // 2))
os.environ.setdefault("OMP_NUM_THREADS", _default_threads)
os.environ.setdefault("MKL_NUM_THREADS", _default_threads)
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "AVX")

import torch
from cpu_backend import NativeLion, configure
from rwkv_x_core import RWKVXConfig, RWKVXModel

threads = configure(args.threads)
torch.manual_seed(42)

torch.set_float32_matmul_precision("high")

cfg = RWKVXConfig(
    vocab_size=65536, n_embd=832, n_layer=17, head_size=64,
    n_moba_layer=5, ctx_len_hint=args.ctx_len, wkv_chunk_size=64
)
model = RWKVXModel(cfg)
if args.compile:
    print("[COMPILE] torch.compile(model) ...")
    model = torch.compile(model)

optimizer = NativeLion(model.parameters(), lr=1e-4)

x = torch.randint(0, 65536, (1, args.ctx_len))
y = torch.randint(0, 65536, (1, args.ctx_len))
tok_per_step = args.ctx_len

print(f"\n{'='*60}")
print(f"SmaulNative CPU Benchmark  |  threads={threads}  ctx_len={args.ctx_len}")
print(f"Model: {model.num_parameters()/1e6:.1f}M params  |  compile={args.compile}")
print(f"{'='*60}")

# Warm-up
print("Warming up (1 step)...")
optimizer.zero_grad(set_to_none=True)
_, loss, _ = model(x, labels=y)
loss.backward()
optimizer.step()
model.zero_grad(set_to_none=True)

results = []
for step in range(args.steps):
    t_fwd0 = time.perf_counter()
    _, loss, _ = model(x, labels=y)
    t_fwd = time.perf_counter() - t_fwd0

    t_bwd0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t_bwd0

    t_opt0 = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    t_opt = time.perf_counter() - t_opt0

    total = t_fwd + t_bwd + t_opt
    tps = tok_per_step / total
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    results.append((t_fwd, t_bwd, t_opt, total, tps, rss))
    print(f"  step {step+1}/{args.steps}: fwd={t_fwd:.2f}s  bwd={t_bwd:.2f}s  opt={t_opt:.3f}s  total={total:.2f}s  {tps:.1f} tok/s  RAM={rss:.0f}MB")

if results:
    avg = [sum(r[i] for r in results)/len(results) for i in range(6)]
    print(f"\nAverage over {args.steps} steps:")
    print(f"  fwd={avg[0]:.2f}s  bwd={avg[1]:.2f}s  opt={avg[2]:.3f}s  total={avg[3]:.2f}s  {avg[4]:.1f} tok/s  peak_RAM={avg[5]:.0f}MB")

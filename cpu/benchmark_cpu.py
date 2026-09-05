#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import torch

_cpu_dir = str(Path(__file__).resolve().parent)
if _cpu_dir not in sys.path:
    sys.path.insert(0, _cpu_dir)

try:
    from cpu_backend import NativeLion, configure
except ImportError:
    from cpu.cpu_backend import NativeLion, configure

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=16_000_000)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--threads", type=int, default=None)
a = p.parse_args()
threads = configure(a.threads)
print(f"torch={torch.__version__} cpu={torch.get_num_threads()} threads={threads}")

p0 = torch.randn(a.size)
g = torch.randn_like(p0)
opt = NativeLion([p0], lr=1e-4)
for _ in range(3):
    p0.grad = g
    opt.step()

t0 = time.perf_counter()
for _ in range(a.steps):
    p0.grad = g
    opt.step()
dt = time.perf_counter() - t0
print(f"threads={threads} steps/s={a.steps/dt:.2f} elements/s={a.size*a.steps/dt/1e6:.1f}M")

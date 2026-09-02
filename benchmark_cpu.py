#!/usr/bin/env python3
import argparse
import time
import torch

from cpu_backend import NativeLion, configure

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=16_000_000)
p.add_argument("--steps", type=int, default=20)
p.add_argument("--threads", type=int, default=None)
a = p.parse_args()
threads = configure(a.threads)

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

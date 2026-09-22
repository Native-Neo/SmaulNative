#!/usr/bin/env python3
"""Benchmark packed native RQT kernels without dequantizing full weights."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rqt import FP4, FP6, RQTLinear, _native_rqt


def _time(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=[FP4, FP6], default=FP6)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--in-features", type=int, default=832)
    parser.add_argument("--out-features", type=int, default=832)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if min(args.rows, args.in_features, args.out_features, args.warmup, args.iters) < 1:
        parser.error("dimensions and iteration counts must be positive")
    if args.threads is not None:
        torch.set_num_threads(max(1, args.threads))

    ext = _native_rqt()
    if ext is None or ext is False:
        raise RuntimeError("native RQT extension is unavailable")
    layer = RQTLinear(nn.Linear(args.in_features, args.out_features, bias=False), args.bits)
    x = torch.randn(args.rows, args.in_features)
    grad = torch.randn(args.rows, args.out_features)
    avg = torch.zeros(args.out_features, args.in_features)
    update = torch.randn(args.out_features, args.in_features)

    forward_ms = _time(
        lambda: ext.rqt_linear_forward(x, layer.packed, layer.scale, args.in_features, args.out_features, args.bits),
        args.warmup, args.iters,
    )
    backward_ms = _time(
        lambda: ext.rqt_linear_backward_input(grad, layer.packed, layer.scale, args.in_features, args.out_features, args.bits),
        args.warmup, args.iters,
    )
    lion_ms = _time(
        lambda: ext.rqt_lion_step(layer.packed, layer.scale, update, avg, args.in_features, args.out_features,
                                  args.bits, 1e-4, 0.9, 0.99, 1e-6),
        args.warmup, args.iters,
    )
    print(f"bits=FP{args.bits} rows={args.rows} in={args.in_features} out={args.out_features} threads={torch.get_num_threads()}")
    print(f"forward_ms={forward_ms:.3f} backward_input_ms={backward_ms:.3f} lion_step_ms={lion_ms:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-prompt CLI for SmaulLinear (see infer_cli.py for interactive mode)."""
import argparse
import time

from inference import LinearInference


def main():
    a = argparse.ArgumentParser(description="SmaulLinear single-prompt CLI")
    a.add_argument("--model", default="./runs/linear")
    a.add_argument("--prompt", default="Hello world")
    a.add_argument("--max", type=int, default=64)
    a.add_argument("--temp", type=float, default=0.7)
    a.add_argument("--topk", type=int, default=50)
    a.add_argument("--topp", type=float, default=0.95)
    a.add_argument("--rep", type=float, default=1.05)
    a.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    a.add_argument("--architecture", default=None, choices=["rawr", "plain"],
                   help="Expected architecture (default: auto-detect from checkpoint)")
    a.add_argument("--embedding-storage", default=None, choices=["ram", "mmap"],
                   help="Embedding backend override (default: checkpoint's)")
    args = a.parse_args()
    if not 1 <= args.max <= 65536:
        a.error("--max must be in [1, 65536]")
    if args.temp < 0 or args.topk < 0 or not 0.0 < args.topp <= 1.0 or args.rep <= 0:
        a.error("invalid sampling args (--temp>=0, --topk>=0, 0<--topp<=1, --rep>0)")
    if len(args.prompt) > 200_000:
        a.error("--prompt too long")
    engine = LinearInference(args.model, device=args.device, dtype=args.dtype,
                             architecture=args.architecture,
                             embedding_storage=args.embedding_storage)
    t0 = time.perf_counter()
    text = engine.generate(args.prompt, max_new_tokens=args.max, temperature=args.temp,
                           top_k=args.topk, top_p=args.topp, repetition_penalty=args.rep)
    el = time.perf_counter() - t0
    n_tok = len(engine.encode(text))
    print(text)
    print(f"[{n_tok} tokens | {el:.2f}s | {n_tok / max(el, 1e-6):.2f} tok/s]")


if __name__ == "__main__":
    main()

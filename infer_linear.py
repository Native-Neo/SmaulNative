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
    args = a.parse_args()
    engine = LinearInference(args.model)
    t0 = time.perf_counter()
    text = engine.generate(args.prompt, max_new_tokens=args.max, temperature=args.temp,
                           top_k=args.topk, top_p=args.topp, repetition_penalty=args.rep)
    el = time.perf_counter() - t0
    print(text)
    print(f"[{len(engine.encode(text))} tokens | {el:.2f}s | {len(engine.encode(text)) / max(el, 1e-6):.2f} tok/s]")


if __name__ == "__main__":
    main()

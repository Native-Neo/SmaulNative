#!/usr/bin/env python3
"""Interactive CLI for SmaulLinear."""

import argparse
import time

from inference import LinearInference


def main():
    p = argparse.ArgumentParser(description="SmaulLinear CLI")
    p.add_argument("--model", default="./runs/linear")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16"])
    p.add_argument("--architecture", default=None, choices=["rawr", "plain"],
                   help="Expected architecture (default: auto-detect from checkpoint)")
    p.add_argument("--embedding-storage", default=None, choices=["ram", "mmap"],
                   help="Embedding backend override (default: checkpoint's)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repeat-penalty", type=float, default=1.05)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--max-history", type=int, default=40,
                   help="Max chat turns kept (oldest dropped with notice)")
    p.add_argument("--prompt", default=None, help="Run one prompt and exit instead of interactive chat")
    p.add_argument("--max", type=int, default=64, help="Max new tokens in single-prompt mode")
    p.add_argument("--temp", type=float, default=0.7, help="Temperature in single-prompt mode")
    p.add_argument("--topk", type=int, default=50, help="Top-k in single-prompt mode")
    p.add_argument("--topp", type=float, default=0.95, help="Top-p in single-prompt mode")
    p.add_argument("--rep", type=float, default=1.05, help="Repetition penalty in single-prompt mode")
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 65536:
        p.error("--max-tokens must be in [1, 65536]")
    if not 1 <= args.max <= 65536:
        p.error("--max must be in [1, 65536]")
    if args.temp < 0 or args.topk < 0 or not 0.0 < args.topp <= 1.0 or args.rep <= 0:
        p.error("invalid single-prompt sampling args")
    if args.prompt is not None and len(args.prompt) > 200_000:
        p.error("--prompt too long")
    if args.temperature < 0 or args.top_k < 0 or not 0.0 < args.top_p <= 1.0 or args.repeat_penalty <= 0:
        p.error("invalid sampling args")
    if args.max_history < 2:
        p.error("--max-history must be >= 2")

    engine = LinearInference(args.model, args.device, args.dtype,
                             architecture=args.architecture,
                             embedding_storage=args.embedding_storage)
    messages: list = []
    system = "You are a helpful local AI assistant. Be concise, accurate, and practical."
    print(f"SmaulLinear | {engine.vocab_size:,} vocab | {engine.device}")
    if args.prompt is not None:
        started = time.perf_counter()
        text = engine.generate(args.prompt, max_new_tokens=args.max, temperature=args.temp,
                               top_k=args.topk, top_p=args.topp,
                               repetition_penalty=args.rep)
        elapsed = time.perf_counter() - started
        tokens = len(engine.encode(text))
        print(text)
        print(f"[{tokens} tokens | {elapsed:.2f}s | {tokens / max(elapsed, 1e-6):.2f} tok/s]")
        return

    print("Commands: /clear, /system <text>, /exit")

    while True:
        try:
            user = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user == "/exit":
            return
        if user == "/clear":
            messages.clear()
            print("[conversation cleared]")
            continue
        if user.startswith("/system "):
            system = user[8:].strip()
            print("[system prompt updated]")
            continue

        messages.append({"role": "user", "content": user})
        if len(messages) > args.max_history:
            # Drop oldest turns (keep pairs) and warn: engine window is 512
            # tokens, so unbounded history silently forgets anyway.
            drop = len(messages) - args.max_history
            del messages[:drop]
            print(f"[history trimmed: dropped {drop} oldest turn(s)]")
        try:
            prompt = engine.chat_prompt(messages, system)
        except ValueError as exc:
            print(f"[error] {exc}")
            messages.pop()
            continue
        print("\nAssistant > ", end="", flush=True)
        started = time.perf_counter()
        answer = []
        interrupted = False
        try:
            for chunk in engine.stream(prompt, max_new_tokens=args.max_tokens, temperature=args.temperature,
                                       top_k=args.top_k, top_p=args.top_p, repetition_penalty=args.repeat_penalty):
                print(chunk, end="", flush=True)
                answer.append(chunk)
        except KeyboardInterrupt:
            print("\n[stopped]")
            interrupted = True
        elapsed = time.perf_counter() - started
        text = "".join(answer)
        if interrupted and not text.strip():
            # Do not save empty interrupted answers as full turns.
            messages.pop()
            continue
        if interrupted:
            text += " [stopped]"
        messages.append({"role": "assistant", "content": text})
        tokens = len(engine.encode(text)) if text else 0
        print(f"\n[{tokens} tokens | {elapsed:.2f}s | {tokens / max(elapsed, 1e-6):.2f} tok/s]")


if __name__ == "__main__":
    main()

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
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 65536:
        p.error("--max-tokens must be in [1, 65536]")
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

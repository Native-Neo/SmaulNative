#!/usr/bin/env python3
"""Interactive CLI for SmaulNative RWKV-X."""

import argparse
import sys
import time

from inference import RWKVXInference


def main():
    p = argparse.ArgumentParser(description="SmaulNative RWKV-X CLI")
    p.add_argument("--model", default="./SmaulNative")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repeat-penalty", type=float, default=1.05)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    engine = RWKVXInference(args.model, args.device, args.dtype)
    messages = []
    system = "You are a helpful local AI assistant. Be concise, accurate, and practical."
    print(f"SmaulNative RWKV-X | {engine.vocab_size:,} vocab | {engine.device}")
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
        prompt = engine.chat_prompt(messages, system)
        print("\nAssistant > ", end="", flush=True)
        started = time.perf_counter()
        answer = []
        try:
            for chunk in engine.stream(prompt, max_new_tokens=args.max_tokens,
                                       temperature=args.temperature, top_k=args.top_k,
                                       top_p=args.top_p, repetition_penalty=args.repeat_penalty):
                print(chunk, end="", flush=True)
                answer.append(chunk)
        except KeyboardInterrupt:
            print("\n[stopped]")
        elapsed = time.perf_counter() - started
        text = "".join(answer)
        messages.append({"role": "assistant", "content": text})
        tokens = len(engine.encode(text))
        print(f"\n[{tokens} tokens | {elapsed:.2f}s | {tokens / max(elapsed, 1e-6):.2f} tok/s]")


if __name__ == "__main__":
    main()

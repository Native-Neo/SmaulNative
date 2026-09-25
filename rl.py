#!/usr/bin/env python3
"""Collect human preferences and GRPO-train SmaulLinear from those choices."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from smaul_linear import SmaulLinear
from tokenizer import SmaulTokenizer
from train import Lion


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass


class SmaulRL:
    MODEL_WINDOW = 512

    def __init__(self, model_dir: str, work_dir: str = "./rl", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.device = torch.device(device)
        except RuntimeError as exc:
            raise ValueError(f"invalid device {device!r}: {exc}") from exc
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        self.tokenizer = SmaulTokenizer.from_file(self.model_dir / "tokenizer.json")
        load_dir = self.work_dir / "policy" if (self.work_dir / "policy").exists() else self.model_dir
        if load_dir != self.model_dir:
            print(f"[RL] resuming policy from {load_dir} (not {self.model_dir})")
        self.model = SmaulLinear.from_pretrained(load_dir).to(self.device)
        # Tokenizer must match the loaded policy, not just model_dir.
        tok_vocab = self.tokenizer.get_vocab_size()
        if tok_vocab != self.model.cfg.vocab_size:
            raise ValueError(f"tokenizer vocab ({tok_vocab}) != model vocab ({self.model.cfg.vocab_size})")
        self.model.train()
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.preference_path = self.work_dir / "preferences.jsonl"
        self._opt: Optional[Lion] = None

    def _encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @staticmethod
    def _filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for policy sampling")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        logits = logits.float() / temperature
        if top_k > 0 and top_k < logits.numel():
            cutoff = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, -float("inf"))
        if 0 < top_p < 1:
            values, indices = torch.sort(logits, descending=True)
            probs = F.softmax(values, -1)
            remove = torch.cumsum(probs, -1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits = logits.masked_fill(torch.zeros_like(remove).scatter(0, indices, remove), -float("inf"))
        return logits

    @classmethod
    def _sample(cls, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> Tuple[int, float]:
        filtered = cls._filter_logits(logits, temperature, top_k, top_p)
        if not bool(torch.isfinite(filtered).any()):
            # All-masked: fall back to greedy on unfiltered logits instead of NaN crash.
            filtered = logits.float()
        log_probs = F.log_softmax(filtered, -1)
        if not bool(torch.isfinite(log_probs).any()):
            raise RuntimeError("sampling failed: all logits non-finite")
        token = int(torch.multinomial(log_probs.exp(), 1))
        return token, float(log_probs[token])

    @staticmethod
    def _validate_gen(max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> None:
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError(f"max_new_tokens must be in [1, 4096], got {max_new_tokens}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for policy sampling")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
                 seed: Optional[int] = None) -> Tuple[str, List[int], List[float]]:
        self._validate_gen(max_new_tokens, temperature, top_k, top_p)
        if seed is not None:
            _seed_all(seed)
        was_training = self.model.training
        self.model.eval()
        try:
            prompt_ids = self._encode(prompt) or [self.bos_id if self.bos_id is not None else self.eos_id]
            if len(prompt_ids) > self.MODEL_WINDOW:
                print(f"[WARN] prompt truncated to last {self.MODEL_WINDOW} tokens ({len(prompt_ids)} provided)")
            ids = prompt_ids[-self.MODEL_WINDOW:]
            logits, _ = self.model(torch.tensor([ids], dtype=torch.long, device=self.device))
            response, old_logprobs = [], []
            for _ in range(max_new_tokens):
                token, logprob = self._sample(logits[0, -1], temperature, top_k, top_p)
                if token == self.eos_id:
                    break
                response.append(token)
                old_logprobs.append(logprob)
                ids = (ids + [token])[-self.MODEL_WINDOW:]
                logits, _ = self.model(torch.tensor([ids], dtype=torch.long, device=self.device))
            return self._decode(response), response, old_logprobs
        finally:
            self.model.train(was_training)

    def candidates(self, prompt: str, count: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float) -> List[Dict]:
        if count < 1 or count > 64:
            raise ValueError(f"count must be in [1, 64], got {count}")
        self._validate_gen(max_new_tokens, temperature, top_k, top_p)
        result = []
        for i in range(count):
            text, tokens, old_logprobs = self.generate(prompt, max_new_tokens, temperature, top_k, top_p, seed=random.randrange(2**31))
            result.append({"id": i, "text": text, "tokens": tokens, "old_logprobs": old_logprobs})
        return result

    @staticmethod
    def _show(candidates: List[Dict]):
        print("\n" + "=" * 80)
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}]\n{candidate['text']}\n")

    def _pick(self, candidates: List[Dict]) -> int:
        self._show(candidates)
        while True:
            try:
                raw = input(f"Choose best response [1-{len(candidates)}]: ").strip()
            except EOFError:
                print("\n[RL] stdin closed; keeping first response")
                return 0
            try:
                choice = int(raw) - 1
                if 0 <= choice < len(candidates):
                    return choice
            except ValueError:
                pass
            print("Invalid choice.")

    def _save(self, prompt: str, candidates: List[Dict], chosen: int):
        import os
        record = {"prompt": prompt, "responses": [c["text"] for c in candidates], "chosen": chosen, "source": "human"}
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with self.preference_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass

    def _logprob(self, prompt: str, response_tokens: List[int], temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        # Use the SAME sliding window as sampling: prompt truncated to window,
        # response truncated so prompt+response fits without OOM.
        prompt_ids = self._encode(prompt) or [self.bos_id if self.bos_id is not None else self.eos_id]
        if not response_tokens:
            return torch.empty(0, device=self.device)
        prompt_ids = prompt_ids[-self.MODEL_WINDOW:]
        max_resp = max(1, self.MODEL_WINDOW - len(prompt_ids) + 1)
        if len(response_tokens) > max_resp:
            print(f"[WARN] _logprob truncating response {len(response_tokens)} -> {max_resp} to fit window")
            response_tokens = response_tokens[:max_resp]
        ids = torch.tensor([prompt_ids + response_tokens], dtype=torch.long, device=self.device)
        logits, _ = self.model(ids)
        start = len(prompt_ids) - 1
        token_logits = logits[0, start:start + len(response_tokens)]
        return torch.stack([F.log_softmax(self._filter_logits(row, temperature, top_k, top_p), -1)[tok]
                            for row, tok in zip(token_logits, response_tokens)])

    def grpo_step(self, prompt: str, candidates: List[Dict], chosen: int, lr: float, clip: float, kl_coef: float,
                   temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> float:
        if not 0 <= chosen < len(candidates):
            raise ValueError("chosen response is out of range")
        if len(candidates) < 2:
            raise ValueError("grpo_step needs at least 2 candidates (std of 1 is NaN)")
        if not 0 < lr < 1 or not 0 <= clip < 1 or not 0 <= kl_coef < 10:
            raise ValueError(f"invalid hyperparams lr={lr} clip={clip} kl_coef={kl_coef}")
        self._validate_gen(256, temperature, top_k, top_p)
        rewards = torch.full((len(candidates),), -1.0, device=self.device)
        rewards[chosen] = 1.0
        std = rewards.std(unbiased=False)
        advantages = (rewards - rewards.mean()) / std.clamp_min(1e-6)
        if not bool(torch.isfinite(advantages).all()):
            raise ValueError("non-finite advantages; check rewards")
        # Reuse one optimizer so momentum persists across steps (fresh Lion
        # every step discards convergence state).
        if self._opt is None:
            self._opt = Lion(list(self.model.parameters()), lr=lr, wd=0.0)
        else:
            self._opt.lr = lr
        opt = self._opt
        losses = []
        for candidate, advantage in zip(candidates, advantages):
            if not candidate["tokens"]:
                continue
            new_logprobs = self._logprob(prompt, candidate["tokens"], temperature, top_k, top_p)
            old_logprobs = torch.tensor(candidate["old_logprobs"], dtype=new_logprobs.dtype, device=self.device)
            if old_logprobs.numel() != new_logprobs.numel():
                raise ValueError("stored and recomputed token log-probabilities have different lengths")
            log_ratio = new_logprobs - old_logprobs
            ratio = torch.exp(log_ratio.clamp(-20, 20))
            clipped = ratio.clamp(1 - clip, 1 + clip)
            adv = advantage.detach().expand_as(ratio)
            losses.append(-torch.minimum(ratio * adv, clipped * adv).mean() + kl_coef * -log_ratio.mean())
        if not losses:
            raise ValueError("no non-empty candidates to train on (not success=0.0)")
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss.detach())):
            raise ValueError(f"non-finite GRPO loss {float(loss):.3f}; checkpoint NOT saved")
        opt.zero_grad(self.model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        opt.step(self.model)
        policy_dir = self.work_dir / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(policy_dir)
        (policy_dir / "tokenizer.json").write_text((self.model_dir / "tokenizer.json").read_text(encoding="utf-8"), encoding="utf-8")
        return float(loss.detach())

    def run(self, prompts: List[str], count: int, max_new_tokens: int, temperature: float, top_k: int,
            top_p: float, lr: float, clip: float, kl_coef: float):
        if count < 2:
            raise ValueError("--responses must be at least 2")
        for prompt in prompts:
            candidates = self.candidates(prompt, count, max_new_tokens, temperature, top_k, top_p)
            chosen = self._pick(candidates)
            self._save(prompt, candidates, chosen)
            loss = self.grpo_step(prompt, candidates, chosen, lr, clip, kl_coef, temperature, top_k, top_p)
            print(f"[SAVED] preference={chosen + 1}/{count}")
            print(f"[RL] loss={loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Human preference collection and RL for SmaulLinear")
    parser.add_argument("--model_dir", default="./runs/linear")
    parser.add_argument("--work_dir", default="./rl")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--rl_lr", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    SmaulRL(args.model_dir, args.work_dir, args.device).run(args.prompt, args.responses, args.max_new_tokens,
        args.temperature, args.top_k, args.top_p, args.rl_lr, args.clip, args.kl_coef)


if __name__ == "__main__":
    main()

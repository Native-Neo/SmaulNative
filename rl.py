#!/usr/bin/env python3
"""Collect human preferences and RL-train SmaulNative from those choices."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel
from tokenizer import SmaulTokenizer


class SmaulRL:
    def __init__(self, model_dir: str, work_dir: str = "./rl", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = SmaulTokenizer.from_file(self.model_dir / "tokenizer.json")
        policy_dir = self.work_dir / "policy"
        load_dir = policy_dir if policy_dir.exists() else self.model_dir
        self.model = RWKVXModel.from_pretrained(load_dir).to(self.device)
        self.model.train()
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.preference_path = self.work_dir / "preferences.jsonl"

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> Tuple[int, float]:
        logits = logits.float()
        if temperature <= 0:
            probs = F.softmax(logits, -1)
            token = int(probs.argmax())
            return token, float(torch.log(probs[token].clamp_min(1e-12)))

        logits = logits / temperature
        if top_k > 0 and top_k < logits.numel():
            cutoff = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < cutoff, -float("inf"))
        if 0 < top_p < 1:
            values, indices = torch.sort(logits, descending=True)
            probs = F.softmax(values, -1)
            remove = torch.cumsum(probs, -1) > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            mask = torch.zeros_like(remove).scatter(0, indices, remove)
            logits = logits.masked_fill(mask, -float("inf"))

        log_probs = F.log_softmax(logits, -1)
        token = int(torch.multinomial(log_probs.exp(), 1))
        return token, float(log_probs[token])

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: Optional[int] = None,
    ) -> Tuple[str, List[int], float]:
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id if self.bos_id is not None else self.eos_id]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, _, state = self.model(ids, state=None, use_cache=True, return_logits=True)
        response = []
        old_logprob = 0.0
        for _ in range(max_new_tokens):
            token, logprob = self._sample(logits[0, -1], temperature, top_k, top_p)
            if token == self.eos_id:
                break
            response.append(token)
            old_logprob += logprob
            logits, _, state = self.model(
                torch.tensor([[token]], device=self.device),
                state=state,
                use_cache=True,
                return_logits=True,
            )
        return self._decode(response), response, old_logprob

    def candidates(
        self,
        prompt: str,
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[Dict]:
        result = []
        for i in range(count):
            text, tokens, old_logprob = self.generate(
                prompt, max_new_tokens, temperature, top_k, top_p, seed=random.randrange(2**31)
            )
            result.append({"id": i, "text": text, "tokens": tokens, "old_logprob": old_logprob})
        return result

    @staticmethod
    def _show(candidates: List[Dict]):
        print("\n" + "=" * 80)
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}]\n{candidate['text']}\n")

    def _pick(self, candidates: List[Dict]) -> int:
        self._show(candidates)
        while True:
            raw = input(f"Choose best response [1-{len(candidates)}]: ").strip()
            try:
                choice = int(raw) - 1
                if 0 <= choice < len(candidates):
                    return choice
            except ValueError:
                pass
            print("Invalid choice.")

    def _save(self, prompt: str, candidates: List[Dict], chosen: int):
        record = {
            "prompt": prompt,
            "responses": [candidate["text"] for candidate in candidates],
            "chosen": chosen,
            "source": "human",
        }
        with self.preference_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _logprob(self, prompt: str, response_tokens: List[int]) -> torch.Tensor:
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id if self.bos_id is not None else self.eos_id]
        if not response_tokens:
            return torch.empty(0, device=self.device)

        ids = torch.tensor([prompt_ids + response_tokens], dtype=torch.long, device=self.device)
        logits, _, _ = self.model(ids, state=None, use_cache=False, return_logits=True)
        start = len(prompt_ids) - 1
        token_logits = logits[0, start:start + len(response_tokens)]
        targets = torch.tensor(response_tokens, dtype=torch.long, device=self.device)
        return F.log_softmax(token_logits.float(), -1).gather(1, targets[:, None]).squeeze(1)

    def grpo_step(
        self,
        prompt: str,
        candidates: List[Dict],
        chosen: int,
        lr: float,
        clip: float,
        kl_coef: float,
    ) -> float:
        if not 0 <= chosen < len(candidates):
            raise ValueError("chosen response is out of range")

        rewards = torch.full((len(candidates),), -1.0, device=self.device)
        rewards[chosen] = 1.0
        advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        losses = []

        for candidate, advantage in zip(candidates, advantages):
            if not candidate["tokens"]:
                continue
            new_logprob = self._logprob(prompt, candidate["tokens"])
            old_mean = candidate["old_logprob"] / len(candidate["tokens"])
            new_mean = new_logprob.mean()
            ratio = torch.exp(new_mean - old_mean)
            clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
            policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage)
            kl = (new_mean - old_mean).pow(2)
            losses.append(policy_loss + kl_coef * kl)

        if not losses:
            return 0.0

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        policy_dir = self.work_dir / "policy"
        policy_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(policy_dir, dtype="fp32", include_upstream=False)
        return float(loss.detach())

    def run(
        self,
        prompts: List[str],
        count: int,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        lr: float,
        clip: float,
        kl_coef: float,
    ):
        if count < 2:
            raise ValueError("--responses must be at least 2")
        for prompt in prompts:
            candidates = self.candidates(prompt, count, max_new_tokens, temperature, top_k, top_p)
            chosen = self._pick(candidates)
            self._save(prompt, candidates, chosen)
            loss = self.grpo_step(prompt, candidates, chosen, lr, clip, kl_coef)
            print(f"[SAVED] preference={chosen + 1}/{count}")
            print(f"[RL] loss={loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Human preference collection and RL for SmaulNative")
    parser.add_argument("--model_dir", default="./SmaulNative")
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
    SmaulRL(args.model_dir, args.work_dir, args.device).run(
        args.prompt,
        args.responses,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.rl_lr,
        args.clip,
        args.kl_coef,
    )


if __name__ == "__main__":
    main()

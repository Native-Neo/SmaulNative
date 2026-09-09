#!/usr/bin/env python3
"""Train from rl.py preferences and run verified automatic RL."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rwkv_x_core import RWKVXModel
from tokenizer import SmaulTokenizer


class PreferenceModel(nn.Module):
    """Small CPU-friendly model that learns which responses the user prefers."""
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.scorer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.scorer(pooled).squeeze(-1)


class AutoRL:
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
        self.preference_model_path = self.work_dir / "preference_model.pt"
        self.preference_meta_path = self.work_dir / "preference_model.meta.json"
        self.preference_model = PreferenceModel(self.tokenizer.get_vocab_size()).to(self.device)
        self.preference_trained = 0
        self._load_preference_model()

    def _load_preference_model(self):
        if self.preference_model_path.exists():
            state = torch.load(self.preference_model_path, map_location=self.device, weights_only=True)
            self.preference_model.load_state_dict(state)
        if self.preference_meta_path.exists():
            try:
                self.preference_trained = max(0, int(json.loads(self.preference_meta_path.read_text()).get("records", 0)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.preference_trained = 0

    def _save_preference_model(self, record_count: int):
        torch.save(self.preference_model.state_dict(), self.preference_model_path)
        self.preference_meta_path.write_text(json.dumps({"records": record_count}))
        self.preference_trained = record_count

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @staticmethod
    def _filter_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 for policy sampling")
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
        log_probs = F.log_softmax(cls._filter_logits(logits, temperature, top_k, top_p), -1)
        token = int(torch.multinomial(log_probs.exp(), 1))
        return token, float(log_probs[token])

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id if self.bos_id is not None else self.eos_id]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        logits, _, state = self.model(ids, state=None, use_cache=True, return_logits=True)
        response, old_logprobs = [], []
        for _ in range(max_new_tokens):
            token, logprob = self._sample(logits[0, -1], temperature, top_k, top_p)
            if token == self.eos_id:
                break
            response.append(token)
            old_logprobs.append(logprob)
            logits, _, state = self.model(torch.tensor([[token]], device=self.device), state=state, use_cache=True, return_logits=True)
        return self._decode(response), response, old_logprobs

    def candidates(self, prompt: str, count: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float):
        candidates = []
        for i in range(count):
            torch.manual_seed(random.randrange(2**31))
            text, tokens, old_logprobs = self.generate(prompt, max_new_tokens, temperature, top_k, top_p)
            candidates.append({"id": i, "text": text, "tokens": tokens, "old_logprobs": old_logprobs})
        return candidates

    def _batch(self, texts: List[str]):
        return self._batch_ids([self._encode(text) for text in texts])

    def _batch_ids(self, token_lists: List[List[int]]):
        max_len = max(1, max(map(len, token_lists)))
        tokens = torch.zeros(len(token_lists), max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row, ids in enumerate(token_lists):
            if ids:
                tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                mask[row, :len(ids)] = True
        return tokens, mask

    def _batch_pairs(self, prompt: str, responses: List[str]):
        prompt_ids = self._encode(prompt)
        sep = [self.eos_id] if self.eos_id is not None else []
        return self._batch_ids([prompt_ids + sep + self._encode(response) for response in responses])

    @torch.no_grad()
    def preference_scores(self, prompt: str, candidates: List[Dict]) -> torch.Tensor:
        tokens, mask = self._batch_pairs(prompt, [candidate["text"] for candidate in candidates])
        self.preference_model.eval()
        return self.preference_model(tokens, mask)

    def preference_count(self) -> int:
        if not self.preference_path.exists():
            return 0
        with self.preference_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def train_preferences(self, epochs: int, lr: float):
        if not self.preference_path.exists():
            print("[PREF] no preferences.jsonl found")
            return
        records = [json.loads(line) for line in self.preference_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            print("[PREF] no preference records found")
            return
        start = min(self.preference_trained, len(records))
        if start == len(records):
            print(f"[PREF] up to date ({len(records)} records)")
            return
        new_records = records[start:]
        optimizer = torch.optim.AdamW(self.preference_model.parameters(), lr=lr)
        self.preference_model.train()
        for epoch in range(epochs):
            random.shuffle(new_records)
            total = 0.0
            valid = 0
            for record in new_records:
                responses = record.get("responses", [])
                chosen = int(record.get("chosen", -1))
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or len(responses) < 2 or not 0 <= chosen < len(responses):
                    continue
                tokens, mask = self._batch_pairs(prompt, responses)
                scores = self.preference_model(tokens, mask)
                rejected = torch.cat((scores[:chosen], scores[chosen + 1:]))
                loss = -F.logsigmoid(scores[chosen] - rejected).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                valid += 1
            print(f"[PREF] epoch={epoch + 1}/{epochs} loss={total / max(1, valid):.5f}")
        self._save_preference_model(len(records))

    def _show(self, candidates: List[Dict]):
        print("\n" + "=" * 80)
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}]\n{candidate['text']}\n")

    def _verify(self, candidates: List[Dict], predicted: int) -> int:
        while True:
            answer = input("Did automated RL choose correctly? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                print(f"[AUTO] confirmed response {predicted + 1}/{len(candidates)}")
                return predicted
            if answer in ("n", "no"):
                self._show(candidates)
                while True:
                    raw = input(f"Which response is better? [1-{len(candidates)}]: ").strip()
                    try:
                        choice = int(raw) - 1
                        if 0 <= choice < len(candidates):
                            print(f"[CORRECTION] response {choice + 1}/{len(candidates)}")
                            return choice
                    except ValueError:
                        pass
                    print("Invalid choice.")
            print("Please answer yes or no.")

    def _save_preference(self, prompt: str, candidates: List[Dict], chosen: int, predicted: int):
        record = {"prompt": prompt, "responses": [candidate["text"] for candidate in candidates], "chosen": chosen,
                  "source": "auto_confirmed" if chosen == predicted else "human_correction", "predicted": predicted}
        with self.preference_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _logprob(self, prompt: str, response_tokens: List[int], temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id if self.bos_id is not None else self.eos_id]
        if not response_tokens:
            return torch.empty(0, device=self.device)
        ids = torch.tensor([prompt_ids + response_tokens], dtype=torch.long, device=self.device)
        logits, _, _ = self.model(ids, state=None, use_cache=False, return_logits=True)
        start = len(prompt_ids) - 1
        token_logits = logits[0, start:start + len(response_tokens)]
        token_logprobs = [F.log_softmax(self._filter_logits(row, temperature, top_k, top_p), -1)[token]
                          for row, token in zip(token_logits, response_tokens)]
        return torch.stack(token_logprobs)

    def grpo_step(self, prompt: str, candidates: List[Dict], chosen: int, lr: float, clip: float, kl_coef: float,
                   temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
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
            new_logprobs = self._logprob(prompt, candidate["tokens"], temperature, top_k, top_p)
            old_logprobs = torch.tensor(candidate["old_logprobs"], dtype=new_logprobs.dtype, device=self.device)
            if old_logprobs.numel() != new_logprobs.numel():
                raise ValueError("stored and recomputed token log-probabilities have different lengths")
            log_ratio = new_logprobs - old_logprobs
            ratio = torch.exp(log_ratio.clamp(-20, 20))
            clipped_ratio = ratio.clamp(1 - clip, 1 + clip)
            token_advantage = advantage.detach().expand_as(ratio)
            policy_loss = -torch.minimum(ratio * token_advantage, clipped_ratio * token_advantage).mean()
            sampled_kl = (torch.exp(log_ratio.clamp(-20, 20)) - log_ratio - 1).mean()
            losses.append(policy_loss + kl_coef * sampled_kl)
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

    def run(self, prompts: List[str], count: int, max_new_tokens: int, temperature: float, top_k: int, top_p: float,
            preference_epochs: int, preference_lr: float, rl_lr: float, clip: float, kl_coef: float, verify: bool):
        if count < 2:
            raise ValueError("--responses must be at least 2")
        print(f"[PREF] loading {self.preference_count()} preference records")
        self.train_preferences(preference_epochs, preference_lr)
        for prompt in prompts:
            candidates = self.candidates(prompt, count, max_new_tokens, temperature, top_k, top_p)
            scores = self.preference_scores(prompt, candidates)
            predicted = int(scores.argmax().item())
            chosen = self._verify(candidates, predicted) if verify else predicted
            self._save_preference(prompt, candidates, chosen, predicted)
            self.train_preferences(preference_epochs, preference_lr)
            loss = self.grpo_step(prompt, candidates, chosen, rl_lr, clip, kl_coef, temperature, top_k, top_p)
            print(f"[RL] loss={loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Automated preference learning and RL for SmaulNative")
    parser.add_argument("--model_dir", default="./SmaulNative")
    parser.add_argument("--work_dir", default="./rl")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--preference_epochs", type=int, default=3)
    parser.add_argument("--preference_lr", type=float, default=1e-3)
    parser.add_argument("--rl_lr", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    AutoRL(args.model_dir, args.work_dir, args.device).run(args.prompt, args.responses, args.max_new_tokens, args.temperature, args.top_k,
        args.top_p, args.preference_epochs, args.preference_lr, args.rl_lr, args.clip, args.kl_coef, not args.no_verify)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train from rl.py preferences and run verified automatic RL for SmaulLinear."""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl import SmaulRL


class PreferenceModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.scorer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.scorer(pooled).squeeze(-1)


class AutoRL(SmaulRL):
    def __init__(self, model_dir: str, work_dir: str = "./rl", device: str = "auto"):
        super().__init__(model_dir, work_dir, device)
        self.preference_model_path = self.work_dir / "preference_model.pt"
        self.preference_meta_path = self.work_dir / "preference_model.meta.json"
        self.preference_model = PreferenceModel(self.tokenizer.get_vocab_size()).to(self.device)
        self.preference_trained = 0
        self._load_preference_model()

    def _load_preference_model(self):
        if self.preference_model_path.exists():
            self.preference_model.load_state_dict(torch.load(self.preference_model_path, map_location=self.device, weights_only=True))
        if self.preference_meta_path.exists():
            try:
                self.preference_trained = max(0, int(json.loads(self.preference_meta_path.read_text()).get("records", 0)))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.preference_trained = 0

    def _save_preference_model(self, record_count: int):
        torch.save(self.preference_model.state_dict(), self.preference_model_path)
        self.preference_meta_path.write_text(json.dumps({"records": record_count}))
        self.preference_trained = record_count

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
        return self._batch_ids([prompt_ids + sep + self._encode(r) for r in responses])

    @torch.no_grad()
    def preference_scores(self, prompt: str, candidates: List[Dict]) -> torch.Tensor:
        tokens, mask = self._batch_pairs(prompt, [c["text"] for c in candidates])
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
            total, valid = 0.0, 0
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

    def _verify(self, candidates: List[Dict], predicted: int) -> int:
        while True:
            answer = input("Did automated RL choose correctly? [Y/n]: ").strip().lower()
            if answer in ("", "y", "yes"):
                print(f"[AUTO] confirmed response {predicted + 1}/{len(candidates)}")
                return predicted
            if answer in ("n", "no"):
                SmaulRL._show(candidates)
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
        record = {"prompt": prompt, "responses": [c["text"] for c in candidates], "chosen": chosen,
                  "source": "auto_confirmed" if chosen == predicted else "human_correction", "predicted": predicted}
        with self.preference_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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
    parser = argparse.ArgumentParser(description="Automated preference learning and RL for SmaulLinear")
    parser.add_argument("--model_dir", default="./runs/linear")
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

#!/usr/bin/env python3
"""Train from rl.py preferences and run verified automatic RL for SmaulLinear."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl import SmaulRL


# Minimum human preference records required before unattended (--no-verify)
# auto-labeling is allowed. Below this the reward model is effectively random
# and training on its own argmax causes a self-reinforcing collapse loop.
MIN_HUMAN_PREFS_FOR_AUTO = 4

# Hard cap for a single (prompt + response) pair fed to the reward model.
# Prevents one huge prompt/response from OOMing the (n, max_len) batch tensor.
MAX_PREF_PAIR_LEN = 2048


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
            try:
                state = torch.load(self.preference_model_path, map_location=self.device, weights_only=True)
                self.preference_model.load_state_dict(state)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                print(f"[WARN] ignoring corrupt preference checkpoint {self.preference_model_path}: {exc}")
        if self.preference_meta_path.exists():
            try:
                meta = json.loads(self.preference_meta_path.read_text())
                self.preference_trained = max(0, int(meta.get("records", 0)))
                self._preference_hash = str(meta.get("sha256", ""))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                self.preference_trained = 0
                self._preference_hash = ""
        else:
            self._preference_hash = ""

    def _prefs_hash(self, lines: List[str]) -> str:
        h = hashlib.sha256()
        for line in lines:
            h.update(line.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    def _save_preference_model(self, record_count: int, raw_lines: List[str]):
        torch.save(self.preference_model.state_dict(), self.preference_model_path)
        meta = {"records": record_count, "sha256": self._prefs_hash(raw_lines)}
        self.preference_meta_path.write_text(json.dumps(meta))
        self.preference_trained = record_count
        self._preference_hash = meta["sha256"]

    def _batch_ids(self, token_lists: List[List[int]]):
        if not token_lists:
            raise ValueError("token_lists must not be empty")
        # Truncate each sequence first so one huge prompt/response cannot OOM
        # the dense (n, max_len) batch tensor.
        truncated = [ids[:MAX_PREF_PAIR_LEN] if len(ids) > MAX_PREF_PAIR_LEN else ids for ids in token_lists]
        max_len = max(1, max(map(len, truncated)))
        tokens = torch.zeros(len(truncated), max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row, ids in enumerate(truncated):
            if ids:
                tokens[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                mask[row, :len(ids)] = True
        return tokens, mask

    def _batch_pairs(self, prompt: str, responses: List[str]):
        if not responses:
            raise ValueError("responses must not be empty")
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

    def train_preferences(self, epochs: int, lr: float) -> int:
        """Incrementally train the reward model on new human records.

        Returns the total number of preference records seen (for meta tracking).
        Never marks a random/untrained model as trained: if there is nothing
        valid to train on, the checkpoint is left untouched.
        """
        if not self.preference_path.exists():
            print("[PREF] no preferences.jsonl found")
            return 0
        records = []
        raw_lines: List[str] = []
        bad_lines = 0
        try:
            with self.preference_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                        raw_lines.append(line.strip())
                    except json.JSONDecodeError:
                        bad_lines += 1
        except OSError as exc:
            print(f"[PREF] could not read {self.preference_path}: {exc}")
            return 0
        if bad_lines:
            print(f"[WARN] skipped {bad_lines} malformed preference line(s)")
        if not records:
            print("[PREF] no preference records found")
            return 0
        if epochs <= 0:
            print("[PREF] epochs<=0, skipping training (checkpoint untouched)")
            return len(records)
        # Hash-based resume: edited/reordered/truncated files retrain from
        # scratch instead of training on the wrong slice.
        start = min(self.preference_trained, len(records))
        prev_hash = getattr(self, "_preference_hash", "")
        if start > 0 and prev_hash:
            if self._prefs_hash(raw_lines[:start]) != prev_hash:
                print("[WARN] preferences file changed since last train; retraining from scratch")
                start = 0
        if start == len(records):
            print(f"[PREF] up to date ({len(records)} records)")
            return len(records)
        new_records = records[start:]
        # Mix a small replay sample to reduce forgetting (old behavior trained
        # new-only). Deterministic: seeded by file hash.
        replay: List[Dict] = []
        if start > 0:
            seed = int(hashlib.sha256("".join(raw_lines[:start]).encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            replay = rng.sample(records[:start], min(start, max(4, len(new_records) // 2)))
        train_pool = new_records + replay
        optimizer = torch.optim.AdamW(self.preference_model.parameters(), lr=lr)
        self.preference_model.train()
        trained_valid = 0
        for epoch in range(epochs):
            random.shuffle(train_pool)
            total, valid = 0.0, 0
            for record in train_pool:
                if not isinstance(record, dict):
                    continue
                responses = record.get("responses", [])
                chosen = int(record.get("chosen", -1))
                prompt = record.get("prompt")
                if not isinstance(prompt, str) or len(responses) < 2 or not 0 <= chosen < len(responses):
                    continue
                try:
                    tokens, mask = self._batch_pairs(prompt, responses)
                except (ValueError, RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    print(f"[WARN] skipping over-long/bad preference record: {exc}")
                    continue
                scores = self.preference_model(tokens, mask)
                rejected = torch.cat((scores[:chosen], scores[chosen + 1:]))
                loss = -F.logsigmoid(scores[chosen] - rejected).mean()
                if not torch.isfinite(loss.detach()):
                    print("[WARN] skipping non-finite preference loss")
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                valid += 1
            trained_valid += valid
            print(f"[PREF] epoch={epoch + 1}/{epochs} loss={total / max(1, valid):.5f}")
        if trained_valid == 0:
            print("[PREF] no valid records trained; checkpoint left untouched (still random)")
            return len(records)
        self._save_preference_model(len(records), raw_lines)
        return len(records)

    def _verify(self, candidates: List[Dict], predicted: int) -> int:
        while True:
            try:
                answer = input("Did automated RL choose correctly? [Y/n]: ").strip().lower()
            except EOFError:
                print("\n[AUTO] stdin closed; keeping predicted choice")
                return predicted
            if answer in ("", "y", "yes"):
                print(f"[AUTO] confirmed response {predicted + 1}/{len(candidates)}")
                return predicted
            if answer in ("n", "no"):
                SmaulRL._show(candidates)
                while True:
                    try:
                        raw = input(f"Which response is better? [1-{len(candidates)}]: ").strip()
                    except EOFError:
                        print("\n[AUTO] stdin closed; keeping predicted choice")
                        return predicted
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
        if max_new_tokens <= 0:
            raise ValueError("--max_new_tokens must be positive")
        print(f"[PREF] loading {self.preference_count()} preference records")
        self.train_preferences(preference_epochs, preference_lr)
        human_records = self.preference_count()
        reward_trained = self.preference_trained
        if not verify and (human_records < MIN_HUMAN_PREFS_FOR_AUTO or reward_trained <= 0):
            raise RuntimeError(
                f"Refusing unattended auto-labeling: found {human_records} human preference record(s) "
                f"(trained={reward_trained}); need >= {MIN_HUMAN_PREFS_FOR_AUTO} human records before "
                f"--no-verify. Collect preferences with rl.py first, or run with verification enabled."
            )
        if not verify:
            print(f"[PREF] auto-labeling with reward model trained on {reward_trained} record(s)")
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

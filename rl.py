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
        # No explicit clip_grad_norm_ here: Lion.step() already applies global
        # norm clipping over dense grads AND FP8 _gw. Pre-clipping dense-only
        # would double-clip dense params while leaving _gw single-clipped.
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


def main_human():
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


def main_auto():
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


def main():
    import sys
    if "--auto" in sys.argv:
        sys.argv.remove("--auto")
        main_auto()
    else:
        main_human()


if __name__ == "__main__":
    main()

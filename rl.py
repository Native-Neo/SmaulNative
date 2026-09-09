#!/usr/bin/env python3
"""Human-in-the-loop GRPO for SmaulNative.

The user ranks a group of responses. The selected response receives the highest
relative reward and the policy is updated with a group-normalized objective.
After enough preferences, a tiny learned preference model can rank responses
without asking the user every time.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tensordict import TensorDict
    from torchrl.data import LazyTensorStorage, ReplayBuffer
except ImportError as exc:
    raise RuntimeError("Install TorchRL with: pip install torchrl tensordict") from exc

from rwkv_x_core import RWKVXModel
from tokenizer import SmaulTokenizer


class PreferenceModel(nn.Module):
    """Small token-preference model used only for automatic ranking."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_score = nn.Embedding(vocab_size, 1)
        nn.init.zeros_(self.token_score.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.token_score(tokens).squeeze(-1)
        mask = tokens.ne(0)
        return scores.masked_fill(~mask, 0).sum(-1) / mask.sum(-1).clamp_min(1)


class SmaulRL:

    def __init__(self, model_dir: str, work_dir: str = "./rl", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = SmaulTokenizer.from_file(self.model_dir / "tokenizer.json")
        self.policy_dir = self.work_dir / "policy"
        load_dir = self.policy_dir if self.policy_dir.exists() else self.model_dir
        self.model = RWKVXModel.from_pretrained(load_dir).to(self.device)
        self.model.train()
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.preference_path = self.work_dir / "preferences.jsonl"
        self.reward_path = self.work_dir / "preference_model.pt"
        self.reward_model = PreferenceModel(self.tokenizer.get_vocab_size()).to(self.device)
        self._load_reward_model()
        self.replay = ReplayBuffer(storage=LazyTensorStorage(4096))

    def _load_reward_model(self):
        if self.reward_path.exists():
            self.reward_model.load_state_dict(
                torch.load(self.reward_path, map_location=self.device, weights_only=True)
            )

    def _save_reward_model(self):
        torch.save(self.reward_model.state_dict(), self.reward_path)

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> Tuple[int, float]:
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
            sorted_mask = torch.zeros_like(remove).scatter(0, indices, remove)
            logits = logits.masked_fill(sorted_mask, -float("inf"))

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
        logits, _, state = self.model(
            ids,
            state=None,
            use_cache=True,
            return_logits=True,
        )
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
        count: int = 8,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> List[Dict]:
        candidates = []
        for i in range(count):
            text, tokens, old_logprob = self.generate(
                prompt,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
                seed=random.randrange(2**31),
            )
            candidates.append({
                "id": i,
                "text": text,
                "tokens": tokens,
                "old_logprob": old_logprob,
            })
        return candidates

    def _human_pick(self, candidates: List[Dict]) -> int:
        print("\n" + "=" * 80)
        for i, candidate in enumerate(candidates, 1):
            print(f"\n[{i}]\n{candidate['text']}\n")

        while True:
            raw = input(f"Choose best response [1-{len(candidates)}]: ").strip()
            try:
                choice = int(raw) - 1
                if 0 <= choice < len(candidates):
                    return choice
            except ValueError:
                pass
            print("Invalid choice.")

    def _reward_scores(self, candidates: List[Dict]) -> torch.Tensor:
        if not candidates:
            return torch.empty(0, device=self.device)

        max_len = max(1, max(len(c["tokens"]) for c in candidates))
        ids = torch.zeros(len(candidates), max_len, dtype=torch.long, device=self.device)
        for row, candidate in enumerate(candidates):
            tokens = candidate["tokens"][:max_len]
            if tokens:
                ids[row, :len(tokens)] = torch.tensor(tokens, device=self.device)
        return self.reward_model(ids)

    def _store_group(self, prompt: str, candidates: List[Dict], rewards: torch.Tensor):
        data = TensorDict(
            {
                "rewards": rewards.detach().cpu(),
                "chosen": torch.tensor([int(torch.argmax(rewards).item())]),
            },
            batch_size=[],
        )
        self.replay.add(data)

        record = {
            "prompt": prompt,
            "responses": [c["text"] for c in candidates],
            "chosen": int(torch.argmax(rewards).item()),
        }
        with self.preference_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def train_preference_model(self, epochs: int = 3, lr: float = 1e-3):
        if not self.preference_path.exists():
            return

        records = [
            json.loads(line)
            for line in self.preference_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(records) < 2:
            return

        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=lr)
        self.reward_model.train()
        for _ in range(epochs):
            random.shuffle(records)
            for record in records:
                chosen = record["chosen"]
                responses = record["responses"]
                token_lists = [self._encode(text) for text in responses]
                max_len = max(1, max(map(len, token_lists)))
                batch = torch.zeros(len(token_lists), max_len, dtype=torch.long, device=self.device)
                for i, tokens in enumerate(token_lists):
                    if tokens:
                        batch[i, :len(tokens)] = torch.tensor(tokens, device=self.device)

                scores = self.reward_model(batch)
                positive = scores[chosen].expand_as(scores)
                loss = -F.logsigmoid(positive - scores).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._save_reward_model()

    def _logprob(self, prompt: str, response_tokens: List[int]) -> torch.Tensor:
        prompt_ids = self._encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id if self.bos_id is not None else self.eos_id]

        ids = torch.tensor(
            [prompt_ids + response_tokens],
            dtype=torch.long,
            device=self.device,
        )
        logits, _, _ = self.model(
            ids,
            state=None,
            use_cache=False,
            return_logits=True,
        )
        start = len(prompt_ids) - 1
        token_logits = logits[0, start:start + len(response_tokens)]
        targets = torch.tensor(response_tokens, dtype=torch.long, device=self.device)
        return F.log_softmax(token_logits.float(), -1).gather(1, targets[:, None]).squeeze(1)

    def grpo_step(
        self,
        prompt: str,
        candidates: List[Dict],
        chosen: Optional[int] = None,
        lr: float = 1e-6,
        clip: float = 0.2,
        kl_coef: float = 0.02,
    ) -> float:
        if chosen is not None:
            rewards = torch.full((len(candidates),), -1.0, device=self.device)
            rewards[chosen] = 1.0
        else:
            rewards = self._reward_scores(candidates)

        advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-6)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        losses = []

        for candidate, advantage in zip(candidates, advantages):
            if not candidate["tokens"]:
                continue

            new_logprob = self._logprob(prompt, candidate["tokens"])
            old_mean = candidate["old_logprob"] / max(1, len(candidate["tokens"]))
            ratio = torch.exp(new_logprob.mean() - old_mean)
            unclipped = ratio * advantage.detach()
            clipped = ratio.clamp(1 - clip, 1 + clip) * advantage.detach()
            policy_loss = -torch.minimum(unclipped, clipped)
            kl = (new_logprob.mean() - old_mean).pow(2)
            losses.append(policy_loss + kl_coef * kl)

        if not losses:
            return 0.0

        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()

        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.policy_dir, dtype="fp32", include_upstream=False)
        return float(loss.detach())

    def run(
        self,
        prompts: List[str],
        count: int,
        max_new_tokens: int,
        auto_after: int,
        lr: float,
        temperature: float,
        top_k: int,
        top_p: float,
    ):
        if count < 2:
            raise ValueError("--responses must be at least 2")

        for prompt in prompts:
            candidates = self.candidates(
                prompt,
                count,
                max_new_tokens,
                temperature,
                top_k,
                top_p,
            )
            preference_count = 0
            if self.preference_path.exists():
                preference_count = sum(1 for _ in self.preference_path.open("r", encoding="utf-8"))

            use_auto = preference_count >= auto_after
            if use_auto:
                rewards = self._reward_scores(candidates)
                chosen = int(rewards.argmax().item())
                print(f"[AUTO] selected response {chosen + 1}/{count}")
            else:
                chosen = self._human_pick(candidates)

            rewards = torch.where(
                torch.arange(count, device=self.device) == chosen,
                torch.tensor(1.0, device=self.device),
                torch.tensor(-1.0, device=self.device),
            )
            self._store_group(prompt, candidates, rewards)
            self.train_preference_model()
            loss = self.grpo_step(prompt, candidates, chosen=chosen, lr=lr)
            print(f"[RL] loss={loss:.5f}")


def main():
    parser = argparse.ArgumentParser(description="Human-in-the-loop GRPO for SmaulNative")
    parser.add_argument("--model_dir", default="./SmaulNative")
    parser.add_argument("--work_dir", default="./rl")
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--responses", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--auto_after", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    SmaulRL(args.model_dir, args.work_dir, args.device).run(
        args.prompt,
        args.responses,
        args.max_new_tokens,
        args.auto_after,
        args.lr,
        args.temperature,
        args.top_k,
        args.top_p,
    )


if __name__ == "__main__":
    main()

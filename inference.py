#!/usr/bin/env python3
"""Inference engine for SmaulNative RWKV-X checkpoints."""

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from tokenizers import Tokenizer

from rwkv_x_core import RWKVXModel


class RWKVXInference:
    def __init__(self, model_dir: str = "./SmaulNative", device: str = "auto", dtype: str = "auto"):
        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = RWKVXModel.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        if self.eos_id is None:
            raise ValueError("tokenizer.json is missing <eos>")
        if dtype != "auto":
            target = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
            self.model = self.model.to(target)
        self.model.eval()
        self.last_prompt_tokens = 0

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                repetition_penalty: float, recent: List[int]) -> int:
        logits = logits.float().clone()
        if repetition_penalty != 1.0 and recent:
            ids = torch.tensor(list(dict.fromkeys(recent)), device=logits.device)
            vals = logits[ids]
            logits[ids] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        if temperature <= 0:
            return int(torch.argmax(logits).item())
        logits /= temperature
        if top_k > 0 and top_k < logits.numel():
            threshold = torch.topk(logits, top_k).values[-1]
            logits[logits < threshold] = -float("inf")
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_idx[remove]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.inference_mode()
    def _forward(self, tokens: List[int], state=None):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model(ids, state=state, use_cache=True, return_logits=True)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
                 stop: Optional[List[str]] = None, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        prompt_tokens = self.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [self.bos_id] if self.bos_id is not None else [self.eos_id]
        self.last_prompt_tokens = len(prompt_tokens)
        _, _, state = self._forward(prompt_tokens)
        recent = prompt_tokens[-128:]
        generated: List[int] = []
        stops = stop or []
        text = ""
        for _ in range(max_new_tokens):
            token = self._sample(_, temperature, top_k, top_p, repetition_penalty, recent)
            if token == self.eos_id:
                break
            generated.append(token)
            recent.append(token)
            recent = recent[-128:]
            text = self.decode(generated)
            if any(text.endswith(s) for s in stops):
                break
            _, _, state = self._forward([token], state)
        return text

    def stream(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
               top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
               stop: Optional[List[str]] = None, seed: Optional[int] = None) -> Iterable[str]:
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        prompt_tokens = self.encode(prompt)
        if not prompt_tokens:
            prompt_tokens = [self.bos_id] if self.bos_id is not None else [self.eos_id]
        self.last_prompt_tokens = len(prompt_tokens)
        logits, _, state = self._forward(prompt_tokens)
        recent = prompt_tokens[-128:]
        generated: List[int] = []
        emitted = ""
        stops = stop or []
        for _ in range(max_new_tokens):
            token = self._sample(logits[0, -1], temperature, top_k, top_p, repetition_penalty, recent)
            if token == self.eos_id:
                break
            generated.append(token)
            recent.append(token)
            recent = recent[-128:]
            current = self.decode(generated)
            delta = current[len(emitted):]
            emitted = current
            if delta:
                yield delta
            if any(emitted.endswith(s) for s in stops):
                break
            logits, _, state = self._forward([token], state)

    def chat_prompt(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        parts = []
        if system:
            parts.append(f"System:\n{system}\n")
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            parts.append(f"{role}:\n{msg.get('content', '')}\n")
        parts.append("Assistant:\n")
        return "\n".join(parts)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterable[str]:
        system = kwargs.pop("system", None)
        prompt = self.chat_prompt(messages, system)
        yield from self.stream(prompt, **kwargs)

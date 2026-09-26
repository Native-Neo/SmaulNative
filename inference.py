#!/usr/bin/env python3
"""Inference engine for SmaulLinear FP8 checkpoints."""

import random
import threading
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from smaul_linear import SmaulLinear
from tokenizer import SmaulTokenizer

MODEL_WINDOW = 262144
MAX_PROMPT_TOKENS = 65536
_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}


def _seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass


class _IncrementalDecoder:
    def __init__(self, tokenizer: SmaulTokenizer):
        self.table = tokenizer.id_to_token
        self.case = None

    def push(self, token_id: int) -> str:
        token = self.table.get(int(token_id), "<unk>")
        if token == "<cap>":
            self.case = "cap"
            return ""
        if token == "<upper>":
            self.case = "upper"
            return ""
        if token in {"<pad>", "<bos>", "<eos>"}:
            return ""
        if token.startswith("<unused_"):
            # Match tokenizer.decode(): invalid IDs surface as <unk>.
            self.case = None
            return "<unk>"
        if self.case == "cap":
            token = token[:1].upper() + token[1:]
        elif self.case == "upper":
            token = token.upper()
        self.case = None
        return token


class LinearInference:
    def __init__(self, model_dir: str = "./runs/linear", device: str = "auto", dtype: str = "auto",
                 architecture: Optional[str] = None, embedding_storage: Optional[str] = None):
        # architecture/embedding_storage default to None = auto-detect from the
        # checkpoint. An explicit value is validated and mismatches fail
        # clearly instead of silently misinterpreting weights.
        if architecture is not None and architecture not in ("rawr", "plain"):
            raise ValueError(f"architecture must be rawr/plain, got {architecture!r}")
        if embedding_storage is not None and embedding_storage not in ("ram", "mmap"):
            raise ValueError(f"embedding_storage must be ram/mmap, got {embedding_storage!r}")
        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device: {device}")
        self.device = torch.device(device)
        self.model = SmaulLinear.from_pretrained(
            self.model_dir, architecture=architecture,
            embedding_storage=embedding_storage).to(self.device)
        self.tokenizer = SmaulTokenizer.from_file(self.model_dir / "tokenizer.json")
        # Fail fast on checkpoint/tokenizer mismatch (silent wrong-tokenization).
        cfg_vocab = self.model.cfg.vocab_size
        tok_vocab = self.tokenizer.get_vocab_size()
        if cfg_vocab != tok_vocab:
            raise ValueError(
                f"checkpoint vocab_size ({cfg_vocab}) != tokenizer vocab ({tok_vocab}); "
                f"retrain tokenizer with matching --vocab or fix {self.model_dir}")
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.last_prompt_tokens = 0
        self.truncated_prompt = False
        # Serializes concurrent generate/stream calls sharing this engine
        # (infer_server.py threads): _prepare mutates last_prompt_tokens and
        # _seed_all touches global RNG, so unsynchronized sharing races.
        self._gen_lock = threading.RLock()
        if dtype != "auto":
            if dtype not in {"fp32", "bf16"}:
                raise ValueError(f"unsupported dtype: {dtype}")
            self.model = self.model.to(torch.bfloat16 if dtype == "bf16" else torch.float32)
            # .to(bf16) also casts FP8 per-tile scales (float32 buffers) to
            # bf16, which breaks the native kernel (expects f32) and degrades
            # the fallback. w8 (uint8) is unaffected; restore scales to f32.
            try:
                from fp8_tile import fp8_modules
                for _, m in fp8_modules(self.model):
                    if m.sc.dtype != torch.float32:
                        m.sc.data = m.sc.data.float()
            except ImportError:
                pass
        self.model.eval()

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return list(self.tokenizer.encode(text).ids)

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
        candidate_idx = None
        if top_k > 0 and top_k < logits.numel():
            candidate_idx = torch.topk(logits, top_k).indices
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask[candidate_idx] = False
            logits[mask] = -float("inf")
        if 0.0 < top_p < 1.0:
            if candidate_idx is None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                remove = torch.cumsum(probs, dim=-1) > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[sorted_idx[remove]] = -float("inf")
            else:
                candidate_logits = logits[candidate_idx]
                sorted_logits, order = torch.sort(candidate_logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                remove = torch.cumsum(probs, dim=-1) > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                logits[candidate_idx[order[remove]]] = -float("inf")
        if not bool(torch.isfinite(logits).any()):
            # Aggressive top_k/top_p (or all-masked logits) left nothing to
            # sample; end gracefully instead of multinomial RuntimeError.
            if self.eos_id is not None:
                return self.eos_id
            return int(torch.argmax(logits.nan_to_num(0.0)).item())
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    @torch.inference_mode()
    def _forward(self, tokens: List[int]):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        # Last-token-only: the sampler reads logits[0, -1], so the full
        # [1, T, V] head projection (e.g. ~64GiB at 256K x 64K vocab) is
        # never materialized. Returned shape is [1, 1, V].
        logits, _ = self.model(ids, last_only=True)
        return logits, None

    def _prepare(self, prompt: str):
        tokens = self.encode(prompt)
        if not tokens:
            tokens = [self.bos_id] if self.bos_id is not None else [self.eos_id]
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise ValueError(f"prompt too long: {len(tokens)} tokens (max {MAX_PROMPT_TOKENS})")
        self.last_prompt_tokens = len(tokens)
        self.truncated_prompt = len(tokens) > MODEL_WINDOW
        if self.truncated_prompt:
            warnings.warn(
                f"prompt truncated to last {MODEL_WINDOW} tokens "
                f"({len(tokens)} provided); early context is ignored",
                RuntimeWarning, stacklevel=3)
        return tokens

    def _validate(self, max_new_tokens: int, temperature: float, top_k: int, top_p: float, repetition_penalty: float):
        if not 1 <= max_new_tokens <= 65536:
            raise ValueError(f"max_new_tokens must be in [1, 65536], got {max_new_tokens}")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
                 top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
                 stop: Optional[List[str]] = None, seed: Optional[int] = None) -> str:
        return "".join(self.stream(prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop, seed))

    def stream(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
               top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
               stop: Optional[List[str]] = None, seed: Optional[int] = None) -> Iterable[str]:
        # Held across yields: concurrent requests serialize instead of racing
        # on _prepare state and global RNG. Callers must exhaust/close the
        # iterator so the lock is released.
        lock = getattr(self, "_gen_lock", None)
        if lock is None:
            # Tolerate __new__-constructed test doubles without __init__.
            lock = self._gen_lock = threading.RLock()
        with lock:
            yield from self._stream_locked(prompt, max_new_tokens, temperature, top_k, top_p,
                                           repetition_penalty, stop, seed)

    def _stream_locked(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7,
               top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.05,
               stop: Optional[List[str]] = None, seed: Optional[int] = None) -> Iterable[str]:
        self._validate(max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        if seed is not None:
            _seed_all(seed)
        ids = self._prepare(prompt)
        logits, _ = self._forward(ids[-MODEL_WINDOW:])
        recent = ids[-128:]
        stops = [s for s in (stop or []) if s]
        decoder = _IncrementalDecoder(self.tokenizer)
        pending = ""
        max_stop_len = max((len(s) for s in stops), default=0)
        for _ in range(max_new_tokens):
            token = self._sample(logits[0, -1], temperature, top_k, top_p, repetition_penalty, recent)
            if token == self.eos_id:
                break
            recent = (recent + [token])[-128:]
            ids.append(token)
            # Sliding window: model only sees the last MODEL_WINDOW tokens.
            # Keep ids bounded so long generations do not grow RAM/CPU linearly.
            if len(ids) > MODEL_WINDOW + max_new_tokens:
                del ids[:len(ids) - (MODEL_WINDOW + max_new_tokens)]
            pending += decoder.push(token)
            stop_pos = min((pending.find(s) for s in stops if pending.find(s) >= 0), default=-1)
            if stop_pos >= 0:
                if stop_pos:
                    yield pending[:stop_pos]
                return
            if max_stop_len:
                safe_len = max(0, len(pending) - max_stop_len + 1)
                if safe_len:
                    yield pending[:safe_len]
                    pending = pending[safe_len:]
            elif pending:
                yield pending
                pending = ""
            logits, _ = self._forward(ids[-MODEL_WINDOW:])
        if pending:
            yield pending

    @staticmethod
    def _sanitize_content(content: str) -> str:
        # Prevent role spoofing like "System:\n..." inside user content from
        # becoming a fake turn when formatted as "Role:\ncontent".
        lines = str(content).splitlines()
        cleaned = []
        for line in lines:
            if line.strip().lower() in ("system:", "assistant:", "user:", "tool:"):
                cleaned.append(line.strip() + " (quoted)")
            else:
                cleaned.append(line)
        return "\n".join(cleaned)

    def chat_prompt(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        parts = [f"System:\n{self._sanitize_content(system)}\n"] if system else []
        for msg in messages:
            role = str(msg.get("role", "user")).lower()
            if role not in _ALLOWED_ROLES:
                raise ValueError(f"invalid role {msg.get('role')!r}; expected one of {sorted(_ALLOWED_ROLES)}")
            if role == "system":
                raise ValueError("pass system instructions via `system=`, not messages with role='system'")
            content = self._sanitize_content(msg.get("content", ""))
            parts.append(f"{role.capitalize()}:\n{content}\n")
        parts.append("Assistant:\n")
        return "\n".join(parts)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterable[str]:
        system = kwargs.pop("system", None)
        yield from self.stream(self.chat_prompt(messages, system), **kwargs)

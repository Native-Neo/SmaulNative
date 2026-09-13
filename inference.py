#!/usr/bin/env python3
"""Inference engine for SmaulNative RWKV-X checkpoints."""

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from rwkv_x_core import RWKVXConfig, RWKVXModel
from tokenizer import SmaulTokenizer


class _TokenGroupNorm(nn.Module):
    """GroupNorm equivalent for token-wise [batch, channels] tensors."""

    def __init__(self, source: nn.GroupNorm):
        super().__init__()
        self.num_groups = source.num_groups
        self.num_channels = source.num_channels
        self.eps = source.eps
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        channels_per_group = self.num_channels // self.num_groups
        y = x.reshape(-1, self.num_groups, channels_per_group)
        mean = y.mean(dim=-1, keepdim=True)
        var = (y - mean).square().mean(dim=-1, keepdim=True)
        y = (y - mean) * torch.rsqrt(var + self.eps)
        y = y.reshape(shape)
        return y * self.weight + self.bias


def _patch_degenerate_groupnorm(module: nn.Module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, _TokenGroupNorm(child))
        else:
            _patch_degenerate_groupnorm(child)


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
        if self.case == "cap":
            token = token[:1].upper() + token[1:]
        elif self.case == "upper":
            token = token.upper()
        self.case = None
        return token


def _gguf_field(reader, name, default=None):
    field = reader.fields.get(name)
    return default if field is None else field.contents()


def _config_from_gguf(reader) -> RWKVXConfig:
    embedding = int(_gguf_field(reader, "embedding_length"))
    vocab = int(_gguf_field(reader, "vocab_size"))
    layers = int(_gguf_field(reader, "block_count"))
    heads = int(_gguf_field(reader, "attention.head_count"))
    head_size = int(_gguf_field(reader, "rwkv_x.head_size"))
    if heads * head_size != embedding:
        raise ValueError("GGUF attention dimensions do not match embedding_length")
    return RWKVXConfig(
        vocab_size=vocab,
        n_embd=embedding,
        n_layer=layers,
        head_size=head_size,
        n_moba_layer=int(_gguf_field(reader, "rwkv_x.n_moba_layer", 0)),
        moba_chunk_size=int(_gguf_field(reader, "rwkv_x.moba_chunk_size", 512)),
        moba_topk=int(_gguf_field(reader, "rwkv_x.moba_topk", 4)),
        ctx_len_hint=int(_gguf_field(reader, "context_length", 2048)),
        wkv_chunk_size=int(_gguf_field(reader, "rwkv_x.wkv_chunk_size", 64)),
        head_size_divisor=int(_gguf_field(reader, "rwkv_x.head_size_divisor", 8)),
        is_moe=bool(_gguf_field(reader, "rwkv_x.is_moe", False)),
        num_experts=int(_gguf_field(reader, "rwkv_x.num_experts", 1)),
        num_experts_per_tok=int(_gguf_field(reader, "rwkv_x.num_experts_per_tok", 1)),
    )


def _tokenizer_from_gguf(reader) -> SmaulTokenizer:
    tokens = _gguf_field(reader, "tokenizer.ggml.tokens")
    if not tokens:
        raise ValueError("GGUF does not contain tokenizer.ggml.tokens")
    vocab = {str(token): i for i, token in enumerate(tokens)}
    for token in ("<pad>", "<unk>", "<bos>", "<eos>"):
        if token not in vocab:
            raise ValueError(f"GGUF tokenizer is missing {token}")
    data = {
        "version": 5,
        "vocab": vocab,
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": ["<cap>", "<upper>"],
        "case_stats": {},
        "unk_id": vocab["<unk>"],
    }
    return SmaulTokenizer(data)


def _load_gguf(path: Path, device: torch.device):
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("GGUF inference requires the 'gguf' Python package") from exc

    reader = gguf.GGUFReader(str(path))
    cfg = _config_from_gguf(reader)
    state = {}
    for tensor in reader.tensors:
        if tensor.tensor_type.name not in {"F32", "F16", "F64"}:
            raise ValueError(f"unsupported GGUF tensor type for {tensor.name}: {tensor.tensor_type.name}")
        state[tensor.name] = torch.from_numpy(tensor.data.copy())

    model = RWKVXModel(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"GGUF tensor mismatch: missing={missing}, unexpected={unexpected}")
    tokenizer = _tokenizer_from_gguf(reader)
    return model.to(device), tokenizer


class RWKVXInference:
    def __init__(self, model_dir: str = "./SmaulNative", device: str = "auto", dtype: str = "auto"):
        self.model_dir = Path(model_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is unavailable")
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported device: {device}")
        self.device = torch.device(device)

        if self.model_dir.is_file() and self.model_dir.suffix.lower() == ".gguf":
            self.model, self.tokenizer = _load_gguf(self.model_dir, self.device)
        else:
            ggufs = sorted(self.model_dir.glob("*.gguf")) if self.model_dir.is_dir() else []
            if ggufs:
                self.model, self.tokenizer = _load_gguf(ggufs[0], self.device)
            else:
                self.model = RWKVXModel.from_pretrained(self.model_dir).to(self.device)
                self.tokenizer = SmaulTokenizer.from_file(self.model_dir / "tokenizer.json")

        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.last_prompt_tokens = 0
        if dtype != "auto":
            if dtype not in {"fp32", "fp16", "bf16"}:
                raise ValueError(f"unsupported dtype: {dtype}")
            if self.device.type == "cpu" and dtype == "fp16":
                dtype = "fp32"
            self.model = self.model.to({"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype])
        _patch_degenerate_groupnorm(self.model)
        self.model.eval()

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
        return int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())

    @torch.inference_mode()
    def _forward(self, tokens: List[int], state=None):
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return self.model(ids, state=state, use_cache=True, return_logits=True)

    def _prepare(self, prompt: str):
        tokens = self.encode(prompt)
        if not tokens:
            tokens = [self.bos_id] if self.bos_id is not None else [self.eos_id]
        self.last_prompt_tokens = len(tokens)
        return tokens

    def _validate_generation_args(self, max_new_tokens: int, temperature: float, top_k: int,
                                  top_p: float, repetition_penalty: float):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
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
        self._validate_generation_args(max_new_tokens, temperature, top_k, top_p, repetition_penalty)
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        prompt_tokens = self._prepare(prompt)
        logits, _, state = self._forward(prompt_tokens)
        recent = prompt_tokens[-128:]
        stops = [s for s in (stop or []) if s]
        decoder = _IncrementalDecoder(self.tokenizer)
        pending = ""
        max_stop_len = max((len(s) for s in stops), default=0)
        for _ in range(max_new_tokens):
            token = self._sample(logits[0, -1], temperature, top_k, top_p, repetition_penalty, recent)
            if token == self.eos_id:
                break
            recent = (recent + [token])[-128:]
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
            logits, _, state = self._forward([token], state)
        if pending:
            yield pending

    def chat_prompt(self, messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
        parts = [f"System:\n{system}\n"] if system else []
        for msg in messages:
            parts.append(f"{msg.get('role', 'user').capitalize()}:\n{msg.get('content', '')}\n")
        parts.append("Assistant:\n")
        return "\n".join(parts)

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs) -> Iterable[str]:
        system = kwargs.pop("system", None)
        yield from self.stream(self.chat_prompt(messages, system), **kwargs)

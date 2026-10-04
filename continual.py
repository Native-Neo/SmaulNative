"""Continual (streaming) training support for the byte-level RAWR-MoE model.

"Infinite learning" here means training continues indefinitely over incoming
stream chunks with bounded memory -- not infinite memory and not solved
forgetting. Three mechanisms reduce forgetting measurably:

- replay/interleaving of previously seen byte chunks (bounded reservoir),
- a comparatively stable shared trunk (smaller LR) with faster experts/router,
- retention reports (old-domain loss before/after new-domain training).

The per-group LRs need no optimizer change: the training loop keeps one
optimizer per group (trunk/expert/router, each with its own momentum state)
and isolates each group's FP8 ``_gw`` grads while it steps (see ``GwStash``).
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch

from dataset import IGNORE_INDEX, PretrainStream, discover_files

GROUP_TRUNK = "trunk"
GROUP_EXPERT = "expert"
GROUP_ROUTER = "router"
GROUPS = (GROUP_TRUNK, GROUP_EXPERT, GROUP_ROUTER)


class ReplayBuffer:
    """Bounded reservoir of past byte chunks for interleaving.

    Stores raw id lists (each a ``ctx+1`` training chunk); memory is bounded
    by ``capacity`` chunks regardless of stream length. Uniform reservoir
    sampling keeps old domains exercising their experts instead of being
    fully replaced by new data.
    """

    def __init__(self, capacity: int = 512, seed: int = 0):
        if capacity < 0:
            raise ValueError(f"replay capacity must be non-negative, got {capacity}")
        self.capacity = int(capacity)
        self._rng = random.Random(seed)
        self._chunks: List[List[int]] = []
        self._seen = 0

    def __len__(self):
        return len(self._chunks)

    def add(self, ids: List[int]) -> None:
        if self.capacity == 0:
            return
        self._seen += 1
        if len(self._chunks) < self.capacity:
            self._chunks.append(list(ids))
        else:
            j = self._rng.randrange(self._seen)
            if j < self.capacity:
                self._chunks[j] = list(ids)

    def sample(self, n: int) -> List[List[int]]:
        if n <= 0 or not self._chunks:
            return []
        n = min(n, len(self._chunks))
        return [list(c) for c in self._rng.sample(self._chunks, n)]


class ContinualStream:
    """Yield ``(x, y, pos)`` byte pairs across domains without resetting state.

    Each domain is a dataset directory streamed with ``PretrainStream``. The
    carry-over id buffer threads through domains (no auto-reset on domain
    change, so recurrent context and the byte stream stay continuous); pass
    ``reset_state=True`` to clear it explicitly between domains. New chunks
    are added to ``replay`` and, when ``replay_rate > 0``, interleaved back
    into the yielded sequence so old data is never fully replaced.
    """

    def __init__(self, domains: List[Path], tokenizer, ctx_len: int,
                 replay: Optional[ReplayBuffer] = None,
                 replay_rate: float = 0.0,
                 reset_state: bool = False,
                 seed: int = 0):
        if not domains:
            raise ValueError("continual stream needs at least one domain")
        if ctx_len < 1:
            raise ValueError("ctx_len must be positive")
        if not 0.0 <= replay_rate <= 1.0 or not math.isfinite(replay_rate):
            raise ValueError(f"replay_rate must be in [0, 1], got {replay_rate!r}")
        self.domains = [Path(d) for d in domains]
        self.tokenizer = tokenizer
        self.ctx_len = int(ctx_len)
        self.replay = replay if replay is not None else ReplayBuffer(0)
        self.replay_rate = float(replay_rate)
        self.reset_state = bool(reset_state)
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, tuple]]:
        carry: List[int] = []
        for domain in self.domains:
            files = discover_files(domain)
            if not files:
                raise RuntimeError(f"No supported files found under {domain}")
            stream = PretrainStream(domain, self.tokenizer, self.ctx_len,
                                    buffer_tokens=carry)
            for x, y, pos in stream:
                ids = x.tolist() + [int(y[-1].item())]
                self.replay.add(ids)
                yield x, y, (str(domain),) + tuple(pos[1:] if len(pos) > 1 else pos)
                if self.replay_rate > 0 and len(self.replay) \
                        and self._rng.random() < self.replay_rate:
                    for chunk in self.replay.sample(1):
                        rx = torch.tensor(chunk[:-1], dtype=torch.long)
                        ry = torch.tensor(chunk[1:], dtype=torch.long)
                        yield rx, ry, (f"replay:{domain}", -1)
            carry = list(stream.buffer_tokens)
            if self.reset_state:
                carry = []


@torch.no_grad()
def evaluate_loss(model, data_dir: Path, tokenizer, ctx_len: int,
                  max_batches: int = 20) -> float:
    """Mean cross-entropy over up to ``max_batches`` batches from ``data_dir``."""
    if max_batches < 1:
        raise ValueError(f"max_batches must be positive, got {max_batches}")
    model.eval()
    stream = PretrainStream(Path(data_dir), tokenizer, ctx_len)
    total, n = 0.0, 0
    for x, y, _ in stream:
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        total += float(loss.item())
        n += 1
        if n >= max_batches:
            break
    model.train()
    if n == 0:
        raise RuntimeError(f"no evaluation batches found under {data_dir}")
    return total / n


def retention_report(old_before: float, old_after: float, new_after: float) -> dict:
    """Summarize a continual step: old-domain change and new-domain fit."""
    return {
        "old_loss_before": float(old_before),
        "old_loss_after": float(old_after),
        "old_loss_delta": float(old_after - old_before),
        "new_loss_after": float(new_after),
        # Positive delta means forgetting; negative means backward transfer.
        "forgetting": float(old_after - old_before),
    }


def partition_model(model) -> Dict[str, dict]:
    """Split parameters and FP8 modules into trunk/expert/router groups.

    The shared RAWR/recurrent trunk (embeddings, norms, linear attention,
    dense FFN, head) stays comparatively stable; MoE experts adapt faster;
    the router has its own rate. Names are derived from the built model, so
    this cannot drift from the architecture.
    """
    from model import SwiFFN_MoE

    trunk_p, expert_p, router_p = [], [], []
    expert_names, router_names = set(), set()
    for bi, block in enumerate(model.blocks):
        ffn = block.ffn
        if isinstance(ffn, SwiFFN_MoE):
            for p in ffn.gate.parameters():
                router_p.append(p)
            router_names.add(f"blocks.{bi}.ffn.gate")
            for expert in ffn.experts:
                for p in expert.parameters():
                    expert_p.append(p)
            expert_names.add(f"blocks.{bi}.ffn.experts")
    trunk_ids = {id(p) for p in expert_p} | {id(p) for p in router_p}
    for p in model.parameters():
        if id(p) not in trunk_ids:
            trunk_p.append(p)

    from kernel.fp8_tile import fp8_modules
    trunk_m, expert_m, router_m = [], [], []
    for name, mod in fp8_modules(model):
        if ".ffn.experts." in name:
            expert_m.append((name, mod))
        elif ".ffn.gate" in name and any(name.startswith(r) for r in router_names):
            router_m.append((name, mod))
        else:
            trunk_m.append((name, mod))
    return {
        GROUP_TRUNK: {"params": trunk_p, "fp8": trunk_m},
        GROUP_EXPERT: {"params": expert_p, "fp8": expert_m},
        GROUP_ROUTER: {"params": router_p, "fp8": router_m},
    }


class GwStash:
    """Isolate one group's FP8 ``_gw`` grads while its optimizer steps.

    The house optimizers discover FP8 modules via ``fp8_modules(model)`` and
    step every module carrying ``_gw``. Hiding the other groups' grads lets
    each group optimizer update only its own modules with its own LR and
    momentum, without any optimizer code change. Restores everything on exit
    (``step()`` clears the active group's ``_gw`` to None, as usual).
    """

    def __init__(self, model, keep: List[tuple]):
        from kernel.fp8_tile import fp8_modules
        self.model = model
        self.keep_ids = {id(m) for _, m in keep}
        self._all = fp8_modules(model)
        self.saved: dict = {}

    def __enter__(self):
        self.saved = {}
        for _, mod in self._all:
            if id(mod) not in self.keep_ids and mod._gw is not None:
                self.saved[id(mod)] = mod._gw
                mod._gw = None
        return self

    def __exit__(self, *exc):
        for _, mod in self._all:
            gw = self.saved.get(id(mod))
            if gw is not None and mod._gw is None:
                mod._gw = gw
        self.saved = {}
        return False


def global_clip_scale(model, clip: float) -> float:
    """Scale all grads to the global norm ``clip``; return the pre-clip norm."""
    from kernel.fp8_tile import fp8_modules
    from model import _grad_norm
    grads = [m._gw for _, m in fp8_modules(model) if m._gw is not None] + \
            [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    t = _grad_norm(grads)
    if not math.isfinite(t):
        return float("inf")
    if t > clip:
        s = clip / (t + 1e-6)
        for _, m in fp8_modules(model):
            if m._gw is not None:
                m._gw.mul_(s)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(s)
    return t


def moe_routing_summary(model) -> dict:
    """Aggregate ``routing_stats()`` over every MoE block for logging."""
    from model import SwiFFN_MoE
    blocks = [m for m in model.modules() if isinstance(m, SwiFFN_MoE)]
    if not blocks:
        return {"moe_blocks": 0}
    counts: List[int] = []
    inactive = 0
    worst = 0.0
    for b in blocks:
        st = b.routing_stats()
        if not counts:
            counts = list(st["token_counts"])
        else:
            counts = [c + n for c, n in zip(counts, st["token_counts"])]
        inactive += st["inactive_experts"]
        worst = max(worst, st["max_usage_frac"])
    total = sum(counts) or 1
    return {
        "moe_blocks": len(blocks),
        "token_counts": counts,
        "usage_frac": [c / total for c in counts],
        "inactive_experts": inactive,
        "max_usage_frac": worst,
    }

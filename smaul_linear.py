import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from compute import get_backend
from fp8_tile import FP8Linear

@dataclass
class LinearConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layer: int = 8
    n_heads: int = 8
    ffn_mult: float = 2.5
    eps: float = 1e-6
    tile: int = 64
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    precision: str = "fp8"
    tokenizer_sha256: str = ""
    dataset_fingerprint: str = ""
    # Architecture selector: "plain" (dense baseline, historical behavior) or
    # "rawr" (sparse structural variant; both use Linear Attention).
    # Code-level default stays "plain" so existing checkpoints/tests that
    # build LinearConfig without this field keep historical behavior; the
    # train/infer CLIs default to "rawr" explicitly.
    architecture: str = "plain"
    # Embedding storage: "ram" (nn.Embedding) or "mmap" (file-backed).
    embedding_storage: str = "ram"
    # Rawr structural sparsity: fraction of hidden/head connections omitted.
    rawr_sparsity: float = 0.5
    rawr_min_degree: int = 4
    rawr_graph_hash: str = ""
    rawr_edge_count: int = 0

    def __post_init__(self):
        if self.precision not in ("fp8", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}; expected 'fp8' or 'fp32'")
        if self.architecture not in ("plain", "rawr"):
            raise ValueError(f"unknown architecture {self.architecture!r}; expected 'plain' or 'rawr'")
        if self.embedding_storage not in ("ram", "mmap"):
            raise ValueError(
                f"unknown embedding_storage {self.embedding_storage!r}; expected 'ram' or 'mmap'")
        for name in ("vocab_size", "d_model", "n_layer", "n_heads", "tile", "num_experts",
                     "num_experts_per_tok"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if not isinstance(self.ffn_mult, (int, float)) or not math.isfinite(self.ffn_mult) \
                or self.ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive finite, got {self.ffn_mult!r}")
        if not isinstance(self.eps, float) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be positive finite, got {self.eps!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.is_moe and self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) > num_experts ({self.num_experts})")
        if self.is_moe and self.architecture == "rawr":
            raise ValueError("MoE upcycling is only supported with architecture='plain'")
        if not isinstance(self.rawr_sparsity, (int, float)) \
                or not 0.0 <= float(self.rawr_sparsity) < 1.0:
            raise ValueError(f"rawr_sparsity must be in [0, 1), got {self.rawr_sparsity!r}")
        if not isinstance(self.rawr_min_degree, int) or self.rawr_min_degree < 1:
            raise ValueError(f"rawr_min_degree must be a positive int, got {self.rawr_min_degree!r}")
        if not isinstance(self.rawr_edge_count, int) or self.rawr_edge_count < 0:
            raise ValueError(f"rawr_edge_count must be non-negative, got {self.rawr_edge_count!r}")

    def save(self, p: Path): Path(p).write_text(json.dumps(asdict(self), indent=2))
    @classmethod
    def load(cls, p: Path):
        try:
            data = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"could not load config {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"config {p} must contain a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid config {p}: {exc}") from exc

def _linear(cfg: LinearConfig, in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    """Precision-selected linear layer: tiled-E4M3 FP8 or plain FP32."""
    if in_f <= 0 or out_f <= 0:
        raise ValueError(f"in_f/out_f must be positive, got {in_f}/{out_f}")
    if cfg.precision == "fp8":
        # FP8Linear supports bias; honor it so fp8/fp32 modes do not diverge.
        return FP8Linear(in_f, out_f, cfg.tile, bias)
    if cfg.precision == "fp32":
        return _DenseLinear(in_f, out_f, bias=bias)
    raise ValueError(f"unknown precision {cfg.precision!r}; expected 'fp8' or 'fp32'")

class _DenseLinear(nn.Module):
    """Plain-FP32 linear with FP8Linear-compatible dtype behavior.

    Computes in float32 and returns the input dtype, so the surrounding model
    code (bf16 activations, fp32 attention core) is identical in both modes.
    """

    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.lin(x.float()).to(x.dtype if x.is_floating_point() else torch.float32)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"RMSNorm eps must be positive finite, got {eps!r}")
        self.weight = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(v + self.eps) * self.weight).to(x.dtype if x.is_floating_point() else torch.float32)

class LinearAttention(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d, self.nh = cfg.d_model, cfg.n_heads
        if d % self.nh != 0:
            raise ValueError(f"d_model ({d}) must be divisible by n_heads ({self.nh})")
        self.hd = d // self.nh
        self.eps = cfg.eps
        self.q = _linear(cfg, d, d)
        self.k = _linear(cfg, d, d)
        self.v = _linear(cfg, d, d)
        self.o = _linear(cfg, d, d)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.nh, self.hd
        q = self.q(x).float().view(B, T, H, D)
        k = self.k(x).float().view(B, T, H, D)
        v = self.v(x).float().view(B, T, H, D)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        y = _LinearAttnFn.apply(q, k, v, self.eps).reshape(B, T, -1)
        return self.o(y.to(x.dtype if x.is_floating_point() else torch.float32))


def _attn_reference(Q, K, V, eps):
    """Pure-torch linear-attention recurrence (exact math reference).

    Same formulas as the native kernel: elu+1 feature map applied by the
    caller, per-step key normalization, FP32 S/z state, causal steps, no
    softmax, no QK^T.
    """
    B, T, H, D = Q.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float32, device=Q.device)
    z = torch.zeros(B, H, D, dtype=torch.float32, device=Q.device)
    ys = []
    for t in range(T):
        kt, vt, qt = K[:, t], V[:, t], Q[:, t]
        kt = kt / kt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        S = S + kt.unsqueeze(-1) @ vt.unsqueeze(-2)
        z = z + kt
        num = (qt.unsqueeze(-2) @ S).squeeze(-2)
        den = (qt * z).sum(-1, keepdim=True).clamp_min(eps)
        ys.append(num / den)
    return torch.stack(ys, 1)


def _attn_reference_backward(dY, Q, K, V, eps):
    """Gradients via autograd through the reference recurrence."""
    wants = (Q.requires_grad, K.requires_grad, V.requires_grad)
    Qr = Q.detach().requires_grad_(wants[0])
    Kr = K.detach().requires_grad_(wants[1])
    Vr = V.detach().requires_grad_(wants[2])
    with torch.enable_grad():
        Yr = _attn_reference(Qr, Kr, Vr, eps)
        outs = [t for t, w in zip((Qr, Kr, Vr), wants) if w]
        grads = torch.autograd.grad(Yr, outs, dY, allow_unused=True) if outs else []
    it = iter(grads)
    return tuple(next(it) if w else None for w in wants)


class _LinearAttnFn(torch.autograd.Function):
    """Linear-attention op: native AVX1 kernel with reference fallback.

    Forward runs the backend kernel (native when available). Backward runs the
    backend two-pass kernel, or autograd through the reference recurrence when
    native is unavailable. Q/K/V must already carry the elu+1 feature map; key
    normalization happens inside the op. Tensors are saved for backward only
    when grads are needed.
    """

    @staticmethod
    def forward(ctx, Q, K, V, eps):
        be = get_backend()
        need = Q.requires_grad or K.requires_grad or V.requires_grad
        Y, DEN = be.attn_forward(Q, K, V, eps, need)
        if need:
            ctx.save_for_backward(Q, K, V, Y, DEN) if DEN is not None else ctx.save_for_backward(Q, K, V, Y)
            ctx.has_den = DEN is not None
            ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dY):
        saved = ctx.saved_tensors
        Q, K, V, Y = saved[0], saved[1], saved[2], saved[3]
        DEN = saved[4] if ctx.has_den else None
        be = get_backend()
        dQ, dK, dV = be.attn_backward(dY, Q, K, V, Y, DEN, ctx.eps)
        return (dQ if Q.requires_grad else None,
                dK if K.requires_grad else None,
                dV if V.requires_grad else None, None)

class SwiFFN(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        h = int(cfg.d_model * cfg.ffn_mult)
        self.gate = _linear(cfg, cfg.d_model, h)
        self.up = _linear(cfg, cfg.d_model, h)
        self.down = _linear(cfg, h, cfg.d_model)
    def forward(self, x):
        return self.down((F.silu(self.gate(x).float()) * self.up(x).float()).to(x.dtype if x.is_floating_point() else torch.float32))


class SparseLinear(nn.Module):
    """Fixed fan-in sparse FP32 linear driven by the Rawr graph.

    Only ``values`` (out_f x K) are stored as parameters; column indices come
    from the shared Rawr graph (kept as a non-persistent buffer, rebuilt from
    ``rawr_graph.json`` on load, so per-layer storage is values-only).
    Omitted connections consume neither storage nor FLOPs.
    """

    def __init__(self, in_f: int, out_f: int, cols):
        super().__init__()
        import torch as _torch

        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"in_f/out_f must be positive, got {in_f}/{out_f}")
        cols = _torch.as_tensor(cols, dtype=_torch.long)
        if cols.dim() != 2 or cols.shape[0] != out_f:
            raise ValueError(f"cols must be [out_f, K], got {tuple(cols.shape)}")
        if int(cols.min()) < 0 or int(cols.max()) >= in_f:
            raise ValueError("cols indices out of range")
        self.in_f, self.out_f = in_f, out_f
        self.register_buffer("cols", cols, persistent=False)
        self.values = nn.Parameter(torch.empty(out_f, cols.shape[1], dtype=torch.float32))
        nn.init.kaiming_uniform_(self.values, a=math.sqrt(5))

    # Cap per-block transient (~128MB): x[..., cols[o0:o1]] materializes
    # (rows, b, K), which at long contexts would otherwise blow up RAM.
    _TRANSIENT_BUDGET = 134217728

    def forward(self, x):
        rows = x.shape[:-1].numel()
        k = self.values.shape[1]
        b = max(1, min(self.out_f, self._TRANSIENT_BUDGET // max(1, rows * k * 8)))
        if b >= self.out_f:
            gathered = x[..., self.cols]
            out = (gathered.float() * self.values).sum(-1)
        else:
            outs = []
            for o0 in range(0, self.out_f, b):
                o1 = min(self.out_f, o0 + b)
                outs.append((x[..., self.cols[o0:o1]].float()
                             * self.values[o0:o1]).sum(-1))
            out = torch.cat(outs, -1)
        return out.to(x.dtype if x.is_floating_point() else torch.float32)


class RawrFFN(nn.Module):
    """SwiGLU FFN with graph-sparse projections (Rawr architecture)."""

    def __init__(self, cfg: LinearConfig, graph):
        super().__init__()
        if graph is None:
            raise ValueError("RawrFFN needs a Rawr graph (pass rawr_graph=)")
        from rawr_graph import hidden_cols

        h = int(cfg.d_model * cfg.ffn_mult)
        d = cfg.d_model
        mp = max(1, min(cfg.rawr_min_degree, d))
        mp_h = max(1, min(cfg.rawr_min_degree, h))
        self.gate = SparseLinear(d, h, hidden_cols(h, d, graph, cfg.rawr_sparsity, mp))
        self.up = SparseLinear(d, h, hidden_cols(h, d, graph, cfg.rawr_sparsity, mp))
        self.down = SparseLinear(h, d, hidden_cols(d, h, graph, cfg.rawr_sparsity, mp_h))

    def forward(self, x):
        return self.down(
            (F.silu(self.gate(x).float()) * self.up(x).float()).to(
                x.dtype if x.is_floating_point() else torch.float32))

class SwiFFN_MoE(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        if cfg.num_experts < 1 or cfg.num_experts_per_tok < 1:
            raise ValueError("num_experts and num_experts_per_tok must be >= 1")
        if cfg.num_experts_per_tok > cfg.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({cfg.num_experts_per_tok}) > num_experts ({cfg.num_experts})")
        # No silent clamp: misconfig must fail fast instead of training a
        # different MoE than requested.
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.experts = nn.ModuleList([SwiFFN(cfg) for _ in range(cfg.num_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

    def balance_loss(self, prob: torch.Tensor) -> torch.Tensor:
        # Switch-Transformer style load-balancing aux loss. Top-k routing is
        # non-differentiable, so without this the gate collapses to 1-2 experts.
        # Callers add `1e-2 * moe.balance_loss(prob)` during training.
        density = prob.mean(0)
        return (density * density * self.num_experts).sum()

    def forward(self, x):
        # NOTE: top-k routing is non-differentiable; the gate only learns via
        # straight-through on topv weights. Monitor expert usage and add
        # balance_loss() during training to prevent collapse.
        prob = torch.softmax(self.gate(x.float()), -1)
        topv, topi = torch.topk(prob, self.top_k, -1)
        denom = topv.sum(-1, keepdim=True)
        # Guard tiny denominators (would explode weights); fall back to uniform.
        tiny = denom.squeeze(-1) < 1e-6
        topv = torch.where(tiny.unsqueeze(-1), torch.full_like(topv, 1.0 / self.top_k),
                           topv / denom.clamp_min(1e-9))
        out = torch.zeros_like(x.float())
        for e, expert in enumerate(self.experts):
            w = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            m = (w.squeeze(-1) > 0)
            if m.any():
                xm = x[m].contiguous()
                out[m] += expert(xm).float() * w[m]
        return out.to(x.dtype if x.is_floating_point() else torch.float32)

class Block(nn.Module):
    def __init__(self, cfg, rawr_graph=None):
        super().__init__()
        d = cfg.d_model
        self.n1 = RMSNorm(d, cfg.eps)
        self.att = LinearAttention(cfg)
        self.n2 = RMSNorm(d, cfg.eps)
        self.n3 = RMSNorm(d, cfg.eps)
        if cfg.is_moe:
            self.ffn = SwiFFN_MoE(cfg)
        elif cfg.architecture == "rawr":
            self.ffn = RawrFFN(cfg, rawr_graph)
        else:
            self.ffn = SwiFFN(cfg)
        self.n4 = RMSNorm(d, cfg.eps)
        self.n5 = RMSNorm(d, cfg.eps)
    def forward(self, x):
        ck = self.training and torch.is_grad_enabled()
        n1x = self.n1(x)
        ax = torch.utils.checkpoint.checkpoint(self.att, n1x, use_reentrant=False) if ck else self.att(n1x)
        a = self.n2(ax.float())
        x = self.n3((x.float() + a.float()).to(x.dtype))
        fx = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False) if ck else self.ffn(x)
        f = self.n4(fx.float())
        return self.n5((x.float() + f.float()).to(x.dtype))

class SmaulLinear(nn.Module):
    def __init__(self, cfg, rawr_graph=None, emb_path=None):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = LinearConfig(**cfg)
        self.cfg = cfg
        if cfg.architecture == "rawr" and rawr_graph is None:
            from rawr_graph import fallback_graph

            rawr_graph = fallback_graph(cfg.vocab_size, cfg.rawr_min_degree)
        if rawr_graph is not None and rawr_graph.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"Rawr graph vocab {rawr_graph.vocab_size} != config vocab {cfg.vocab_size}")
        self.rawr_graph = rawr_graph
        from embeddings import create_embedding

        # Common embedding interface: RAM (nn.Embedding behavior) or mmap
        # (file-backed, OS-paged). FP32 table in both cases so Lion steps
        # do not stagnate; cast to bf16 on forward.
        self.emb = create_embedding(cfg.vocab_size, cfg.d_model,
                                    cfg.embedding_storage, emb_path)
        self.n0 = RMSNorm(cfg.d_model, cfg.eps)
        self.blocks = nn.ModuleList([Block(cfg, rawr_graph) for _ in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model, cfg.eps)
        if cfg.architecture == "rawr":
            from rawr_graph import hidden_cols

            mp = max(1, min(cfg.rawr_min_degree, cfg.d_model))
            self.head = SparseLinear(cfg.d_model, cfg.vocab_size,
                                     hidden_cols(cfg.vocab_size, cfg.d_model, rawr_graph,
                                                 cfg.rawr_sparsity, mp))
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.head.weight, 0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, labels=None, last_only=False):
        """Full-sequence forward (training) or last-token-only (inference).

        ``last_only=True`` runs the identical trunk (embedding, norms, Linear
        Attention blocks) over the whole context but projects only the final
        position through the LM head, returning ``[B, 1, V]`` instead of
        ``[B, T, V]``. Head math is position-independent (dense ``nn.Linear``
        or ``SparseLinear`` row gather), so the returned row matches the last
        row of the full computation. Training (``labels=...``) always uses
        the full path; combining it with ``last_only`` is rejected.
        """
        if last_only and labels is not None:
            raise ValueError("last_only inference cannot compute a loss (labels given)")
        x = self.emb(idx).to(torch.bfloat16)
        x = self.n0(x.float()).to(torch.bfloat16)
        for b in self.blocks:
            x = b(x)
        x = self.nf(x.float())
        if last_only:
            # Keep the dim so downstream [0, -1] indexing is unchanged.
            x = x[:, -1:, :]
        logits = self.head(x.float())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return logits, loss

    def save_pretrained(self, out: Path):
        from safetensors.torch import save_file
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.architecture == "rawr" and self.rawr_graph is not None:
            self.cfg.rawr_graph_hash = self.rawr_graph.digest
            self.cfg.rawr_edge_count = len(self.rawr_graph.edges)
            from rawr_graph import save_graph

            save_graph(self.rawr_graph, out / "rawr_graph.json")
        if self.cfg.embedding_storage == "mmap":
            from embeddings import EMBEDDING_FILE

            emb = self.emb
            dest = out / EMBEDDING_FILE
            if getattr(emb, "path", None) is not None and Path(emb.path) == dest:
                emb.flush()
            else:
                # Chunked file-to-file copy; never holds two full tables.
                import numpy as np

                src = np.memmap(str(emb.path), dtype=np.float32, mode="r",
                                shape=(self.cfg.vocab_size, self.cfg.d_model)) \
                    if hasattr(emb, "path") else emb.weight.detach().cpu().float().numpy()
                import numpy as _np

                tmp_e = dest.with_suffix(".dat.tmp")
                dst = _np.memmap(str(tmp_e), dtype=_np.float32, mode="w+",
                                 shape=(self.cfg.vocab_size, self.cfg.d_model))
                for r0 in range(0, self.cfg.vocab_size, 1024):
                    r1 = min(self.cfg.vocab_size, r0 + 1024)
                    dst[r0:r1] = src[r0:r1]
                dst.flush()
                del dst
                try:
                    del src
                except Exception:
                    pass
                os.replace(tmp_e, dest)
            sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
                  if not k.startswith("emb.")}
        else:
            sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        tmp = out / "model.safetensors.tmp"
        save_file(sd, str(tmp))
        os.replace(tmp, out / "model.safetensors")
        self.cfg.save(out / "config.json")

    @classmethod
    def from_pretrained(cls, d: Path, device: str = "cpu", architecture=None,
                        embedding_storage=None):
        from safetensors.torch import load_file
        d = Path(d)
        cfg = LinearConfig.load(d / "config.json")
        if architecture is not None and architecture != cfg.architecture:
            raise ValueError(
                f"checkpoint architecture is {cfg.architecture!r}, "
                f"but {architecture!r} was requested; refusing to misinterpret "
                f"(rawr <-> plain weights are not interchangeable)")
        storage = embedding_storage or cfg.embedding_storage
        if storage not in ("ram", "mmap"):
            raise ValueError(f"unknown embedding_storage {storage!r}")
        graph = None
        if cfg.architecture == "rawr":
            from rawr_graph import load_graph

            gp = d / "rawr_graph.json"
            if not gp.exists():
                raise FileNotFoundError(
                    f"Rawr checkpoint {d} is missing rawr_graph.json")
            graph = load_graph(gp)
            if graph.vocab_size != cfg.vocab_size:
                raise ValueError(
                    f"Rawr graph vocab {graph.vocab_size} != config vocab {cfg.vocab_size}")
            if cfg.rawr_graph_hash and graph.digest != cfg.rawr_graph_hash:
                raise ValueError(
                    f"Rawr graph hash {graph.digest} != checkpoint record "
                    f"{cfg.rawr_graph_hash}; refusing to load mismatched graph")
        if storage == "mmap":
            from embeddings import EMBEDDING_FILE

            ep = d / EMBEDDING_FILE
            if not ep.exists():
                raise FileNotFoundError(
                    f"mmap checkpoint {d} is missing {EMBEDDING_FILE}")
            m = cls(cfg, rawr_graph=graph, emb_path=ep)
        else:
            m = cls(cfg, rawr_graph=graph)
        try:
            sd = load_file(str(d / "model.safetensors"), device=device)
        except RuntimeError as exc:
            raise RuntimeError(f"could not load checkpoint {d}: {exc}") from exc
        try:
            if storage == "mmap":
                missing, unexpected = m.load_state_dict(sd, strict=False)
                missing = set(missing)
                if unexpected:
                    raise RuntimeError(f"unexpected keys: {sorted(unexpected)}")
                # emb.weight lives in embeddings.dat, not the safetensors file.
                ok_missing = {k for k in missing if k.startswith("emb.")}
                if missing - ok_missing:
                    raise RuntimeError(
                        f"checkpoint {d} ({cfg.architecture}) missing keys: "
                        f"{sorted(missing - ok_missing)}")
            else:
                m.load_state_dict(sd, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint incompatible with config (arch={cfg.architecture} "
                f"fp8<->fp32 key change w8/sc vs weight?): {exc}"
            ) from exc
        # A rawr checkpoint loaded as plain (or vice versa) can never reach
        # here: key shapes/names differ and strict/explicit checks fail first.
        return m

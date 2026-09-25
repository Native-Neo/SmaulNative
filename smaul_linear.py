import json
import math
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

    def __post_init__(self):
        if self.precision not in ("fp8", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}; expected 'fp8' or 'fp32'")

    def save(self, p: Path): Path(p).write_text(json.dumps(asdict(self), indent=2))
    @classmethod
    def load(cls, p: Path): return cls(**json.loads(Path(p).read_text()))

def _linear(cfg: LinearConfig, in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    """Precision-selected linear layer: tiled-E4M3 FP8 or plain FP32."""
    if cfg.precision == "fp8":
        return FP8Linear(in_f, out_f, cfg.tile)
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
        assert d % self.nh == 0
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

class SwiFFN_MoE(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        self.top_k = min(cfg.num_experts, cfg.num_experts_per_tok)
        if self.top_k < 1:
            raise ValueError("num_experts_per_tok must be >= 1")
        self.experts = nn.ModuleList([SwiFFN(cfg) for _ in range(cfg.num_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
    def forward(self, x):
        prob = torch.softmax(self.gate(x.float()), -1)
        topv, topi = torch.topk(prob, self.top_k, -1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x.float())
        for e, expert in enumerate(self.experts):
            w = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            m = (w.squeeze(-1) > 0)
            if m.any():
                out[m] += expert(x[m]).float() * w[m]
        return out.to(x.dtype if x.is_floating_point() else torch.float32)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.n1 = RMSNorm(d, cfg.eps)
        self.att = LinearAttention(cfg)
        self.n2 = RMSNorm(d, cfg.eps)
        self.n3 = RMSNorm(d, cfg.eps)
        self.ffn = SwiFFN_MoE(cfg) if cfg.is_moe else SwiFFN(cfg)
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
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = LinearConfig(**cfg)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        with torch.no_grad():
            self.emb.weight.data.normal_(0, 0.02)
            self.emb.weight.data = self.emb.weight.data.to(torch.bfloat16)
        self.n0 = RMSNorm(cfg.d_model, cfg.eps)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model, cfg.eps)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.head.weight, 0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, labels=None):
        x = self.emb(idx).to(torch.bfloat16)
        x = self.n0(x.float()).to(torch.bfloat16)
        for b in self.blocks:
            x = b(x)
        x = self.nf(x.float())
        logits = self.head(x.float())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return logits, loss

    def save_pretrained(self, out: Path):
        from safetensors.torch import save_file
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(sd, str(out / "model.safetensors"))
        self.cfg.save(out / "config.json")

    @classmethod
    def from_pretrained(cls, d: Path):
        from safetensors.torch import load_file
        d = Path(d)
        m = cls(LinearConfig.load(d / "config.json"))
        m.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
        return m

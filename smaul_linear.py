import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

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
    tokenizer_sha256: str = ""
    dataset_fingerprint: str = ""

    def save(self, p: Path): Path(p).write_text(json.dumps(asdict(self), indent=2))
    @classmethod
    def load(cls, p: Path): return cls(**json.loads(Path(p).read_text()))

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
        self.q = FP8Linear(d, d, cfg.tile)
        self.k = FP8Linear(d, d, cfg.tile)
        self.v = FP8Linear(d, d, cfg.tile)
        self.o = FP8Linear(d, d, cfg.tile)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.nh, self.hd
        q = self.q(x).float().view(B, T, H, D)
        k = self.k(x).float().view(B, T, H, D)
        v = self.v(x).float().view(B, T, H, D)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k / (k.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        S = torch.zeros(B, H, D, D, dtype=torch.float32, device=x.device)
        z = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        ys = []
        for t in range(T):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]
            S = S + kt.unsqueeze(-1) @ vt.unsqueeze(-2)
            z = z + kt
            num = (qt.unsqueeze(-2) @ S).squeeze(-2)
            den = (qt * z).sum(-1, keepdim=True).clamp_min(self.eps)
            ys.append(num / den)
        y = torch.stack(ys, 1).reshape(B, T, -1)
        return self.o(y.to(x.dtype if x.is_floating_point() else torch.float32))

class SwiFFN(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        h = int(cfg.d_model * cfg.ffn_mult)
        self.gate = FP8Linear(cfg.d_model, h, cfg.tile)
        self.up = FP8Linear(cfg.d_model, h, cfg.tile)
        self.down = FP8Linear(h, cfg.d_model, cfg.tile)
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

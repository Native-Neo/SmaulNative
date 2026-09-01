# rwkv_x_core.py -- CPU-pure-PyTorch RWKV-7 + MOBA hybrid (howard-hou/RWKV-X math, no CUDA kernel).
# Decay: w_eff = exp(-exp(w_raw)) (real kernel) == exp(-0.606531*sigmoid(w0+g)) (used below). Verified equal.

import math
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# Config

@dataclass
class RWKVXConfig:
    vocab_size: int = 65530
    n_embd: int = 832
    n_layer: int = 20
    head_size: int = 64
    n_moba_layer: int = 5
    moba_chunk_size: int = 512
    moba_topk: int = 4
    dropout: float = 0.0
    head_size_divisor: int = 8
    ctx_len_hint: int = 1024
    wkv_chunk_size: int = 64
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text()))

    def approx_param_count(self) -> int:
        C, V, L = self.n_embd, self.vocab_size, self.n_layer
        emb_head = 2 * V * C
        d_decay = max(32, round(1.8 * (C ** 0.5) / 32) * 32)
        d_aaa = d_decay
        d_mv = max(32, round(1.3 * (C ** 0.5) / 32) * 32)
        d_gate = max(32, round(0.6 * (C ** 0.8) / 32) * 32)
        tmix = 4 * C * C + C * (2 * d_decay + 2 * d_aaa + 2 * d_mv + 2 * d_gate)
        cmix = 8 * C * C
        moba_att = 4 * C * C
        rwkv_layers = (L - self.n_moba_layer) * (tmix + cmix)
        moba_layers = self.n_moba_layer * (moba_att + cmix)
        return emb_head + rwkv_layers + moba_layers


def config_for_target_params(target_params: int, vocab_size: int = 65530,
                              n_embd: int = 768, n_moba_layer: int = 3,
                              head_size: int = 64) -> RWKVXConfig:
    if n_embd % head_size != 0:
        raise ValueError(f"n_embd ({n_embd}) must be divisible by head_size ({head_size})")
    best = None
    for n_layer in range(4, 80):
        cfg = RWKVXConfig(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
                           n_moba_layer=min(n_moba_layer, n_layer), head_size=head_size)
        diff = abs(cfg.approx_param_count() - target_params)
        if best is None or diff < best[0]:
            best = (diff, cfg)
    return best[1]


def _wkv_run_chunk(state: torch.Tensor, w_c: torch.Tensor, k_c: torch.Tensor, v_c: torch.Tensor,
                    kk_c: torch.Tensor, a_c: torch.Tensor, r_c: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    Tc = w_c.shape[1]
    ys = []
    for t in range(Tc):
        w_t = w_c[:, t]
        k_t = k_c[:, t]
        v_t = v_c[:, t]
        kk_t = kk_c[:, t]
        a_t = a_c[:, t]
        r_t = r_c[:, t]
        vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
        ab = (-kk_t).unsqueeze(-1) @ (kk_t * a_t).unsqueeze(-2)
        state = state * w_t.unsqueeze(-2) + state @ ab.float() + vk.float()
        y_t = (state.to(dtype=r_t.dtype) @ r_t.unsqueeze(-1)).squeeze(-1)
        ys.append(y_t)
    return state, torch.stack(ys, dim=1)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, cfg: RWKVXConfig, layer_id: int):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.head_size = cfg.head_size
        C = cfg.n_embd
        self.n_head = C // self.head_size
        assert C % self.head_size == 0, "n_embd must be divisible by head_size"
        H, N = self.n_head, self.head_size
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(1, cfg.n_layer - 1)
            ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)
            ddd = torch.arange(C, dtype=torch.float32).view(1, 1, C) / C
            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.x_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            def ortho_init(x, scale):
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                return x
            d_decay = max(32, round(1.8 * (C ** 0.5) / 32) * 32)
            self.w1 = nn.Parameter(torch.zeros(C, d_decay))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(d_decay, C), 0.1))
            decay_speed = torch.tensor([-7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5) for n in range(C)])
            self.w0 = nn.Parameter(decay_speed.reshape(1, 1, C) + 0.5)
            d_aaa = max(32, round(1.8 * (C ** 0.5) / 32) * 32)
            self.a1 = nn.Parameter(torch.zeros(C, d_aaa))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(d_aaa, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))
            d_mv = max(32, round(1.3 * (C ** 0.5) / 32) * 32)
            if layer_id != 0:
                self.v1 = nn.Parameter(torch.zeros(C, d_mv))
                self.v2 = nn.Parameter(ortho_init(torch.zeros(d_mv, C), 0.1))
                self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)
            d_gate = max(32, round(0.6 * (C ** 0.8) / 32) * 32)
            self.g1 = nn.Parameter(torch.zeros(C, d_gate))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(d_gate, C), 0.1))
            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5) * (cfg.head_size_divisor ** 2))
            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.output.weight.data.zero_()

    def forward(self, x: torch.Tensor, v_first: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size
        if state is not None and isinstance(state, tuple):
            s_tensor, x_prev_last = state
            prev0 = x_prev_last.unsqueeze(1)
            state = s_tensor
        else:
            prev0 = torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x_prev = torch.cat([prev0, x[:, :-1, :]], dim=1)
        xx = x_prev - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
        g_ = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + g_).float()))
        r_ = r.view(B, T, H, N)
        w_ = w.view(B, T, H, N)
        k_ = k.view(B, T, H, N)
        v_ = v.view(B, T, H, N)
        kk_ = kk.view(B, T, H, N)
        a_ = a.view(B, T, H, N)
        if state is None:
            state = torch.zeros(B, H, N, N, dtype=torch.float32, device=x.device)
        else:
            state = state.to(dtype=torch.float32)
        use_checkpoint = self.training and torch.is_grad_enabled()
        chunk_size = max(1, self.cfg.wkv_chunk_size) if use_checkpoint else T
        ys_chunks = []
        t0 = 0
        while t0 < T:
            t1 = min(t0 + chunk_size, T)
            w_c, k_c, v_c = w_[:, t0:t1], k_[:, t0:t1], v_[:, t0:t1]
            kk_c, a_c, r_c = kk_[:, t0:t1], a_[:, t0:t1], r_[:, t0:t1]
            if use_checkpoint:
                state, y_c = torch.utils.checkpoint.checkpoint(_wkv_run_chunk, state, w_c, k_c, v_c, kk_c, a_c, r_c, use_reentrant=False)
            else:
                state, y_c = _wkv_run_chunk(state, w_c, k_c, v_c, kk_c, a_c, r_c)
            ys_chunks.append(y_c)
            t0 = t1
        xx_out = torch.cat(ys_chunks, dim=1).reshape(B, T, C)
        xx_out = self.ln_x(xx_out.reshape(B * T, C)).reshape(B, T, C)
        xx_out = xx_out + ((r_ * k_ * self.r_k).sum(dim=-1, keepdim=True) * v_).reshape(B, T, C)
        y = self.output(xx_out * g)
        return y, v_first, (state, x[:, -1, :])


class RWKV_CMix_x070(nn.Module):
    def __init__(self, cfg: RWKVXConfig, layer_id: int):
        super().__init__()
        C = cfg.n_embd
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / cfg.n_layer)
            ddd = torch.arange(C, dtype=torch.float32).view(1, 1, C) / C
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        self.key.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.value.weight.data.zero_()

    def forward(self, x, x_prev_last: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        prev0 = x_prev_last.unsqueeze(1) if x_prev_last is not None else torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        x_prev = torch.cat([prev0, x[:, :-1, :]], dim=1)
        xx = x_prev - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k), x[:, -1, :]


class RWKV_CMix_MoE(nn.Module):
    def __init__(self, cfg: RWKVXConfig, layer_id: int):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = min(cfg.num_experts, cfg.num_experts_per_tok)
        self.experts = nn.ModuleList([RWKV_CMix_x070(cfg, layer_id) for _ in range(self.num_experts)])
        self.gate = nn.Linear(cfg.n_embd, self.num_experts, bias=False)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

    def forward(self, x, x_prev_last: Optional[torch.Tensor] = None):
        B, T, C = x.shape
        logits = self.gate(x)
        top_val, top_idx = torch.topk(logits, k=self.top_k, dim=-1)
        top_w = torch.softmax(top_val, dim=-1)
        out = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = (top_idx == e_id)
            if not torch.any(mask):
                continue
            e_out, _ = expert(x, x_prev_last)
            weight = torch.where(mask, top_w, torch.zeros_like(top_w)).sum(dim=-1, keepdim=True)
            out = out + e_out * weight
        return out, x[:, -1, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: RWKVXConfig):
        super().__init__()
        C = cfg.n_embd
        self.n_head = C // cfg.head_size
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.output.weight.data.zero_()

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        q, k, v = self.receptance(x), self.key(x), self.value(x)
        q = q.view(B, T, H, C // H).transpose(1, 2)
        k = k.view(B, T, H, C // H).transpose(1, 2)
        v = v.view(B, T, H, C // H).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.output(y)


class MOBABlock(nn.Module):
    def __init__(self, cfg: RWKVXConfig, layer_id: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.att = CausalSelfAttention(cfg)
        self.ffn = RWKV_CMix_x070(cfg, layer_id) if not cfg.is_moe else RWKV_CMix_MoE(cfg, layer_id)

    def forward(self, x, cmix_state=None):
        x = x + self.att(self.ln1(x))
        ffn_out, new_cmix_state = self.ffn(self.ln2(x), cmix_state)
        x = x + ffn_out
        return x, new_cmix_state


class RWKVBlock(nn.Module):
    def __init__(self, cfg: RWKVXConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(cfg.n_embd)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.att = RWKV_Tmix_x070(cfg, layer_id)
        self.ffn = RWKV_CMix_x070(cfg, layer_id) if not cfg.is_moe else RWKV_CMix_MoE(cfg, layer_id)

    def forward(self, x, v_first, tmix_state=None, cmix_state=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first, new_tmix_state = self.att(self.ln1(x), v_first, tmix_state)
        x = x + xx
        ffn_out, new_cmix_state = self.ffn(self.ln2(x), cmix_state)
        x = x + ffn_out
        return x, v_first, new_tmix_state, new_cmix_state


class RWKVXModel(nn.Module):
    def __init__(self, cfg: RWKVXConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else None
        n_moba = cfg.n_moba_layer
        n_rwkv = cfg.n_layer - n_moba
        assert n_rwkv > 0, "n_moba_layer must be < n_layer"
        self.rwkv_blocks = nn.ModuleList([RWKVBlock(cfg, i) for i in range(n_rwkv)])
        self.moba_blocks = nn.ModuleList([MOBABlock(cfg, n_rwkv + i) for i in range(n_moba)])
        interval = max(1, n_rwkv // max(1, n_moba)) if n_moba > 0 else n_rwkv
        order = []
        ri = 0
        for m in range(n_moba):
            take = interval if m < n_moba - 1 else n_rwkv - ri
            order += [("rwkv", ri + k) for k in range(take)]
            ri += take
            order.append(("moba", m))
        order += [("rwkv", ri + k) for k in range(n_rwkv - ri)]
        self._order = order
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None,
                state: Optional[dict] = None, use_cache: bool = False):
        B, T = idx.shape
        x = self.emb(idx)
        if self.dropout is not None:
            x = self.dropout(x)
        v_first = torch.empty_like(x)
        if state is None:
            tmix_state = [None] * len(self.rwkv_blocks)
            cmix_state = [None] * len(self._order)
        else:
            tmix_state = state["tmix"]
            cmix_state = state["cmix"]
        new_tmix_state = list(tmix_state)
        new_cmix_state = list(cmix_state)
        for pos, (kind, i) in enumerate(self._order):
            if kind == "rwkv":
                block = self.rwkv_blocks[i]
                x, v_first, ts, cs = block(x, v_first, tmix_state[i], cmix_state[pos])
                new_tmix_state[i] = ts
                new_cmix_state[pos] = cs
            else:
                block = self.moba_blocks[i]
                x, cs = block(x, cmix_state[pos])
                new_cmix_state[pos] = cs
        x = self.ln_out(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        new_state = {"tmix": new_tmix_state, "cmix": new_cmix_state} if use_cache else None
        return logits, loss, new_state

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def upstream_compatible_state_dict(self) -> dict:
        out = {}
        out["rwkv.emb.weight"] = self.emb.weight
        for i, blk in enumerate(self.rwkv_blocks):
            prefix = f"rwkv.blocks.{i}."
            sd = blk.state_dict()
            for k, v in sd.items():
                out[prefix + k] = v
        out["rwkv.ln_out.weight"] = self.ln_out.weight
        out["rwkv.ln_out.bias"] = self.ln_out.bias
        out["rwkv.head.weight"] = self.head.weight
        for i, blk in enumerate(self.moba_blocks):
            prefix = f"moba.{i}."
            sd = blk.state_dict()
            for k, v in sd.items():
                out[prefix + k] = v
        return out

    def save_pretrained(self, out_dir: Path):
        from safetensors.torch import save_file
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(sd, str(out_dir / "model.safetensors"))
        self.cfg.save(out_dir / "config.json")
        torch.save({k: v.detach().cpu() for k, v in self.upstream_compatible_state_dict().items()}, out_dir / "rwkvx_upstream_compatible.pth")

    @classmethod
    def from_pretrained(cls, in_dir: Path):
        from safetensors.torch import load_file
        in_dir = Path(in_dir)
        cfg = RWKVXConfig.load(in_dir / "config.json")
        model = cls(cfg)
        sd = load_file(str(in_dir / "model.safetensors"))
        packed = [k for k in sd if k.endswith(".packed")]
        if packed:
            from qat import QuantizedLinear
            for key in packed:
                if not key.endswith(".packed"):
                    continue
                base = key[:-len(".packed")]
                scale_key = base + ".scale"
                if scale_key not in sd:
                    raise RuntimeError(f"QAT checkpoint is missing {scale_key}")
                parent_name, attr = base.rsplit(".", 1)
                if attr not in ("key", "value"):
                    raise RuntimeError(f"Unexpected packed QAT tensor: {key}")
                parent = model.get_submodule(parent_name)
                old = getattr(parent, attr)
                if not isinstance(old, nn.Linear) or old.bias is not None:
                    raise RuntimeError(f"Cannot restore quantized layer {base}")
                numel = old.out_features * old.in_features
                expected = (numel * 3 + 7) // 8
                if sd[key].numel() != expected:
                    raise RuntimeError(f"Invalid packed size for {key}: expected {expected}, got {sd[key].numel()}")
                if sd[scale_key].numel() != old.out_features:
                    raise RuntimeError(f"Invalid scale size for {scale_key}")
                setattr(parent, attr, QuantizedLinear(sd[key], sd[scale_key], old.weight.shape))
        model.load_state_dict(sd, strict=True)
        return model

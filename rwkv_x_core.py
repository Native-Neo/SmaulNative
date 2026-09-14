# rwkv_x_core.py -- RWKV-7 + MOBA hybrid, pure PyTorch CPU.
# decay: w_eff = exp(-exp(w_raw)) == exp(-0.606531*sigmoid(w0+g)).

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


@dataclass
class RWKVXConfig:
    vocab_size: int = 65530
    n_embd: int = 832
    n_layer: int = 17
    head_size: int = 64
    n_moba_layer: int = 5
    moba_chunk_size: int = 512
    moba_topk: int = 4
    dropout: float = 0.0
    head_size_divisor: int = 8
    ctx_len_hint: int = 2048
    wkv_chunk_size: int = 64
    checkpoint_ffn: bool = True
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    qat_bits: int = 0
    rqt_bits: int = 0
    quantization_bits: int = 0

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(Path(path).read_text()))

    def approx_param_count(self) -> int:
        C, V, L = self.n_embd, self.vocab_size, self.n_layer
        dd = max(32, round(1.8 * C**0.5 / 32) * 32)
        dm = max(32, round(1.3 * C**0.5 / 32) * 32)
        dg = max(32, round(0.6 * C**0.8 / 32) * 32)
        tmix = 4 * C * C + C * (4 * dd + 2 * dm + 2 * dg)
        cmix = 8 * C * C
        if self.is_moe:
            cmix = self.num_experts * cmix + C * self.num_experts
        return 2 * V * C + (L - self.n_moba_layer) * (tmix + cmix) + self.n_moba_layer * (4 * C * C + cmix)


def config_for_target_params(target_params: int, vocab_size: int = 65530, n_embd: int = 832,
                             n_moba_layer: int = 5, head_size: int = 64) -> RWKVXConfig:
    if n_embd % head_size:
        raise ValueError(f"n_embd ({n_embd}) must be divisible by head_size ({head_size})")
    best = None
    for n_layer in range(4, 80):
        cfg = RWKVXConfig(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer,
                          n_moba_layer=min(n_moba_layer, n_layer - 1), head_size=head_size)
        diff = abs(cfg.approx_param_count() - target_params)
        if best is None or diff < best[0]:
            best = diff, cfg
    return best[1]


def _wkv_run_chunk(state, w_c, k_c, v_c, kk_c, a_c, r_c):
    k_c, v_c, kk_c, a_c = k_c.float(), v_c.float(), kk_c.float(), a_c.float()
    ys = []
    for t in range(w_c.shape[1]):
        w_t, k_t, v_t, kk_t, a_t, r_t = (w_c[:, t], k_c[:, t], v_c[:, t], kk_c[:, t], a_c[:, t], r_c[:, t])
        u = (-kk_t).unsqueeze(-1)
        av = (kk_t * a_t).unsqueeze(-2)
        flat = state.reshape(-1, state.shape[-2], state.shape[-1])
        state = (state * w_t.unsqueeze(-2) + torch.bmm(torch.bmm(flat, u.reshape(-1, state.shape[-2], 1)),
            av.reshape(-1, 1, state.shape[-1])).reshape_as(state) + torch.bmm(
            v_t.unsqueeze(-1).reshape(-1, state.shape[-1], 1), k_t.unsqueeze(-2).reshape(-1, 1, state.shape[-1])).reshape_as(state))
        ys.append(torch.bmm(state.reshape(-1, state.shape[-2], state.shape[-1]),
            r_t.unsqueeze(-1).reshape(-1, state.shape[-1], 1)).reshape(r_t.shape))
    return state, torch.stack(ys, dim=1)


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.cfg, self.layer_id, self.head_size = cfg, layer_id, cfg.head_size
        C, H, N = cfg.n_embd, cfg.n_embd // cfg.head_size, cfg.head_size
        self.n_head = H
        assert C % N == 0
        with torch.no_grad():
            r0 = layer_id / max(1, cfg.n_layer - 1)
            r1 = 1.0 - layer_id / cfg.n_layer
            ddd = torch.arange(C, dtype=torch.float32).view(1, 1, C) / C
            self.x_r = nn.Parameter(1 - torch.pow(ddd, 0.2 * r1))
            self.x_w = nn.Parameter(1 - torch.pow(ddd, 0.9 * r1))
            self.x_k = nn.Parameter(1 - (torch.pow(ddd, 0.9 * r1) + 0.4 * r0))
            self.x_v = nn.Parameter(1 - (torch.pow(ddd, 0.4 * r1) + 0.6 * r0))
            self.x_a = nn.Parameter(1 - torch.pow(ddd, 0.9 * r1))
            self.x_g = nn.Parameter(1 - torch.pow(ddd, 0.2 * r1))
            def oi(x, scale):
                if x.ndim == 2:
                    nn.init.orthogonal_(x, gain=max(1.0, math.sqrt(x.shape[0] / x.shape[1])) * scale)
                return x
            dd = max(32, round(1.8 * C**0.5 / 32) * 32)
            dm = max(32, round(1.3 * C**0.5 / 32) * 32)
            dg = max(32, round(0.6 * C**0.8 / 32) * 32)
            self.w1 = nn.Parameter(torch.zeros(C, dd))
            self.w2 = nn.Parameter(oi(torch.zeros(dd, C), 0.1))
            self.w0 = nn.Parameter(torch.tensor([-7 + 5 * (n / (C - 1))**(0.85 + r0**0.5) for n in range(C)]).reshape(1, 1, C) + 0.5)
            self.a1 = nn.Parameter(torch.zeros(C, dd))
            self.a2 = nn.Parameter(oi(torch.zeros(dd, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))
            if layer_id:
                self.v1 = nn.Parameter(torch.zeros(C, dm))
                self.v2 = nn.Parameter(oi(torch.zeros(dm, C), 0.1))
                self.v0 = nn.Parameter(torch.ones(1, 1, C))
            self.g1 = nn.Parameter(torch.zeros(C, dg))
            self.g2 = nn.Parameter(oi(torch.zeros(dg, C), 0.1))
            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=1e-5 * cfg.head_size_divisor**2)
        self.receptance.weight.data.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
        self.key.weight.data.uniform_(-0.05 / C**0.5, 0.05 / C**0.5)
        self.value.weight.data.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
        self.output.weight.data.zero_()

    def forward(self, x, v_first, state=None):
        B, T, C = x.shape
        H, N = self.n_head, self.head_size
        prev0 = state[1].unsqueeze(1) if isinstance(state, tuple) else torch.zeros(B, 1, C, dtype=x.dtype, device=x.device)
        state = state[0] if isinstance(state, tuple) else state
        xx = torch.cat([prev0, x[:, :-1]], 1) - x
        xr, xw, xk, xv, xa, xg = (x + xx * p for p in (self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g))
        r = self.receptance(xr)
        g_ = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        v_first = v if self.layer_id == 0 else v_first
        v = v if self.layer_id == 0 else v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = F.normalize((k * self.k_k).view(B, T, H, N), dim=-1).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + g_).float()))
        r_, w_, k_, v_, kk_, a_ = (z.view(B, T, H, N) for z in (r, w, k, v, kk, a))
        state = torch.zeros(B, H, N, N, dtype=torch.float32, device=x.device) if state is None else state.float()
        ys = []
        checkpoint = self.training and torch.is_grad_enabled() and x.device.type != "cpu"
        cs = max(1, self.cfg.wkv_chunk_size) if checkpoint else T
        for t0 in range(0, T, cs):
            args = (w_[:, t0:t0 + cs], k_[:, t0:t0 + cs], v_[:, t0:t0 + cs], kk_[:, t0:t0 + cs], a_[:, t0:t0 + cs], r_[:, t0:t0 + cs])
            state, y = (torch.utils.checkpoint.checkpoint(_wkv_run_chunk, state, *args, use_reentrant=False)
                        if checkpoint else _wkv_run_chunk(state, *args))
            ys.append(y)
        out = ys[0].reshape(B, T, C) if len(ys) == 1 else torch.cat(ys, 1).reshape(B, T, C)
        out = self.ln_x(out.reshape(B * T, C)).reshape(B, T, C)
        out = out + ((r_ * k_ * self.r_k).sum(-1, keepdim=True) * v_).reshape(B, T, C)
        return self.output(out * g), v_first, (state, x[:, -1])


class RWKV_CMix_x070(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        C = cfg.n_embd
        r = 1 - layer_id / cfg.n_layer
        ddd = torch.arange(C, dtype=torch.float32).view(1, 1, C) / C
        self.x_k = nn.Parameter(1 - torch.pow(ddd, r**4))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        self.key.weight.data.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
        self.value.weight.data.zero_()

    def forward(self, x, x_prev_last=None):
        prev = x_prev_last.unsqueeze(1) if x_prev_last is not None else torch.zeros(x.size(0), 1, x.size(-1), dtype=x.dtype, device=x.device)
        xx = torch.cat([prev, x[:, :-1]], 1) - x
        return self.value(torch.relu(self.key(x + xx * self.x_k))**2), x[:, -1]

    def forward_selected(self, x, prev):
        return self.value(torch.relu(self.key(x + (prev - x) * self.x_k.view(-1)))**2)


class RWKV_CMix_MoE(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = min(cfg.num_experts, cfg.num_experts_per_tok)
        self.experts = nn.ModuleList([RWKV_CMix_x070(cfg, layer_id) for _ in range(self.num_experts)])
        self.gate = nn.Linear(cfg.n_embd, self.num_experts, bias=False)
        self.top_k < 1 and (_ for _ in ()).throw(ValueError("num_experts_per_tok must be >= 1"))

    def forward(self, x, x_prev_last=None):
        probs = torch.softmax(self.gate(x), -1)
        topv, topi = torch.topk(probs, self.top_k, -1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        prev = torch.cat([
            x_prev_last.unsqueeze(1) if x_prev_last is not None else torch.zeros_like(x[:, :1]),
            x[:, :-1],
        ], 1)
        for e, expert in enumerate(self.experts):
            weight = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            mask = weight.squeeze(-1) > 0
            if mask.any():
                expert_weight = weight[mask].reshape(-1, 1)
                out[mask] += expert.forward_selected(x[mask], prev[mask]) * expert_weight
        return out, x[:, -1]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        C = cfg.n_embd
        self.n_head = C // cfg.head_size
        self.chunk_size = cfg.moba_chunk_size
        self.top_k = max(0, cfg.moba_topk)
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.receptance.weight.data.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
        self.key.weight.data.uniform_(-0.05 / C**0.5, 0.05 / C**0.5)
        self.value.weight.data.uniform_(-0.5 / C**0.5, 0.5 / C**0.5)
        self.output.weight.data.zero_()

    def forward(self, x, cache=None, use_cache=False):
        B, T, C = x.shape
        H, N, cs, kt = self.n_head, C // self.n_head, self.chunk_size, self.top_k
        q = self.receptance(x).view(B, T, H, N).transpose(1, 2)
        k = self.key(x).view(B, T, H, N).transpose(1, 2)
        v = self.value(x).view(B, T, H, N).transpose(1, 2)

        if cache is not None:
            if T != 1:
                raise ValueError("cached MOBA attention expects one-token decode")
            pk, pv = cache
            past = pk.size(2)
            cur_start = (past // cs) * cs
            prev_k, prev_v = pk[:, :, :cur_start], pv[:, :, :cur_start]
            cur_k = torch.cat((pk[:, :, cur_start:], k), 2)
            cur_v = torch.cat((pv[:, :, cur_start:], v), 2)
            if prev_k.size(2) and kt > 0:
                n_prev = cur_start // cs
                pkc = prev_k.view(B, H, n_prev, cs, N)
                pvc = prev_v.view(B, H, n_prev, cs, N)
                npick = min(kt, n_prev)
                top = torch.einsum("bhqd,bhkd->bhqk", q, pkc.mean(3)).topk(npick, -1).indices
                idx = top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, cs, N)
                sk = torch.gather(pkc.unsqueeze(2).expand(-1, -1, 1, -1, -1, -1), 3, idx).reshape(B, H, npick * cs, N)
                sv = torch.gather(pvc.unsqueeze(2).expand(-1, -1, 1, -1, -1, -1), 3, idx).reshape(B, H, npick * cs, N)
                y = F.scaled_dot_product_attention(q, torch.cat((sk, cur_k), 2), torch.cat((sv, cur_v), 2), is_causal=False)
            else:
                y = F.scaled_dot_product_attention(q, cur_k, cur_v, is_causal=False)
            return self.output(y.transpose(1, 2).reshape(B, 1, C)), (torch.cat((pk, k), 2), torch.cat((pv, v), 2)) if use_cache else None

        n_chunks = (T + cs - 1) // cs
        if kt <= 0 or n_chunks <= kt + 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pad = n_chunks * cs - T
            kc = F.pad(k, (0, 0, 0, pad)).view(B, H, n_chunks, cs, N)
            vc = F.pad(v, (0, 0, 0, pad)).view(B, H, n_chunks, cs, N)
            km = kc.mean(3)
            out = torch.zeros(B, H, T, N, dtype=q.dtype, device=x.device)
            for i in range(n_chunks):
                lo, hi = i * cs, min((i + 1) * cs, T)
                qi = q[:, :, lo:hi]
                ownk, ownv = kc[:, :, i, :hi - lo], vc[:, :, i, :hi - lo]
                if i == 0:
                    yi = F.scaled_dot_product_attention(qi, ownk, ownv, is_causal=True)
                else:
                    npick = min(kt, i)
                    top = torch.einsum("bhd,bhkd->bhk", qi.mean(2), km[:, :, :i]).topk(npick, -1).indices
                    idx = top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, cs, N)
                    sk = torch.gather(kc[:, :, :i], 2, idx).reshape(B, H, npick * cs, N)
                    sv = torch.gather(vc[:, :, :i], 2, idx).reshape(B, H, npick * cs, N)
                    hist_mask = torch.ones(hi - lo, npick * cs, dtype=torch.bool, device=x.device)
                    causal = torch.tril(torch.ones(hi - lo, hi - lo, dtype=torch.bool, device=x.device))
                    base_mask = torch.cat((hist_mask, causal), 1)
                    yi = F.scaled_dot_product_attention(qi, torch.cat((sk, ownk), 2), torch.cat((sv, ownv), 2), attn_mask=base_mask)
                out[:, :, lo:hi] = yi
            y = out
        return self.output(y.transpose(1, 2).contiguous().view(B, T, C)), (k, v) if use_cache else None


class MOBABlock(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.cfg_checkpoint_ffn = cfg.checkpoint_ffn
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.att = CausalSelfAttention(cfg)
        self.ffn = RWKV_CMix_x070(cfg, layer_id) if not cfg.is_moe else RWKV_CMix_MoE(cfg, layer_id)

    def forward(self, x, cmix_state=None, att_state=None, use_cache=False):
        checkpoint = self.training and torch.is_grad_enabled()
        if checkpoint and att_state is None and not use_cache:
            att_out, _ = torch.utils.checkpoint.checkpoint(self.att, self.ln1(x), use_reentrant=False)
            new_att_state = None
        else:
            att_out, new_att_state = self.att(self.ln1(x), att_state, use_cache)
        x = x + att_out
        if self.cfg_checkpoint_ffn and self.training and torch.is_grad_enabled():
            ffn_out, new_cmix_state = torch.utils.checkpoint.checkpoint(self.ffn, self.ln2(x), cmix_state, use_reentrant=False)
        else:
            ffn_out, new_cmix_state = self.ffn(self.ln2(x), cmix_state)
        return x + ffn_out, new_cmix_state, new_att_state


class RWKVBlock(nn.Module):
    def __init__(self, cfg, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.cfg_checkpoint_ffn = cfg.checkpoint_ffn
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(cfg.n_embd)
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.att = RWKV_Tmix_x070(cfg, layer_id)
        self.ffn = RWKV_CMix_x070(cfg, layer_id) if not cfg.is_moe else RWKV_CMix_MoE(cfg, layer_id)

    def forward(self, x, v_first, tmix_state=None, cmix_state=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first, ts = self.att(self.ln1(x), v_first, tmix_state)
        x = x + xx
        if self.cfg_checkpoint_ffn and self.training and torch.is_grad_enabled():
            fo, cs = torch.utils.checkpoint.checkpoint(self.ffn, self.ln2(x), cmix_state, use_reentrant=False)
        else:
            fo, cs = self.ffn(self.ln2(x), cmix_state)
        return x + fo, v_first, ts, cs


class RWKVXModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RWKVXConfig(**cfg)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout else None
        n_moba, n_rwkv = cfg.n_moba_layer, cfg.n_layer - cfg.n_moba_layer
        assert n_rwkv > 0
        self.rwkv_blocks = nn.ModuleList([RWKVBlock(cfg, i) for i in range(n_rwkv)])
        self.moba_blocks = nn.ModuleList([MOBABlock(cfg, n_rwkv + i) for i in range(n_moba)])
        interval = max(1, n_rwkv // max(1, n_moba)) if n_moba else n_rwkv
        self._order, ri = [], 0
        for m in range(n_moba):
            take = interval if m < n_moba - 1 else n_rwkv - ri
            self._order += [("rwkv", ri + k) for k in range(take)]
            ri += take
            self._order.append(("moba", m))
        self._order += [("rwkv", ri + k) for k in range(n_rwkv - ri)]
        self.ln_out = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, labels=None, state=None, use_cache=False, return_logits=True):
        B, T = idx.shape
        x = self.emb(idx)
        x = self.dropout(x) if self.dropout else x
        v_first = state.get("v_first") if state is not None else None
        if state is None:
            tmix_state, cmix_state, att_state = [None] * len(self.rwkv_blocks), [None] * len(self._order), [None] * len(self.moba_blocks)
        else:
            tmix_state, cmix_state, att_state = state["tmix"], state["cmix"], state.get("moba_att", [None] * len(self.moba_blocks))
        nts, ncs, nas = list(tmix_state), list(cmix_state), list(att_state)
        for pos, (kind, i) in enumerate(self._order):
            if kind == "rwkv":
                x, v_first, ts, cs = self.rwkv_blocks[i](x, v_first, tmix_state[i], cmix_state[pos])
                nts[i], ncs[pos] = ts, cs
            else:
                x, cs, ats = self.moba_blocks[i](x, cmix_state[pos], att_state[i], use_cache)
                ncs[pos], nas[i] = cs, ats
        x = self.ln_out(x)
        logits = self.head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        logits = logits if return_logits else None
        new_state = {"tmix": nts, "cmix": ncs, "moba_att": nas, "v_first": v_first} if use_cache else None
        return logits, loss, new_state

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def upstream_compatible_state_dict(self):
        out = {"rwkv.emb.weight": self.emb.weight}
        for i, blk in enumerate(self.rwkv_blocks):
            for k, v in blk.state_dict().items():
                out[f"rwkv.blocks.{i}.{k}"] = v
        out["rwkv.ln_out.weight"] = self.ln_out.weight
        out["rwkv.ln_out.bias"] = self.ln_out.bias
        out["rwkv.head.weight"] = self.head.weight
        for i, blk in enumerate(self.moba_blocks):
            for k, v in blk.state_dict().items():
                out[f"moba.{i}.{k}"] = v
        return out

    def save_pretrained(self, out_dir: Path, dtype: str = "fp32", include_upstream: bool = True):
        from safetensors.torch import save_file
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cast = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        sd = {}
        for k, v in self.state_dict().items():
            v = v.detach().cpu().contiguous()
            sd[k] = v.to(cast) if cast is not None and v.is_floating_point() else v
        save_file(sd, str(out_dir / "model.safetensors"))
        self.cfg.save(out_dir / "config.json")
        if include_upstream:
            torch.save({k: v.detach().cpu() for k, v in self.upstream_compatible_state_dict().items()}, out_dir / "rwkvx_upstream_compatible.pth")

    @classmethod
    def from_pretrained(cls, in_dir: Path):
        from safetensors.torch import load_file
        in_dir = Path(in_dir)
        cfg = RWKVXConfig.load(in_dir / "config.json")
        sd = load_file(str(in_dir / "model.safetensors"))
        model = cls(cfg)
        packed = [k[:-7] for k in sd if k.endswith(".packed")]
        if packed:
            from qat import QuantizedLinear
            for path in packed:
                sk, pk, shape = path + ".scale", path + ".packed", path + ".weight_shape"
                if sk not in sd or shape not in sd:
                    raise RuntimeError(f"quantized layer {path} is missing shape metadata")
                parent_path, name = path.rsplit(".", 1) if "." in path else ("", path)
                parent = model.get_submodule(parent_path) if parent_path else model
                setattr(parent, name, QuantizedLinear(sd[pk], sd[sk], sd[shape].tolist()))
        else:
            qat_keys = [k for k in sd if k.endswith(".weight_fq.scale") or k.endswith(".act_fq.scale")]
            if qat_keys:
                from qat import prepare_qat
                prepare_qat(model)
        model.load_state_dict(sd, strict=True)
        return model

import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TILE = 64
E4M3_MAX = 448.0
_EXT = None
_LUT_CACHE = {}

def _tables(device, dtype=torch.float32):
    key = (str(device), str(dtype))
    hit = _LUT_CACHE.get(key)
    if hit is not None:
        return hit
    v = torch.empty(256, dtype=torch.float32)
    for c in range(256):
        e, m = (c >> 3) & 15, c & 7
        s = -1.0 if c & 128 else 1.0
        if e == 15 and m == 7:
            v[c] = s * 448.0
        elif e == 0:
            v[c] = s * m * 2.0 ** -9
        else:
            v[c] = s * (1 + m / 8.0) * 2.0 ** (e - 7)
    order = torch.argsort(v)
    sval = v[order]
    out = (v.to(device=device, dtype=dtype), order.to(device),
           sval.to(device=device, dtype=dtype),
           ((sval[:-1] + sval[1:]) * 0.5).to(device=device, dtype=dtype))
    _LUT_CACHE[key] = out
    return out

def _lut(device, dtype=torch.float32):
    return _tables(device, dtype)[0]

def _ext():
    global _EXT
    if _EXT is not None:
        return None if _EXT is False else _EXT
    try:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent
        _EXT = load(name="smaul_fp8_ivb", sources=[str(root / "fp8_cpu.cpp")],
            extra_cflags=["-O3", "-mavx", "-mf16c", "-msse4.2", "-mno-avx2", "-mno-avx512f", "-ffp-contract=off"],
            verbose=False)
    except Exception as exc:
        _EXT = False
        warnings.warn(f"FP8 native ext unavailable; torch tiled fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
    return None if _EXT is False else _EXT

def quantize_tiles(w32, tile=TILE):
    with torch.no_grad():
        w32 = w32.float().contiguous()
        out_f, in_f = w32.shape
        nt = (in_f + tile - 1) // tile
        pad = nt * tile - in_f
        if pad:
            w32 = torch.cat([w32, torch.zeros(out_f, pad, dtype=w32.dtype, device=w32.device)], 1)
        _, order, _, bounds = _tables(w32.device, torch.float32)
        blk = w32.reshape(out_f, nt, tile)
        amax = blk.abs().amax(dim=2).clamp_min(1e-12)
        sc = (amax / E4M3_MAX).clamp_min(1e-12)
        n = (blk / sc[..., None]).clamp(-E4M3_MAX, E4M3_MAX)
        nf = n.reshape(-1)
        code = order[torch.bucketize(nf, bounds).clamp(0, 255)].reshape(out_f, nt, tile)
        code.reshape(-1)[nf == 0] = 0
        return code.reshape(out_f, nt * tile)[:, :in_f].to(torch.uint8), sc

def decode_tile(w, s, o0, o1, t, tile=TILE, dtype=torch.float32):
    lut = _lut(w.device, dtype)
    blk = w[o0:o1, t * tile:(t + 1) * tile].long()
    return lut[blk] * s[o0:o1, t].to(dtype)[:, None]

class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, s, in_f, out_f, tile, mod):
        ctx.save_for_backward(x, w, s)
        ctx.in_f, ctx.out_f, ctx.tile, ctx.mod, ctx.xshape = in_f, out_f, tile, mod, x.shape
        ctx.need_x = ctx.needs_input_grad[0]
        x2 = x.reshape(-1, in_f).float().contiguous()
        e = _ext()
        if e is not None and x2.device.type == "cpu":
            y = e.fp8_forward(x2, w.reshape(-1), s.reshape(-1), in_f, out_f, tile)
        else:
            y = torch.zeros(x2.shape[0], out_f, dtype=torch.float32, device=x.device)
            OB, nt = 64, (in_f + tile - 1) // tile
            for o0 in range(0, out_f, OB):
                o1 = min(o0 + OB, out_f)
                acc = torch.zeros(x2.shape[0], o1 - o0, dtype=torch.float32, device=x.device)
                for t in range(nt):
                    k1 = min(in_f, (t + 1) * tile)
                    acc += x2[:, t * tile:k1] @ decode_tile(w, s, o0, o1, t, tile).T
                y[:, o0:o1] = acc
        return y.reshape(*x.shape[:-1], out_f).to(x.dtype if x.dtype != torch.float32 else torch.float32)

    @staticmethod
    def backward(ctx, g):
        x, w, s = ctx.saved_tensors
        in_f, out_f, tile, mod = ctx.in_f, ctx.out_f, ctx.tile, ctx.mod
        g2 = g.reshape(-1, out_f).float().contiguous()
        x2 = x.reshape(-1, in_f).float().contiguous()
        gx = None
        if ctx.need_x:
            e = _ext()
            if e is not None and g2.device.type == "cpu":
                gx = e.fp8_backward_input(g2.contiguous(), w.reshape(-1).contiguous(), s.reshape(-1).contiguous(), in_f, out_f, tile).reshape(ctx.xshape)
            else:
                gx = torch.zeros_like(x2)
                OB, nt = 64, (in_f + tile - 1) // tile
                for o0 in range(0, out_f, OB):
                    o1 = min(o0 + OB, out_f)
                    gb = g2[:, o0:o1]
                    for t in range(nt):
                        gx[:, t * tile:min(in_f, (t + 1) * tile)] += gb @ decode_tile(w, s, o0, o1, t, tile)
                gx = gx.reshape(ctx.xshape)
        if mod.training:
            dw = (g2.T @ x2).float()
            if mod._gw is None:
                mod._gw = dw
            else:
                mod._gw.add_(dw)
        return gx, None, None, None, None, None, None

class FP8Linear(nn.Module):
    def __init__(self, in_f, out_f, tile=TILE, bias=False):
        super().__init__()
        self.in_f, self.out_f, self.tile = in_f, out_f, tile
        nt = (in_f + tile - 1) // tile
        w0 = torch.empty(out_f, in_f // 1 if in_f else 1, dtype=torch.float32)
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        wq, sq = quantize_tiles(w0, tile)
        self.register_buffer("w8", wq)
        self.register_buffer("sc", sq)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self._gw = None
        self.trainable = True

    @classmethod
    def from_float(cls, lin, tile=TILE):
        m = cls(lin.in_features, lin.out_features, tile, lin.bias is not None)
        wq, sq = quantize_tiles(lin.weight.detach().float(), tile)
        m.w8.copy_(wq)
        m.sc.copy_(sq)
        if lin.bias is not None and m.bias is not None:
            with torch.no_grad():
                m.bias.copy_(lin.bias.detach())
        return m

    def forward(self, x):
        return _Fn.apply(x, self.w8, self.sc, self.in_f, self.out_f, self.tile, self) + (0 if self.bias is None else self.bias)

    @torch.no_grad()
    def requant(self, update=None, decay=0.0):
        nt = (self.in_f + self.tile - 1) // self.tile
        OB = 64
        for o0 in range(0, self.out_f, OB):
            o1 = min(o0 + OB, self.out_f)
            cur = torch.cat([decode_tile(self.w8, self.sc, o0, o1, t, self.tile) for t in range(nt)], dim=1)
            if decay:
                cur.mul_(1 - decay)
            if update is not None:
                cur.sub_(update[o0:o1].to(cur.dtype))
            wq, sq = quantize_tiles(cur, self.tile)
            self.w8[o0:o1].copy_(wq)
            self.sc[o0:o1].copy_(sq)
        self._gw = None

    def err_stats(self):
        with torch.no_grad():
            nt = (self.in_f + self.tile - 1) // self.tile
            cur = torch.cat([decode_tile(self.w8, self.sc, 0, self.out_f, t, self.tile) for t in range(nt)], dim=1)
            return {"amax_fp8": cur.abs().amax().item(), "mean_scale": self.sc.mean().item()}

def fp8_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, FP8Linear)]

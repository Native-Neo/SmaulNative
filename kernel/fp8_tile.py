import math
import threading

import torch
import torch.nn as nn

from kernel.compute import get_backend

TILE = 64
_OB = 64
E4M3_MAX = 448.0

# Storage dtype for accumulated FP8 weight gradients (`FP8Linear._gw`).
#
# These buffers are the single largest optimizer-side allocation in the model:
# one [out_f, in_f] tensor per FP8 module, ~480 MiB for the 256M preset. BF16
# halves that. Precision: the block product `g2.T @ x2` is computed in FP32 and
# added into the narrower buffer, so the gradient is rounded exactly once, and
# `_gw` is cleared every step (`zero_grad`), so there is no long accumulation
# chain to drift. Under *gradient accumulation* (several backwards per step)
# this would round on every add and should be revisited.
#
# Consumers must therefore read `_gw` blockwise and must not call `.float()` on
# the whole tensor -- that would allocate an FP32 copy the same size as the
# buffer it replaced, leaving peak memory unchanged.
GW_DTYPE = torch.bfloat16

_LUT_CACHE: dict = {}
_LUT_MAX_ENTRIES = 32

def _tables(device, dtype=torch.float32):
    key = (str(device), str(dtype))
    hit = _LUT_CACHE.get(key)
    if hit is not None:
        # Refresh LRU order.
        _LUT_CACHE.pop(key)
        _LUT_CACHE[key] = hit
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
    if len(_LUT_CACHE) >= _LUT_MAX_ENTRIES:
        _LUT_CACHE.pop(next(iter(_LUT_CACHE)))
    _LUT_CACHE[key] = out
    return out

def _lut(device, dtype=torch.float32):
    return _tables(device, dtype)[0]

_QUANT_TABLE_CACHE: dict = {}


def _quant_tables(device):
    """(order uint8[256], bounds float32[255]) for the native quantizer.

    Derived from the same cached ``_tables`` the torch path uses, so the
    native and torch quantizers cannot disagree about the codebook -- which
    matters because E4M3 has two codes for +448 and two for -448, and the tie
    order comes from torch.argsort.
    """
    key = str(device)
    hit = _QUANT_TABLE_CACHE.get(key)
    if hit is None:
        _, order, _, bounds = _tables(device, torch.float32)
        hit = (order.to(torch.uint8).contiguous(), bounds.contiguous())
        _QUANT_TABLE_CACHE[key] = hit
    return hit


def quantize_tiles(w32, tile=TILE):
    import warnings
    if tile <= 0:
        raise ValueError(f"tile must be > 0, got {tile}")
    if w32.dim() != 2:
        raise ValueError(f"w32 must be 2D [out_f, in_f], got dim={w32.dim()}")
    out_f, in_f = w32.shape
    if out_f <= 0 or in_f <= 0:
        raise ValueError(f"out_f/in_f must be positive, got {out_f}/{in_f}")
    with torch.no_grad():
        w32 = w32.float().contiguous()
        nt = (in_f + tile - 1) // tile
        # The native kernel returns codes padded to a whole number of tiles;
        # both paths are truncated here so the contract is identical.
        native = get_backend().fp8_quantize_tiles(w32, tile)
        if native is not None:
            codes, sc, nonfinite = native
        else:
            pad = nt * tile - in_f
            if pad:
                w32 = torch.cat([w32, torch.zeros(out_f, pad, dtype=w32.dtype,
                                                 device=w32.device)], 1)
            nonfinite = int((~torch.isfinite(w32)).sum())
            _, order, _, bounds = _tables(w32.device, torch.float32)
            blk = w32.reshape(out_f, nt, tile)
            amax = torch.where(torch.isfinite(blk), blk.abs(), 0.0).amax(dim=2).clamp_min(1e-12)
            sc = (amax / E4M3_MAX).clamp_min(1e-12)
            n = torch.nan_to_num(blk / sc[..., None], nan=0.0).clamp(-E4M3_MAX, E4M3_MAX)
            nf = n.reshape(-1)
            codes = order[torch.bucketize(nf, bounds).clamp(0, 255)].reshape(out_f, nt, tile)
            codes = codes.reshape(out_f, nt * tile)
            codes.reshape(-1)[nf == 0] = 0
        if nonfinite:
            # Previously silently saturated to max-finite with a tiny scale,
            # hiding divergence. Warn so training issues surface.
            warnings.warn(f"quantize_tiles: {nonfinite} non-finite weight(s) saturated to finite E4M3",
                          RuntimeWarning, stacklevel=2)
        return codes[:, :in_f].contiguous().to(torch.uint8), sc

def decode_tile(w, s, o0, o1, t, tile=TILE, dtype=torch.float32):
    """Decode rows [o0:o1] of one tile ``t`` to ``dtype``. Ragged-safe.

    Kept per-tile because the torch fallback (``compute._torch_forward`` /
    ``_torch_backward_input``) wants exactly one tile, and because the last tile
    may be short. The requant hot path uses ``decode_block`` instead -- see
    there for why.
    """
    lut = _lut(w.device, dtype)
    blk = w[o0:o1, t * tile:(t + 1) * tile].long()
    return lut[blk] * s[o0:o1, t].to(dtype)[:, None]


def decode_block(w, s, o0, o1, in_f, tile=TILE, dtype=torch.float32):
    """Decode rows [o0:o1] of *all* tiles to ``[o1-o0, in_f]`` in one pass.

    Bit-identical to ``cat([decode_tile(w, s, o0, o1, t) for t in range(nt)])``,
    ragged last tile included -- the requant path rewrites the stored codes from
    this, so any difference would silently change the model.

    Faster because the decode is dispatch-bound, not bandwidth-bound. Decoding a
    64x512 block tile-by-tile measured 1.52 ms, and the int64 advanced-index
    gather inside a single tile is 0.295 ms of that; one ``index_select`` with
    an int32 index over the whole block is 0.232 ms, i.e. 6.5x. The requant
    path calls this once per 64-row block, so 32 FP8 modules x 8 blocks per
    optimizer step goes from ~455 ms to ~45 ms. Two things make the difference:
    one kernel launch instead of ``ceil(in_f/tile)``, and ``index_select``
    (which takes an int32 index) instead of ``lut[idx]`` (which does not).
    """
    lut = _lut(w.device, dtype)
    nt = (in_f + tile - 1) // tile
    rows = w[o0:o1, :in_f]
    # Codes are uint8, so 0..255: the int32 cast is exact, and index_select
    # takes IntTensor or LongTensor, so this is a narrowing of the gather index
    # (4 bytes) rather than a 2x-widening one.
    vals = torch.index_select(lut, 0, rows.reshape(-1).to(torch.int32))
    vals = vals.view(o1 - o0, in_f)
    # Per-tile scale expanded to columns. repeat_interleave then slice rather
    # than a per-tile loop, so the scale is materialized once.
    return vals * s[o0:o1, :nt].to(dtype).repeat_interleave(tile, 1)[:, :in_f]

class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, s, in_f, out_f, tile, mod):
        ctx.save_for_backward(x, w, s)
        ctx.in_f, ctx.out_f, ctx.tile, ctx.mod, ctx.xshape = in_f, out_f, tile, mod, x.shape
        ctx.need_x = ctx.needs_input_grad[0]
        x2 = x.reshape(-1, in_f).float().contiguous()
        y = get_backend().fp8_forward(x2, w, s, in_f, out_f, tile)
        return y.reshape(*x.shape[:-1], out_f).to(x.dtype)

    @staticmethod
    def backward(ctx, g):
        x, w, s = ctx.saved_tensors
        in_f, out_f, tile, mod = ctx.in_f, ctx.out_f, ctx.tile, ctx.mod
        g2 = g.reshape(-1, out_f).float().contiguous()
        x2 = x.reshape(-1, in_f).float().contiguous()
        gx = None
        if ctx.need_x:
            gx = get_backend().fp8_backward_input(g2.contiguous(), w, s, in_f, out_f, tile).reshape(ctx.xshape)
        if mod.training:
            with mod._gw_lock:
                if mod._gw is None:
                    mod._gw = torch.zeros(out_f, in_f, dtype=GW_DTYPE, device=g2.device)
                elif mod._gw.device != g2.device:
                    # Device changed mid-training (e.g. .to(device)); migrate.
                    mod._gw = mod._gw.to(g2.device)
                gw = mod._gw
                if gw.shape != (out_f, in_f):
                    raise RuntimeError(f"_gw shape {tuple(gw.shape)} != ({out_f}, {in_f})")
                for o0 in range(0, out_f, _OB):
                    o1 = min(o0 + _OB, out_f)
                    # RHS is FP32; the destination may be narrower, so this rounds
                    # the block product exactly once. Opt.step() must therefore
                    # consume _gw blockwise and never materialize a full FP32 copy.
                    gw[o0:o1].add_(g2[:, o0:o1].T @ x2)
        return gx, None, None, None, None, None, None

class FP8Linear(nn.Module):
    def __init__(self, in_f, out_f, tile=TILE, bias=False):
        super().__init__()
        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"in_f/out_f must be positive, got {in_f}/{out_f}")
        if tile <= 0:
            raise ValueError(f"tile must be > 0, got {tile}")
        self.in_f, self.out_f, self.tile = in_f, out_f, tile
        self._gw_lock = threading.Lock()
        nt = (in_f + tile - 1) // tile
        w0 = torch.empty(out_f, in_f, dtype=torch.float32)
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        wq, sq = quantize_tiles(w0, tile)
        self.register_buffer("w8", wq)
        self.register_buffer("sc", sq)
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self._gw = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_gw_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._gw_lock = threading.Lock()

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
        out = _Fn.apply(x, self.w8, self.sc, self.in_f, self.out_f, self.tile, self)
        return out if self.bias is None else out + self.bias

    @torch.no_grad()
    def _requant_block(self, o0, o1, update, decay):
        # One whole-block decode, not ceil(in_f/tile) per-tile decodes: see
        # decode_block. The block's own FP32 [o1-o0, in_f] temporary is
        # unchanged, so peak memory still does not scale with the matrix.
        cur = decode_block(self.w8, self.sc, o0, o1, self.in_f, self.tile)
        if decay:
            cur.mul_(1 - decay)
        if update is not None:
            cur.sub_(update.to(cur.dtype))
        wq, sq = quantize_tiles(cur, self.tile)
        self.w8[o0:o1].copy_(wq)
        self.sc[o0:o1].copy_(sq)

    @torch.no_grad()
    def requant(self, update=None, decay=0.0):
        if update is not None:
            if update.dim() != 2 or update.shape != (self.out_f, self.in_f):
                raise ValueError(
                    f"requant update must be [{self.out_f}, {self.in_f}], "
                    f"got {tuple(update.shape)} (transposed [in_f, out_f] is a common bug)")
        for o0 in range(0, self.out_f, _OB):
            o1 = min(o0 + _OB, self.out_f)
            self._requant_block(o0, o1, update[o0:o1] if update is not None else None, decay)
        self._gw = None

    @torch.no_grad()
    def fused_lion_requant(self, gw, st, lr, wd, b1, b2):
        """Tile-local Lion update fused into requantization.

        Same elementwise math as the full-matrix path (sign step from FP32
        momentum, decay folded into requant), but computed one output block
        (``_OB`` rows) at a time: no full-matrix ``upd`` transient is ever
        built, and only the block's tiles are decoded. Clears ``self._gw``.
        """
        for o0 in range(0, self.out_f, _OB):
            o1 = min(o0 + _OB, self.out_f)
            # Widened per block, not per matrix: .float() on an fp32 block is a
            # no-op view, on a narrow block a block-sized copy. Either way no
            # full-matrix FP32 transient is ever built.
            g_b, s_b = gw[o0:o1].float(), st[o0:o1]
            upd_b = (s_b * b1 + g_b * (1.0 - b1)).sign() * lr
            s_b.mul_(b2).add_(g_b, alpha=1.0 - b2)
            self._requant_block(o0, o1, upd_b, lr * wd)
        self._gw = None

    def err_stats(self):
        with torch.no_grad():
            nt = (self.in_f + self.tile - 1) // self.tile
            # Stream tiles: avoid materializing the full [out_f, in_f] FP32 matrix (OOM on 4k+).
            amax = 0.0
            ssum, scount = 0.0, 0
            for t in range(nt):
                blk = decode_tile(self.w8, self.sc, 0, self.out_f, t, self.tile)
                amax = max(amax, float(blk.abs().amax()))
                ssum += float(self.sc[:, t].float().sum())
                scount += self.sc.shape[0]
            return {"amax_fp8": amax, "mean_scale": ssum / max(1, scount)}

def fp8_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, FP8Linear)]

#!/usr/bin/env python3
"""SmaulLinear FP8 trainer: pretraining with tiled E4M3 weights, FP32 Lion/SmaulOpt.

Checkpoints are resume-free for Lion (hyperparams only); SmaulOpt
(--optimizer smaul) additionally writes FP32 m/v states for exact resume.
"""
import argparse
import json
import math
import os
import signal
import time
from pathlib import Path

import torch

from dataset import PretrainStream, TokenizerWrapper, discover_files, iter_texts
from kernel.fp8_tile import fp8_modules
from smaul_linear import LinearConfig, SmaulLinear
from tokenizer import ensure_tokenizer

# Named model-size presets: preset name -> dict(vocab, d, layers, heads, ffn_mult).
# Effective fp32-equivalent params ~= 2*vocab*d + (5*layers+2)*d
#   + layers*(4*d*d + 3*d*int(d*ffn_mult)). FP8 per-tile scales add ~1-2% on top.
# Presets are added one per commit, largest first.
PRESETS: dict = {
    # ~1,024M params (1.024B target, 64K vocab).
    "1B": {"vocab": 65536, "d": 2048, "layers": 18, "heads": 16, "ffn_mult": 2.0},
    # ~508M params (512M target, 64K vocab).
    "512M": {"vocab": 65536, "d": 1536, "layers": 13, "heads": 12, "ffn_mult": 2.0},
    # ~260M params (256M target, 64K vocab).
    "256M": {"vocab": 65536, "d": 1024, "layers": 12, "heads": 8, "ffn_mult": 2.0},
    # ~132M params (128M target).
    "128M": {"vocab": 8000, "d": 1024, "layers": 11, "heads": 8, "ffn_mult": 2.0},
    # ~65M params (64M target).
    "64M": {"vocab": 8000, "d": 768, "layers": 9, "heads": 12, "ffn_mult": 2.0},
    # ~32M params (32M target, matches previous defaults).
    "32M": {"vocab": 8000, "d": 512, "layers": 8, "heads": 8, "ffn_mult": 2.5},
    # ~16M params (16M target).
    "16M": {"vocab": 4000, "d": 512, "layers": 4, "heads": 8, "ffn_mult": 2.5},
    # ~7.8M params (8M target).
    "8M": {"vocab": 2000, "d": 448, "layers": 3, "heads": 7, "ffn_mult": 2.0},
    # ~4.2M params (4M target).
    "4M": {"vocab": 512, "d": 256, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~2.1M params (2M target).
    "2M": {"vocab": 256, "d": 256, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~1.05M params (1M target).
    "1M": {"vocab": 256, "d": 128, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~526K params (512K target).
    "512K": {"vocab": 128, "d": 128, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~256K params (256K target).
    "256K": {"vocab": 64, "d": 64, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~132K params (128K target).
    "128K": {"vocab": 64, "d": 64, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~67K params (64K target).
    "64K": {"vocab": 32, "d": 56, "layers": 2, "heads": 4, "ffn_mult": 2.0},
    # ~33K params (32K target).
    "32K": {"vocab": 32, "d": 32, "layers": 3, "heads": 2, "ffn_mult": 2.0},
    # ~16.6K params (16K target).
    "16K": {"vocab": 96, "d": 32, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~8.2K params (8K target).
    "8K": {"vocab": 48, "d": 24, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~4.2K params (4K target).
    "4K": {"vocab": 48, "d": 16, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~2.1K params (2K target).
    "2K": {"vocab": 24, "d": 12, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~1.08K params (1K target).
    "1K": {"vocab": 24, "d": 8, "layers": 1, "heads": 2, "ffn_mult": 2.0},
}


def list_presets() -> dict:
    return dict(PRESETS)


def estimate_params(vocab: int, d: int, layers: int, ffn_mult: float) -> int:
    h = int(d * ffn_mult)
    return 2 * vocab * d + (5 * layers + 2) * d + layers * (4 * d * d + 3 * d * h)


def apply_preset(args) -> None:
    name = getattr(args, "preset", None)
    if not name:
        return
    try:
        p = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown --preset {name!r}; use --list-presets (have {sorted(PRESETS)})") from None
    for k in ("vocab", "d", "layers", "heads", "ffn_mult"):
        if k in p:
            setattr(args, k, p[k])

STOP = False
def _h(sig, fr):
    global STOP
    STOP = True
    print("\n[stop] finishing step then saving")

# Elements per block in the global grad-norm reduction (see _grad_norm).
# 2**19 x 4B = 2 MiB FP32 upcast, which is small enough to stay resident in
# cache and large enough that the Python-level loop is not the bottleneck.
_NORM_BLOCK = 1 << 19


def _grad_norm(grads) -> float:
    """L2 norm of the concatenated gradients, accumulated in float64.

    Two things this deliberately does NOT do.

    1. It does not widen a whole gradient to the accumulator dtype first.
       The previous form built a full FP32 copy of *every* gradient and held
       them in a list for the whole call, then made two FP64 copies of one of
       them -- ~10x the gradient bytes in transients (measured 314 MiB for a
       single 31 MiB bf16 gradient). That is precisely the allocation
       ``SmaulOpt.narrow_grads_`` exists to avoid, two lines later in the
       same call. Upcasting one block at a time bounds the transient to
       _NORM_BLOCK elements instead.

    2. It does not accumulate in float32 or bfloat16. This is not a stylistic
       choice and float32 is NOT a drop-in: ``torch.linalg.vector_norm``
       accumulates linearly, so on a 16M-element tensor its float32 result is
       6.4e-4 relative off (bfloat16 3.5e-4, since the squares are formed in
       the input dtype) -- far worse than float32's own ~6e-8 capability.
       These gradients *are* the clip threshold, so that error would be a
       silent change in clipping behaviour. Blocking the upcast keeps the
       exact float64 result at a fraction of the memory and time.

    Measured on this repo, one 8000x2048 bf16 gradient, 2 threads::

        form                            transient      time    rel err
        .float() in list + .double()   314.2 MiB    0.196 s   0
        vector_norm(dtype=float32)        ~0 MiB    0.042 s   6.4e-4
        this function                  10.7 MiB    0.081 s   1.1e-16

    Non-finite entries propagate: a NaN gradient yields a NaN norm and an
    infinite one an infinite norm, so callers no longer need a separate
    ``isfinite`` scan over every gradient (that scan measured 0.130 s, i.e.
    149% of the norm computation itself, because it re-reads all the bytes).
    """
    total = torch.zeros((), dtype=torch.float64)
    for g in grads:
        n = int(g.numel())
        if n == 0:
            continue
        # reshape(-1) is a free view for the contiguous gradients autograd
        # produces; a non-contiguous gradient costs one bf16 copy, which is
        # still no worse than the FP32 copy the old form always made.
        flat = g.reshape(-1)
        for i in range(0, n, _NORM_BLOCK):
            blk = flat[i:i + _NORM_BLOCK].float()
            total += torch.linalg.vector_norm(blk, ord=2,
                                              dtype=torch.float64).pow(2)
    return float(total.sqrt().item())

def install_handlers() -> None:
    # Install SIGINT/SIGTERM handlers explicitly from main() only.
    # Importing train (e.g. benchmark.py imports Lion) must not
    # hijack process signals as a side effect.
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _h)
        except (OSError, ValueError):
            pass

class Lion:
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), wd=0.01, clip=1.0):
        self.p = [p for p in params if p.requires_grad]
        import math as _math
        # Same fail-fast contract SmaulOpt applies to its own hyperparameters:
        # coerce, then check finiteness and range. Only `clip` was checked here,
        # so a negative learning rate silently ascended the loss, a beta outside
        # [0, 1) ran the momentum away, and a NaN learning rate turned every
        # weight into NaN without a word. train.py's CLI happens to validate
        # these before reaching here, but Lion is also constructed directly --
        # by rl.py, and by any caller importing it.
        try:
            _lr = float(lr)
            _b1, _b2 = float(betas[0]), float(betas[1])
            _wd = float(wd)
            _clip = float(clip)
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"invalid hyperparameters lr={lr!r} betas={betas!r} "
                f"wd={wd!r} clip={clip!r}") from None
        if not _math.isfinite(_lr) or _lr <= 0:
            raise ValueError(f"lr must be positive finite, got {lr!r}")
        if not _math.isfinite(_b1) or not 0.0 <= _b1 < 1.0:
            raise ValueError(f"betas[0] must be in [0, 1), got {betas[0]!r}")
        if not _math.isfinite(_b2) or not 0.0 <= _b2 < 1.0:
            raise ValueError(f"betas[1] must be in [0, 1), got {betas[1]!r}")
        if not _math.isfinite(_wd) or _wd < 0:
            raise ValueError(f"wd must be non-negative finite, got {wd!r}")
        if not _math.isfinite(_clip) or _clip <= 0:
            raise ValueError(f"clip must be positive finite, got {clip!r}")
        self.lr, self.b1, self.b2, self.wd, self.clip = _lr, _b1, _b2, _wd, _clip
        self.m = {}
    def zero_grad(self, model=None):
        for p in self.p:
            # Detach first so stale graph refs cannot accumulate.
            p.grad = None
        if model is not None:
            for _, m in fp8_modules(model):
                m._gw = None
        else:
            # Without the model we cannot clear tiled FP8 grads; warn once
            # because the next step would otherwise double-count them.
            import warnings
            warnings.warn("Lion.zero_grad called without model; FP8 _gw not cleared", RuntimeWarning,
                          stacklevel=2)
    @torch.no_grad()
    def _clip(self, mods, mx=1.0):
        grads = [m._gw for _, m in mods if m._gw is not None] + \
                [p.grad for p in self.p if p.grad is not None]
        if not grads:
            return 0.0
        # Non-finite grads (or a non-finite norm) would poison quantized
        # weights via sign(); report inf so the caller skips the step.
        t = _grad_norm(grads)
        if not math.isfinite(t):
            return float("inf")
        if t > mx:
            s = mx / (t + 1e-6)
            for _, m in mods:
                if m._gw is not None:
                    m._gw.mul_(s)
            for p in self.p:
                if p.grad is not None:
                    p.grad.mul_(s)
        return t
    @torch.no_grad()
    def step(self, model):
        mods = fp8_modules(model)
        norm = self._clip(mods, self.clip)
        if norm == float("inf"):
            # Non-finite grads would poison quantized weights via sign().
            # Clear them and skip the update; caller also guards loss.
            self.zero_grad(model)
            return norm
        if not any(m._gw is not None for _, m in mods) \
                and not any(p.grad is not None for p in self.p):
            # No gradients at all: nothing to update. Return before advancing
            # the (nonexistent) bias correction or evicting live momentum as
            # "dead", exactly as the inf path above skips the counter.
            return 0.0
        live = set()
        for _, m in mods:
            if m._gw is None:
                continue
            # Passed narrow: fused_lion_requant widens per output block, so no
            # full-matrix FP32 copy of the gradient is ever built.
            gw = m._gw
            st = self.m.get(m)
            if st is None or st.shape != tuple(gw.shape):
                st = torch.zeros(tuple(gw.shape), dtype=torch.float32, device=gw.device)
                self.m[m] = st
            elif st.device != gw.device:
                st = st.to(gw.device)
                self.m[m] = st
            live.add(m)
            m.fused_lion_requant(gw, st, self.lr, self.wd, self.b1, self.b2)
        for p in self.p:
            if p.grad is None:
                continue
            g = p.grad
            st = self.m.get(p)
            if st is None or st.shape != tuple(p.shape):
                st = torch.zeros(tuple(p.shape), dtype=torch.float32, device=g.device)
                self.m[p] = st
            elif st.device != g.device:
                st = st.to(g.device)
                self.m[p] = st
            live.add(p)
            # Blocked over rows for 2-D parameters so the sign-update never
            # holds full-matrix transients (the embedding is the large one).
            # Elementwise throughout, so blocking is bit-exact. Narrower
            # gradients widen per block via promotion, never whole.
            if g.dim() >= 2:
                rows = g.shape[0]
                for o0 in range(0, rows, 256):
                    o1 = min(o0 + 256, rows)
                    gb = g[o0:o1].float()
                    sb = st[o0:o1]
                    upd = sb.mul(self.b1).add(gb, alpha=1 - self.b1).sign()
                    if self.wd:
                        p[o0:o1].mul_(1 - self.lr * self.wd)
                    p[o0:o1].add_(upd.to(p.dtype), alpha=-self.lr)
                    sb.mul_(self.b2).add_(gb, alpha=1 - self.b2)
            else:
                gb = g.float()
                upd = st.mul(self.b1).add(gb, alpha=1 - self.b1).sign()
                if self.wd:
                    p.mul_(1 - self.lr * self.wd)
                p.add_(upd.to(p.dtype), alpha=-self.lr)
                st.mul_(self.b2).add_(gb, alpha=1 - self.b2)
        # Evict momentum for dead params/modules (e.g. architecture change).
        for k in list(self.m):
            if k not in live:
                del self.m[k]
        return norm
    def state_dict(self):
        # Resume-free: momentum (self.m) is intentionally not saved.
        return {"name": "lion", "lr": self.lr, "wd": self.wd, "betas": [self.b1, self.b2],
                "clip": self.clip}
    def load_state_dict(self, d):
        # Tolerant loader: old checkpoints have no "name"; SmaulOpt checkpoints
        # may be loaded for hyperparams only (Lion needs no momentum state).
        # Use .get so unknown/missing keys keep current values.
        self.lr = d.get("lr", d.get("learning_rate", self.lr))
        self.wd = d.get("wd", d.get("weight_decay", self.wd))
        betas = d.get("betas", [self.b1, self.b2])
        # Allow Smaul-style beta_m/beta_v keys when betas missing.
        if "betas" not in d and ("beta_m" in d or "beta_v" in d):
            try:
                betas = [float(d.get("beta_m", self.b1)), float(d.get("beta_v", self.b2))]
            except (TypeError, ValueError):
                betas = [self.b1, self.b2]
        try:
            self.b1, self.b2 = float(betas[0]), float(betas[1])
        except (TypeError, IndexError, ValueError):
            pass
        try:
            clip = float(d.get("clip", self.clip))
            if clip > 0:
                self.clip = clip
        except (TypeError, ValueError):
            pass


class SmaulOpt:
    """SmaulOpt: small deterministic FP32-arithmetic adaptive optimizer.

    Per trainable parameter stores exactly two states:
      m: momentum (EMA of gradient)
      v: EMA of elementwise gradient magnitude |g|

    Update (t is 1-indexed global step)::

      m_t = beta_m * m_{t-1} + (1 - beta_m) * g_t
      v_t = beta_v * v_{t-1} + (1 - beta_v) * |g_t|
      m_hat = m_t / (1 - beta_m**t)
      v_hat = v_t / (1 - beta_v**t)
      u_t = m_hat / (v_hat + epsilon)
      theta_t = theta_{t-1} - lr * u_t - lr * wd * theta_{t-1}

    Weight decay is decoupled. No gradient clipping/normalization is done
    inside the update; the existing global-norm clip is applied to the
    incoming (already clipped by caller convention) grads exactly like Lion.
    FP8 weights (buffers w8/sc, grads in _gw) are updated via the existing
    requant path blockwise; no FP32 master copy is created.

    State storage precision (``state_dtype``, default ``bf16``)
    ---------------------------------------------------------
    The equations above are ALWAYS evaluated in FP32, for every
    ``state_dtype``. Only the *stored* buffers change width, so the math is
    identical and only the quantization of the persisted state differs:

    =======  ==================  ======================================
    dtype    bytes/element       storage
    =======  ==================  ======================================
    bf16     2                   bfloat16 (default; FP32 exponent range)
    fp16     2                   float16 (clips at 65504, flushes <6e-8)
    fp32     4                   float32 (lossless reference)
    =======  ==================  ======================================

    ``bf16`` is the default because it halves the state at essentially no
    accuracy cost: measured ~0.07% relative parameter error against the
    ``fp32`` state, versus ~18% for a 1-byte integer state that was tried and
    removed. ``bf16`` is also the cheapest reduced width -- narrowing a
    float state is just a cast, whereas an integer state additionally needs
    an absmax reduction, a scale, a divide, a round, a clamp, and a
    non-finite guard before the integer cast. Measured per 64x512 block over
    5 repeats: bf16 336 us vs 1772 us for the integer path, i.e. ~5x, for
    only 2x less memory.

    Pass ``state_dtype="fp32"`` for the lossless v1 behavior. The 1-byte
    integer path was removed on purpose: it was ~250x less accurate, ~5x
    slower, and had a divergence mode (a heavy-tailed gradient sets one
    block's scale, so other elements' ``v`` rounds to zero and
    ``u = m_hat / (0 + epsilon)`` blows up, measured |u| ~ 1.7e10). Re-adding
    it would need outlier handling, not just a wider integer.

    FP8 weights are unaffected by ``state_dtype``: only optimizer state
    storage narrows, never the weight format or the update math.

    Factored ``v`` (``factor_v``, default on)
    -----------------------------------------
    For a matrix-shaped state ``[R, C]`` the full ``v`` is replaced by two
    BF16 vectors, ``v_row[R]`` and ``v_col[C]``. ``m`` is always full-size.

    **Derivation.** The EMA is *linear*, so it commutes with means::

        rowmean_i(v_t) = beta_v * rowmean_i(v_{t-1}) + (1 - beta_v) * rowmean_i(|g_t|)

    Therefore maintaining an EMA of the row means and of the column means
    yields the **exact** row and column marginals of the true ``v_t`` -- no
    approximation at that stage. The only approximation is dropping the
    rank/interaction term. Given exact marginals ``R`` and ``C`` and grand
    mean ``G = mean(R) = mean(C)``, the unique rank-1 field consistent with
    both marginals and the grand mean is the outer product::

        v_hat[i, j] ~= R_hat[i] * C_hat[j] / G_hat

    That is the marginal-preserving reconstruction, and it is what this
    implements. It is *not* AdaFactor's form: AdaFactor factors ``EMA(g^2)``
    with ``R_i * C_j`` and no division, and it only ever needs the result
    under a ``sqrt``. SmaulOpt's statistic is ``EMA(|g|)``, which is
    non-negative, so the division by the grand mean is what keeps the
    reconstruction consistent with the stored marginals. Measured on random
    gradients, dividing roughly halves the error: 0.30 relative versus 0.64
    for the un-divided form, and the reconstruction is exact for a rank-1
    ``v``.

    Consequences to be aware of (measured, see the numerical comparison test):
    a rank-1 field is *dense*, so a sparse ``v`` cannot be represented. On a
    0.5%-nonzero gradient the reconstruction under-estimates ``v`` about 3x at
    the non-zeros, which inflates the step there, since ``u`` has ``v`` in the
    denominator. This optimizer's memory win comes from the embedding and LM
    head, which are exactly the large and often sparse tensors.

    Factoring is decided by shape alone -- no architecture-specific names --
    and applies when the parameter is 2-D with both extents >= 2, where
    ``R + C <= R * C`` so the factored form is never larger. Vectors and
    scalars keep a full ``v``.
    """

    # Rows per requant/quantize block for FP8 modules (matches fp8_tile._OB).
    # Must stay 64: the block is requantized as a unit, so this is part of the
    # weight-update granularity, not just a cache hint.
    _OB = 64
    # Rows per block for *dense* 2-D parameters. Unconstrained by requantization
    # (they are not quantized), so it is picked for speed: see _row_blocks.
    _DENSE_OB = 256
    _STATE_DTYPES = ("bf16", "fp16", "fp32")

    def __init__(self, params, lr=1e-4, beta_m=0.9, beta_v=0.999, epsilon=1e-8,
                 weight_decay=0.01, clip=1.0, state_dtype="bf16", update_clip=10.0,
                 factor_v=True, grad_dtype="bf16"):
        import math as _math
        try:
            _lr = float(lr)
        except (TypeError, ValueError):
            raise ValueError(f"lr must be a positive finite float, got {lr!r}") from None
        try:
            _bm = float(beta_m)
        except (TypeError, ValueError):
            raise ValueError(f"beta_m must be in [0, 1), got {beta_m!r}") from None
        try:
            _bv = float(beta_v)
        except (TypeError, ValueError):
            raise ValueError(f"beta_v must be in [0, 1), got {beta_v!r}") from None
        try:
            _eps = float(epsilon)
        except (TypeError, ValueError):
            raise ValueError(f"epsilon must be positive finite, got {epsilon!r}") from None
        try:
            _wd = float(weight_decay)
        except (TypeError, ValueError):
            raise ValueError(f"weight_decay must be non-negative finite, got {weight_decay!r}") from None
        try:
            _clip = float(clip)
        except (TypeError, ValueError):
            raise ValueError(f"clip must be positive, got {clip!r}") from None
        if not _math.isfinite(_lr) or _lr <= 0:
            raise ValueError(f"lr must be positive finite, got {lr!r}")
        if not _math.isfinite(_bm) or not 0.0 <= _bm < 1.0:
            raise ValueError(f"beta_m must be in [0, 1), got {beta_m!r}")
        if not _math.isfinite(_bv) or not 0.0 <= _bv < 1.0:
            raise ValueError(f"beta_v must be in [0, 1), got {beta_v!r}")
        if not _math.isfinite(_eps) or _eps <= 0:
            raise ValueError(f"epsilon must be positive finite, got {epsilon!r}")
        if not _math.isfinite(_wd) or _wd < 0:
            raise ValueError(f"weight_decay must be non-negative finite, got {weight_decay!r}")
        if not _math.isfinite(_clip) or _clip <= 0:
            raise ValueError(f"clip must be positive, got {clip!r}")
        if state_dtype not in self._STATE_DTYPES:
            raise ValueError(
                f"state_dtype must be one of {self._STATE_DTYPES}, got {state_dtype!r}")
        try:
            _uclip = float(update_clip)
        except (TypeError, ValueError):
            raise ValueError(
                f"update_clip must be a positive finite float, got {update_clip!r}") from None
        if not _math.isfinite(_uclip) or _uclip <= 0:
            raise ValueError(f"update_clip must be positive finite, got {update_clip!r}")
        if grad_dtype not in (None, "bf16", "fp16", "fp32"):
            raise ValueError(
                f"grad_dtype must be None, 'bf16', 'fp16' or 'fp32', got {grad_dtype!r}")
        # "fp32" is spelled as None internally: keep the gradients as autograd left
        # them and do not narrow them.
        if grad_dtype == "fp32":
            grad_dtype = None
        factor_v = bool(factor_v)
        self.p = [p for p in params if p.requires_grad]
        self.lr = _lr
        self.beta_m = _bm
        self.beta_v = _bv
        self.epsilon = _eps
        self.weight_decay = _wd
        self.clip = _clip
        self.state_dtype = state_dtype
        # Bound on |u| for reduced-precision state. In exact arithmetic
        # |EMA(g)| <= EMA(|g|) implies |u| < 1. Narrow state can nudge it just
        # over that line (m_hat and v_hat use different bias corrections early
        # on), so this clamps marginally in normal operation. It is kept as a
        # guard rather than a crutch: the catastrophic mode where |u| ran to
        # ~1e10 and destroyed the weights belonged to the removed 1-byte integer
        # path, and a future state format could reintroduce it. fp32 is never
        # clamped. See test_update_clip_is_defensive_not_load_bearing.
        self.update_clip = _uclip
        self.m: dict = {}
        # v is held either full-size (self.v) or factored into two marginal
        # vectors (self.v_row / self.v_col), decided per-state by shape.
        self.v: dict = {}
        self.v_row: dict = {}
        self.v_col: dict = {}
        self.factor_v = factor_v
        self.grad_dtype = grad_dtype
        # Set from a checkpoint when one is loaded; the default assumes full-v.
        self.checkpoint_factor_v = bool(factor_v)
        # Global step for bias correction. Named step_count (not step) so it
        # does not shadow the step() method.
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # State storage (v2). The update itself is always FP32; these helpers only
    # decide how wide the persisted m/v buffers are. They are no-ops for
    # state_dtype fp32, which keeps the lossless path allocation-identical.
    # ------------------------------------------------------------------
    def narrow_grads_(self, model=None):
        """Cast dense ``p.grad`` to ``grad_dtype`` after ``backward()``.

        Called by the training loop between ``backward()`` and ``step()``. The
        FP32 gradient produced by autograd is rounded exactly once, on a value
        that is already fully summed over the batch and context, so this is the
        benign case -- not an accumulation in reduced precision. The FP32
        temporary is released immediately, which is the point: dense gradients
        are the largest remaining allocation after ``_gw`` (~512 MiB for the
        256M preset).

        No-op when ``grad_dtype`` is None. FP8 ``_gw`` is already stored at
        ``kernel.fp8_tile.GW_DTYPE`` and needs nothing here.
        """
        if self.grad_dtype is None:
            return self
        dt = torch.bfloat16 if self.grad_dtype == "bf16" else torch.float16
        for p in self.p:
            if p.grad is None or not p.grad.is_floating_point():
                continue
            # PyTorch refuses a grad whose dtype differs from the parameter's
            # grad_dtype (which defaults to the param dtype); None is the
            # documented opt-out that allows any floating dtype.
            p.grad_dtype = None
            if p.grad.dtype != dt:
                p.grad = p.grad.to(dt)
        return self

    def _storage_dtype(self, signed):
        if self.state_dtype == "bf16":
            return torch.bfloat16
        if self.state_dtype == "fp16":
            return torch.float16
        return torch.float32

    def _empty_state(self, ref_shape, ref_device, signed):
        return torch.zeros(ref_shape, dtype=self._storage_dtype(signed), device=ref_device)

    def _row_blocks(self, tensor, block=None):
        """Yield (block_index, o0, o1) for a 2-D state, one block per ``block`` rows.

        Blocks are independent -- each computes its own slice of the update from
        that slice of the gradient and writes back to it -- so the block *count*
        is a pure performance knob and cannot change the result. Verified: the
        dense factored-v update below is bit-identical at 64, 128, 256, 512, 1000
        and 8000 rows per block.

        Two sizes, because the two callers want different things:

        - ``_OB`` (64) for the 2-D FP8 module states. It has to match
          ``kernel.fp8_tile._OB`` because the block is requantized as a unit:
          ``_requant_block`` decodes and re-encodes exactly the rows it is
          given. Any other size would change the requantization granularity.
        - ``_DENSE_OB`` (256) for dense 2-D parameters, whose only per-block
          transient is the row block itself. They were sharing ``_OB``, which
          meant the 8000x512 embedding -- 95% of the trainable values at
          ``--rawr-sparsity 0.99`` -- ran 125 Python iterations of ~13 elementwise
          ops for 108 ms. At 256 rows it is 31 iterations and 78 ms, with a 512
          KiB transient that still fits L2. It goes back up past that (8000 rows
          in one block is 124 ms and a 16 MiB transient), so 256 is measured, not
          assumed.
        """
        step = self._OB if block is None else block
        n = tensor.shape[0]
        for b, o0 in enumerate(range(0, n, step)):
            yield b, o0, min(o0 + step, n)

    # ------------------------------------------------------------------
    # Factored v. Shape-driven only: no architecture or module names.
    # ------------------------------------------------------------------
    def _factor_shape(self, shape):
        """True when this state's v should be stored factored.

        2-D with both extents >= 2. For such shapes ``R + C <= R * C``, so the
        factored form is never larger than the full one; 1-D and 0-D states
        cannot be row/column factored and keep a full v.
        """
        if not self.factor_v or len(shape) != 2:
            return False
        r, c = int(shape[0]), int(shape[1])
        return r >= 2 and c >= 2

    @staticmethod
    def _mean_abs(g, dim):
        """mean(|g|) along `dim` as a reduction, with no [R, C] temporary.

        ``g.abs().mean(dim)`` materializes a full-size copy of |g|. The L1 norm
        is the same quantity as a fused reduction, so it allocates only the
        output vector. Accumulated in FP32 even when `g` is narrower: a bf16
        reduction is ~2e-3 relative off, and the class contract is that the
        update math is always FP32 with only the stored buffers narrow.
        """
        if dim == 0:
            n = g.shape[0]
        else:
            n = g.shape[1]
        return torch.linalg.vector_norm(g, ord=1, dim=dim, dtype=torch.float32) / n

    def _v_marginal_state(self, key, shape, device):
        """Fetch or lazily create the factored (row, col) marginals for `key`."""
        r, c = int(shape[0]), int(shape[1])
        dt = self._storage_dtype(signed=True)
        st_r = self.v_row.get(key)
        if st_r is None or st_r.shape != (r,):
            st_r = torch.zeros(r, dtype=dt, device=device)
            self.v_row[key] = st_r
        elif st_r.device != device:
            st_r = st_r.to(device)
            self.v_row[key] = st_r
        st_c = self.v_col.get(key)
        if st_c is None or st_c.shape != (c,):
            st_c = torch.zeros(c, dtype=dt, device=device)
            self.v_col[key] = st_c
        elif st_c.device != device:
            st_c = st_c.to(device)
            self.v_col[key] = st_c
        return st_r, st_c

    @staticmethod
    def _factored_hat(r32, c32, bc):
        """Bias-corrected marginals and grand mean for the reconstruction."""
        r_hat = r32 / bc
        c_hat = c32 / bc
        # The grand mean is 0 only when v is 0 everywhere, i.e. every gradient
        # seen so far was 0. Unguarded, R*C/0 would be 0/0 = NaN, and the NaN
        # would then make u non-finite and skip the block -- silently dropping
        # that block's weight decay. A tiny positive floor keeps the
        # reconstruction at 0 so u = m_hat/(0+eps) = 0, matching full-v, which
        # applies decay and no gradient step.
        return r_hat, c_hat, r_hat.mean().clamp_min(1e-30)

    @staticmethod
    def _reconstruct_block(r_hat, c_hat, g_mean, o0, o1):
        """v_hat[o0:o1, :] ~= R_hat[o0:o1, None] * C_hat[None, :] / G.

        Only the requested row block is materialized, so the full [R, C] v is
        never allocated.
        """
        return torch.outer(r_hat[o0:o1], c_hat) / g_mean

    def zero_grad(self, model=None):
        for p in self.p:
            p.grad = None
        if model is not None:
            for _, m in fp8_modules(model):
                m._gw = None
        else:
            import warnings
            warnings.warn("SmaulOpt.zero_grad called without model; FP8 _gw not cleared",
                          RuntimeWarning, stacklevel=2)

    @torch.no_grad()
    def _clip(self, mods, mx=1.0):
        # Identical semantics to Lion._clip: global norm in float64 (see
        # _grad_norm), non-finite grads -> inf (caller skips the step).
        grads = [m._gw for _, m in mods if m._gw is not None] + \
                [p.grad for p in self.p if p.grad is not None]
        if not grads:
            return 0.0
        t = _grad_norm(grads)
        if not math.isfinite(t):
            return float("inf")
        if t > mx:
            s = mx / (t + 1e-6)
            for _, m in mods:
                if m._gw is not None:
                    m._gw.mul_(s)
            for p in self.p:
                if p.grad is not None:
                    p.grad.mul_(s)
        return t

    @torch.no_grad()
    def step(self, model):
        mods = fp8_modules(model)
        norm = self._clip(mods, self.clip)
        if norm == float("inf"):
            # Non-finite grads would poison FP32 states and quantized weights.
            # Clear and skip; step counter does not advance so bias
            # correction stays aligned with actual updates.
            self.zero_grad(model)
            return norm
        if not any(m._gw is not None for _, m in mods) \
                and not any(p.grad is not None for p in self.p):
            # Same for the empty step: no gradients, no update, so the counter
            # must not advance and live momentum must not be evicted as "dead".
            return 0.0
        self.step_count += 1
        t = int(self.step_count)
        bc1 = 1.0 - self.beta_m ** t
        bc2 = 1.0 - self.beta_v ** t
        # beta in [0,1) and t>=1 guarantees bc>0; guard rounding anyway.
        if bc1 <= 0.0:
            bc1 = 1e-12
        if bc2 <= 0.0:
            bc2 = 1e-12
        live = set()
        decay = self.lr * self.weight_decay
        # FP8 weights: the accumulated _gw may be narrower than FP32 (see
        # kernel.fp8_tile.GW_DTYPE). It is read blockwise and NEVER widened whole
        # -- a full .float() would allocate an FP32 copy the size of the buffer it
        # replaced, cancelling the saving. Elementwise promotion inside the FP32
        # accumulators handles the widening per block instead.
        for _, mod in mods:
            if mod._gw is None:
                continue
            gw = mod._gw
            shape = tuple(gw.shape)
            live.add(mod)
            st_m = self.m.get(mod)
            if st_m is None or st_m.shape != shape:
                st_m = self._empty_state(shape, gw.device, signed=True)
                self.m[mod] = st_m
            elif st_m.device != gw.device:
                st_m = st_m.to(gw.device)
                self.m[mod] = st_m
            if self._factor_shape(shape):
                # ---- factored v: maintain the exact marginals of EMA(|g|) ----
                st_r, st_c = self._v_marginal_state(mod, shape, gw.device)
                r32 = st_r.float()
                c32 = st_c.float()
                r32.mul_(self.beta_v).add_(self._mean_abs(gw, 1), alpha=1.0 - self.beta_v)
                c32.mul_(self.beta_v).add_(self._mean_abs(gw, 0), alpha=1.0 - self.beta_v)
                r_hat, c_hat, g_mean = self._factored_hat(r32, c32, bc2)
                for _b, o0, o1 in self._row_blocks(st_m):
                    m_b = st_m[o0:o1] if self.state_dtype == "fp32" else st_m[o0:o1].float()
                    m_b.mul_(self.beta_m).add_(gw[o0:o1], alpha=1.0 - self.beta_m)
                    v_hat = self._reconstruct_block(r_hat, c_hat, g_mean, o0, o1)
                    u_b = (m_b / bc1) / (v_hat + self.epsilon)
                    # The weight write is gated on a finite u, but the m state is
                    # not: m is just an EMA of a (already finite-checked) gradient,
                    # so skipping it would leave m stale while the step counter
                    # advanced and bias correction desynchronized.
                    if bool(torch.isfinite(u_b).all()):
                        if self.state_dtype != "fp32":
                            u_b.clamp_(-self.update_clip, self.update_clip)
                        mod._requant_block(o0, o1, u_b * self.lr, decay)
                    if self.state_dtype != "fp32":
                        st_m[o0:o1].copy_(m_b)
                if self.state_dtype != "fp32":
                    st_r.copy_(r32)
                    st_c.copy_(c32)
                self.v.pop(mod, None)
            else:
                st_v = self.v.get(mod)
                if st_v is None or st_v.shape != shape:
                    st_v = self._empty_state(shape, gw.device, signed=False)
                    self.v[mod] = st_v
                elif st_v.device != gw.device:
                    st_v = st_v.to(gw.device)
                    self.v[mod] = st_v
                for _b, o0, o1 in self._row_blocks(st_m):
                    # ---- FP32 state arithmetic for this block ----
                    if self.state_dtype == "fp32":
                        m_b = st_m[o0:o1]
                        v_b = st_v[o0:o1]
                    else:
                        m_b = st_m[o0:o1].float()
                        v_b = st_v[o0:o1].float()
                    m_b.mul_(self.beta_m).add_(gw[o0:o1], alpha=1.0 - self.beta_m)
                    v_b.mul_(self.beta_v).add_(gw[o0:o1].abs(), alpha=1.0 - self.beta_v)
                    u_b = (m_b / bc1) / (v_b / bc2 + self.epsilon)
                    # Gate only the weight write. A non-finite u must not be
                    # folded into quantized storage, but the m/v EMAs still
                    # have to be persisted: m_b/v_b are temporaries in the
                    # narrow-state case, so `continue`-ing past the copy_()
                    # below would silently discard them while step_count has
                    # already advanced, desynchronizing bias correction. (The
                    # factored branch above has always done it this way.)
                    finite = bool(torch.isfinite(u_b).all())
                    if finite:
                        if self.state_dtype != "fp32":
                            # Bound |u|: the invariant |EMA(g)| <= EMA(|g|) should
                            # keep it near 1, so this only engages on state noise.
                            u_b.clamp_(-self.update_clip, self.update_clip)
                        mod._requant_block(o0, o1, u_b * self.lr, decay)
                    # ---- narrow the state block back to storage width ----
                    if self.state_dtype != "fp32":
                        st_m[o0:o1].copy_(m_b)
                        st_v[o0:o1].copy_(v_b)
                self.v_row.pop(mod, None)
                self.v_col.pop(mod, None)
            mod._gw = None
        for p in self.p:
            if p.grad is None:
                continue
            # p.grad may be narrower than FP32 (see narrow_grads_). Use it as-is:
            # the FP32 state accumulators promote per block, so no full-size FP32
            # copy of the gradient is ever built.
            g = p.grad
            st_m = self.m.get(p)
            if st_m is None or st_m.shape != tuple(p.shape):
                st_m = self._empty_state(tuple(p.shape), g.device, signed=True)
                self.m[p] = st_m
            elif st_m.device != g.device:
                st_m = st_m.to(g.device)
                self.m[p] = st_m
            live.add(p)
            if self._factor_shape(tuple(p.shape)):
                # ---- factored v for a 2-D dense parameter ----
                st_r, st_c = self._v_marginal_state(p, p.shape, g.device)
                r32 = st_r.float()
                c32 = st_c.float()
                r32.mul_(self.beta_v).add_(self._mean_abs(g, 1), alpha=1.0 - self.beta_v)
                c32.mul_(self.beta_v).add_(self._mean_abs(g, 0), alpha=1.0 - self.beta_v)
                r_hat, c_hat, g_mean = self._factored_hat(r32, c32, bc2)
                for _b, o0, o1 in self._row_blocks(st_m, self._DENSE_OB):
                    m_b = st_m[o0:o1] if self.state_dtype == "fp32" else st_m[o0:o1].float()
                    m_b.mul_(self.beta_m).add_(g[o0:o1], alpha=1.0 - self.beta_m)
                    v_hat = self._reconstruct_block(r_hat, c_hat, g_mean, o0, o1)
                    u = (m_b / bc1) / (v_hat + self.epsilon)
                    # Gate only the weight write; m always advances (see the FP8
                    # path above for why).
                    if bool(torch.isfinite(u).all()):
                        if self.state_dtype != "fp32":
                            u.clamp_(-self.update_clip, self.update_clip)
                        if self.weight_decay:
                            p[o0:o1].mul_(1.0 - self.lr * self.weight_decay)
                        p[o0:o1].add_(u.to(p.dtype), alpha=-self.lr)
                    if self.state_dtype != "fp32":
                        st_m[o0:o1].copy_(m_b)
                if self.state_dtype != "fp32":
                    st_r.copy_(r32)
                    st_c.copy_(c32)
                self.v.pop(p, None)
            else:
                st_v = self.v.get(p)
                if st_v is None or st_v.shape != tuple(p.shape):
                    st_v = self._empty_state(tuple(p.shape), g.device, signed=False)
                    self.v[p] = st_v
                elif st_v.device != g.device:
                    st_v = st_v.to(g.device)
                    self.v[p] = st_v
                # ---- FP32 state arithmetic for the whole (dense) parameter ----
                if self.state_dtype == "fp32":
                    m_b = st_m
                    v_b = st_v
                else:
                    m_b = st_m.float()
                    v_b = st_v.float()
                m_b.mul_(self.beta_m).add_(g, alpha=1.0 - self.beta_m)
                v_b.mul_(self.beta_v).add_(g.abs(), alpha=1.0 - self.beta_v)
                m_hat = m_b / bc1
                v_hat = v_b / bc2
                u = m_hat / (v_hat + self.epsilon)
                # As in the FP8 block path: gate only the parameter update, and
                # always persist m/v, or a narrow-state tensor would lose its
                # EMA for this step while step_count still advanced.
                finite = bool(torch.isfinite(u).all())
                if finite:
                    if self.state_dtype != "fp32":
                        # Bound |u|: see the note in the FP8 block path above.
                        u.clamp_(-self.update_clip, self.update_clip)
                    if self.weight_decay:
                        # Decoupled: theta <- theta * (1 - lr*wd), same as
                        # theta - lr*wd*theta.
                        p.mul_(1.0 - self.lr * self.weight_decay)
                    if u.dtype != p.dtype:
                        u = u.to(p.dtype)
                    # Ensure device match (states migrate with grads; param is source).
                    if u.device != p.device:
                        u = u.to(p.device)
                    p.add_(u, alpha=-self.lr)
                # ---- narrow the state back to storage width ----
                if self.state_dtype != "fp32":
                    st_m.copy_(m_b)
                    st_v.copy_(v_b)
                self.v_row.pop(p, None)
                self.v_col.pop(p, None)
        for store in (self.m, self.v, self.v_row, self.v_col):
            for k in list(store):
                if k not in live:
                    del store[k]
        return norm

    def state_dict(self):
        return {
            "name": "smaul",
            "step": int(self.step_count),
            "lr": self.lr,
            "learning_rate": self.lr,
            "beta_m": self.beta_m,
            "beta_v": self.beta_v,
            "betas": [self.beta_m, self.beta_v],
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "wd": self.weight_decay,
            "clip": self.clip,
            "state_dtype": self.state_dtype,
            "factor_v": bool(self.factor_v),
            # Both change the update: update_clip bounds |u|, and grad_dtype
            # decides what precision the clipped gradients are stored at. A
            # resume that silently reverted them would train a different
            # optimizer than the one that wrote the checkpoint.
            "update_clip": self.update_clip,
            "grad_dtype": self.grad_dtype,
        }

    def load_state_dict(self, d):
        if not isinstance(d, dict):
            raise ValueError(f"SmaulOpt checkpoint must be a dict, got {type(d).__name__}")
        name = d.get("name", None)
        if name != "smaul":
            if name is None:
                raise ValueError(
                    "checkpoint has no optimizer name; refusing to load non-SmaulOpt "
                    "(e.g. Lion/AdamW) state into SmaulOpt because m/v are missing")
            raise ValueError(
                f"cannot load optimizer {name!r} state into SmaulOpt (expected 'smaul'); "
                "m/v states are not interchangeable")
        import math as _math
        lr = d.get("lr", d.get("learning_rate", self.lr))
        bm = d.get("beta_m", None)
        bv = d.get("beta_v", None)
        if bm is None or bv is None:
            betas = d.get("betas", [self.beta_m, self.beta_v])
            try:
                bm = float(betas[0]) if bm is None else float(bm)
                bv = float(betas[1]) if bv is None else float(bv)
            except (TypeError, IndexError, ValueError):
                raise ValueError(f"invalid betas in checkpoint: {betas!r}") from None
        eps = d.get("epsilon", self.epsilon)
        wd = d.get("weight_decay", d.get("wd", self.weight_decay))
        clip = d.get("clip", self.clip)
        step = d.get("step", self.step_count)
        try:
            _lr, _bm, _bv = float(lr), float(bm), float(bv)
            _eps, _wd, _clip = float(eps), float(wd), float(clip)
            _step = int(step)
        except (TypeError, ValueError):
            raise ValueError(f"invalid SmaulOpt checkpoint hyperparams: {d!r}") from None
        if not _math.isfinite(_lr) or _lr <= 0:
            raise ValueError(f"checkpoint lr invalid: {lr!r}")
        if not _math.isfinite(_bm) or not 0.0 <= _bm < 1.0:
            raise ValueError(f"checkpoint beta_m invalid: {bm!r}")
        if not _math.isfinite(_bv) or not 0.0 <= _bv < 1.0:
            raise ValueError(f"checkpoint beta_v invalid: {bv!r}")
        if not _math.isfinite(_eps) or _eps <= 0:
            raise ValueError(f"checkpoint epsilon invalid: {eps!r}")
        if not _math.isfinite(_wd) or _wd < 0:
            raise ValueError(f"checkpoint weight_decay invalid: {wd!r}")
        if not _math.isfinite(_clip) or _clip <= 0:
            raise ValueError(f"checkpoint clip invalid: {clip!r}")
        if _step < 0:
            raise ValueError(f"checkpoint step invalid: {step!r}")
        sdt = d.get("state_dtype", "bf16")
        if sdt not in SmaulOpt._STATE_DTYPES:
            raise ValueError(
                f"checkpoint state_dtype must be one of {SmaulOpt._STATE_DTYPES}, got {sdt!r}")
        fv = d.get("factor_v", None)
        if fv is not None and not isinstance(fv, bool):
            raise ValueError(f"checkpoint factor_v must be a bool, got {fv!r}")
        uclip = d.get("update_clip", self.update_clip)
        try:
            _uclip = float(uclip)
        except (TypeError, ValueError):
            raise ValueError(f"checkpoint update_clip invalid: {uclip!r}") from None
        if not _math.isfinite(_uclip) or _uclip <= 0:
            raise ValueError(f"checkpoint update_clip invalid: {uclip!r}")
        gdt = d.get("grad_dtype", self.grad_dtype)
        if gdt not in (None, "bf16", "fp16", "fp32"):
            raise ValueError(f"checkpoint grad_dtype invalid: {gdt!r}")
        if gdt == "fp32":
            # Same spelling as the constructor: fp32 means "do not narrow".
            gdt = None
        self.lr, self.beta_m, self.beta_v = _lr, _bm, _bv
        self.epsilon, self.weight_decay, self.clip = _eps, _wd, _clip
        self.update_clip, self.grad_dtype = _uclip, gdt
        self.step_count = _step
        # State storage width follows the checkpoint; stored m/v buffers are
        # recast by the state loader, so a resumed run keeps the saved width.
        self.state_dtype = sdt
        # Checkpoints written before factoring existed have no factor_v key; they
        # are always full-v, so default the flag to False for them.
        self.checkpoint_factor_v = bool(fv) if fv is not None else False


def _save_optimizer(out: Path, opt, model=None) -> None:
    # JSON, not pickle: torch.save would be arbitrary-code-exec on load.
    (out / "optimizer.json").write_text(json.dumps(opt.state_dict(), indent=2), encoding="utf-8")
    if isinstance(opt, SmaulOpt) and model is not None:
        _save_smaul_states(out, opt, model)


def _save_smaul_states(out: Path, opt: "SmaulOpt", model) -> None:
    from safetensors.torch import save_file
    param_names = {id(p): n for n, p in model.named_parameters()}
    mod_names = {id(m): n for n, m in fp8_modules(model)}
    tensors = {}

    def _base(key_obj):
        if id(key_obj) in param_names:
            return f"param.{param_names[id(key_obj)]}"
        if id(key_obj) in mod_names:
            return f"fp8.{mod_names[id(key_obj)]}"
        return None

    for key_obj, m_state in opt.m.items():
        base = _base(key_obj)
        if base is None:
            continue
        # Stored at the checkpoint's own width (bf16/fp16/fp32) so the file size
        # reflects the real state footprint.
        tensors[f"m.{base}"] = m_state.detach().cpu().contiguous()
        st_r = opt.v_row.get(key_obj)
        st_c = opt.v_col.get(key_obj)
        if st_r is not None and st_c is not None:
            # Factored v: the two marginal vectors, never the full [R, C].
            tensors[f"v_row.{base}"] = st_r.detach().cpu().contiguous()
            tensors[f"v_col.{base}"] = st_c.detach().cpu().contiguous()
        else:
            v_state = opt.v.get(key_obj)
            if v_state is not None:
                tensors[f"v.{base}"] = v_state.detach().cpu().contiguous()
    if tensors:
        tmp = out / "optimizer_state.safetensors.tmp"
        save_file(tensors, str(tmp))
        os.replace(tmp, out / "optimizer_state.safetensors")


def _load_optimizer(out: Path, opt, model=None):
    """Load optimizer.json (and SmaulOpt states) into opt.

    Raises ValueError with a clear message when a checkpoint created by
    another optimizer cannot provide the required state.
    """
    out = Path(out)
    data = json.loads((out / "optimizer.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{out / 'optimizer.json'} must contain a JSON object")
    opt.load_state_dict(data)
    if isinstance(opt, SmaulOpt) and model is not None:
        _load_smaul_states(out, opt, model)
    return opt


def _load_smaul_states(out: Path, opt: "SmaulOpt", model) -> None:
    from safetensors.torch import load_file
    out = Path(out)
    sp = out / "optimizer_state.safetensors"
    param_by_name = dict(model.named_parameters())
    mod_by_name = dict(fp8_modules(model))
    if not sp.exists():
        if opt.step_count and (opt.m or opt.v or opt.v_row or opt.v_col):
            raise ValueError(f"SmaulOpt checkpoint {out} is missing optimizer_state.safetensors")
        if int(opt.state_dict().get("step", opt.step_count)) > 0:
            raise ValueError(
                f"SmaulOpt checkpoint {out} records step={opt.step_count} but has no "
                "optimizer_state.safetensors; refusing to pretend state exists")
        return
    try:
        blobs = load_file(str(sp), device="cpu")
    except RuntimeError as exc:
        raise RuntimeError(f"could not load SmaulOpt states {sp}: {exc}") from exc
    new_m: dict = {}
    new_v: dict = {}
    new_vr: dict = {}
    new_vc: dict = {}
    for k, tens in blobs.items():
        if not isinstance(k, str) or "." not in k:
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        kind, rest = k.split(".", 1)
        if kind not in ("m", "v", "v_row", "v_col"):
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        if rest.startswith("param."):
            obj = param_by_name.get(rest[len("param."):], None)
            if obj is None:
                raise ValueError(
                    f"SmaulOpt state key {k!r} has no matching parameter in current model")
        elif rest.startswith("fp8."):
            obj = mod_by_name.get(rest[len("fp8."):], None)
            if obj is None:
                raise ValueError(
                    f"SmaulOpt state key {k!r} has no matching FP8 module in current model")
        else:
            raise ValueError(f"invalid SmaulOpt state key {k!r}")
        target = {"m": new_m, "v": new_v, "v_row": new_vr, "v_col": new_vc}[kind]
        target[obj] = tens
    # A model may legitimately hold both forms: 2-D params are factored while
    # 1-D/0-D params keep a full v. What is never valid is one object having
    # both, or a half-written factored pair.
    for obj in new_m:
        has_full = obj in new_v
        r_ok, c_ok = obj in new_vr, obj in new_vc
        if r_ok != c_ok:
            raise ValueError(
                f"SmaulOpt factored state incomplete for one object "
                f"(v_row={r_ok}, v_col={c_ok}); refusing partial load")
        if has_full and r_ok:
            raise ValueError(
                "SmaulOpt state for one object has both full-v and factored-v keys; "
                "refusing to guess")
        if not has_full and not r_ok:
            raise ValueError(f"SmaulOpt state for one object has no v at all; refusing")
    for obj, tens in list(new_vr.items()):
        # Each marginal is 1-D and must line up with its own extent of m:
        # v_row is the mean over dim=1 so it is R long, v_col the mean over
        # dim=0 so it is C long. That is what catches a transposed pair, and
        # for a non-square state it catches it outright.
        #
        # The previous check here compared v_row's shape to v_col's *reversed*
        # shape, which cannot fire: both marginals are 1-D, so `shape[::-1]` is
        # the shape itself, and the two clauses below were mutually exclusive by
        # construction. It was unreachable for every state the optimizer writes,
        # while reading as a guard against exactly this mistake.
        tm = new_m.get(obj)
        if tm is None:
            continue
        if tm.dim() == 2:
            if tens.dim() != 1 or int(tens.shape[0]) != int(tm.shape[0]):
                raise ValueError(
                    f"SmaulOpt v_row has shape {tuple(tens.shape)}, expected ({int(tm.shape[0])},) "
                    f"for an m of {tuple(tm.shape)}; a transposed or mismatched pair")
            tc = new_vc[obj]
            if tc.dim() != 1 or int(tc.shape[0]) != int(tm.shape[1]):
                raise ValueError(
                    f"SmaulOpt v_col has shape {tuple(tc.shape)}, expected ({int(tm.shape[1])},) "
                    f"for an m of {tuple(tm.shape)}; a transposed or mismatched pair")
    # Shape check against live objects (fail clearly on arch change).
    for obj, tm in new_m.items():
        # FP8 modules store [out_f, in_f]; dense params store param shape.
        if hasattr(obj, "_gw"):
            expected = (int(obj.out_f), int(obj.in_f))
        else:
            try:
                expected = tuple(obj.shape)
            except AttributeError:
                expected = None
        if expected is not None and tuple(tm.shape) != expected:
            raise ValueError(
                f"SmaulOpt state shape {tuple(tm.shape)} != expected {expected}; "
                "checkpoint incompatible with current model")
    for obj, tv in new_v.items():
        tm = new_m.get(obj)
        if tm is not None and tuple(tv.shape) != tuple(tm.shape):
            raise ValueError("SmaulOpt m/v shape mismatch; checkpoint corrupt")
    # Cast the restored buffers to the checkpoint's declared storage width so
    # a resumed run keeps the same state footprint and math path.
    want = opt._storage_dtype(signed=True)
    opt.m = {k: v.to(want).contiguous() for k, v in new_m.items()}
    opt.v_row = {k: v.to(want).contiguous() for k, v in new_vr.items()}
    opt.v_col = {k: v.to(want).contiguous() for k, v in new_vc.items()}
    opt.v = {k: v.to(want).contiguous() for k, v in new_v.items()}
    # opt.factor_v keeps whatever the caller configured, so full-vs-factored can
    # be compared without touching source. Two mismatches need a decision:
    if opt.v_row and not opt.factor_v:
        raise ValueError(
            f"SmaulOpt checkpoint {out} holds factored v but factor_v is disabled; "
            "a factored state cannot be expanded into a full v without inventing the "
            "rank term. Resume with factor_v enabled, or start a new run.")
    migrated = 0
    if opt.factor_v and opt.v:
        for key_obj, full in list(opt.v.items()):
            shape = tuple(full.shape)
            if not opt._factor_shape(shape):
                continue
            # Explicit migration, full-v -> factored. The marginals of a stored v
            # are exactly recoverable, so R and C are preserved exactly; only the
            # rank term is dropped, which is what factoring approximates anyway.
            f32 = full.to(want).float()
            opt.v_row[key_obj] = f32.mean(dim=1).to(want).contiguous()
            opt.v_col[key_obj] = f32.mean(dim=0).to(want).contiguous()
            del opt.v[key_obj]
            migrated += 1
        if migrated:
            print(f"[smaul] migrated {migrated} full-v tensor(s) to factored v on load; "
                  "row/col marginals preserved exactly, rank term dropped. "
                  "The next save writes the factored form.")


def _validate_args(args) -> None:
    for name in ("batch", "ctx", "steps", "d", "layers", "heads", "vocab", "threads", "log_every",
                 "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive, got {getattr(args, name)}")
    if args.d % args.heads != 0:
        raise ValueError(f"--d ({args.d}) must be divisible by --heads ({args.heads})")
    if not 0 < args.lr < 10:
        raise ValueError(f"--lr looks invalid: {args.lr}")
    if not 0 <= args.wd < 10:
        raise ValueError(f"--wd looks invalid: {args.wd}")
    if getattr(args, "optimizer", "smaul") not in ("lion", "smaul"):
        raise ValueError(f"--optimizer must be lion/smaul, got {getattr(args, 'optimizer')!r}")
    for _n, _flag in (("beta_m", "--beta-m"), ("beta_v", "--beta-v")):
        _v = getattr(args, _n, None)
        if _v is None:
            continue
        if not isinstance(_v, (int, float)) or not math.isfinite(float(_v)):
            raise ValueError(f"{_flag} must be finite, got {_v!r}")
        if not 0.0 <= float(_v) < 1.0:
            raise ValueError(f"{_flag} must be in [0.0, 1.0), got {_v!r}")
    _eps = getattr(args, "epsilon", None)
    if _eps is not None:
        if not isinstance(_eps, (int, float)) or not math.isfinite(float(_eps)) \
                or float(_eps) <= 0:
            raise ValueError(f"--epsilon must be positive finite, got {_eps!r}")
    ff = getattr(args, "ffn_mult", 2.5)
    if not isinstance(ff, (int, float)) or not math.isfinite(ff) or ff <= 0:
        raise ValueError(f"--ffn_mult must be positive finite, got {ff!r}")

def _tok(args, out: Path):
    from tokenizer import VERSION as _TOK_VERSION

    tp = Path(args.tokenizer) if args.tokenizer else out / "tokenizer.json"
    if tp.exists():
        try:
            from tokenizer import load as _load
            t = _load(tp)
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"could not load tokenizer {tp}: {exc}") from exc
        if t.get_vocab_size() == args.vocab and t.data.get("version") == _TOK_VERSION:
            return t, tp
        print("[tok] vocab/version mismatch, rebuilding")
    else:
        print(f"[tok] {tp} not found, training new tokenizer")
    data_dir = Path(args.data)
    if not data_dir.exists():
        raise ValueError(f"--data {data_dir} does not exist")
    files = discover_files(data_dir)
    if not files:
        raise RuntimeError(f"no training files found in {data_dir}")
    texts = (t for t, _, _ in iter_texts(files))
    max_records = max(0, int(getattr(args, "tok_records", 0) or 0))
    return ensure_tokenizer(tp, texts, args.vocab, max_records=max_records), tp


def _sha_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_fingerprint(files) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in sorted(str(p) for p in files):
        try:
            st = Path(p).stat()
            h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
        except OSError:
            h.update(p.encode())
    return h.hexdigest()[:16]

def main():
    install_handlers()
    a = argparse.ArgumentParser()
    a.add_argument("--data", default="./datasets")
    a.add_argument("--out", default="./runs/linear")
    a.add_argument("--tokenizer", default=None,
                   help="Tokenizer path (default: <out>/tokenizer.json)")
    a.add_argument("--preset", default=None,
                   help="Named size preset (overrides --vocab/--d/--layers/--heads/--ffn_mult); see --list-presets")
    a.add_argument("--list-presets", action="store_true", help="List size presets with estimated params and exit")
    a.add_argument("--vocab", type=int, default=8000)
    a.add_argument("--d", type=int, default=512)
    a.add_argument("--layers", type=int, default=8)
    a.add_argument("--heads", type=int, default=8)
    a.add_argument("--ffn_mult", type=float, default=2.5)
    a.add_argument("--architecture", choices=("rawr", "plain"), default="rawr",
                   help="Model architecture: Rawr sparse (default) or plain dense baseline")
    a.add_argument("--embedding-storage", choices=("ram", "mmap"), default="ram",
                   help="Embedding table storage (default ram)")
    a.add_argument("--rawr-sparsity", type=float, default=0.9,
                   help="Rawr: fraction of hidden/head connections omitted [0, 1). "
                        "This sets K = d_model * (1 - sparsity) columns kept per "
                        "SparseLinear row, so it is really a column-count knob: at the "
                        "old 0.5 default (K = d/2) the [B,T,out_f,K] gather moved 7.7 GiB "
                        "per forward vs 15.3 GiB dense -- half the compute cut while "
                        "still paying every rawr overhead. 0.9 (K ~ d/20) is 5x less "
                        "traffic. Note LinearConfig keeps 0.5 as its code-level default "
                        "so library callers and legacy checkpoints are unaffected.")
    a.add_argument("--rawr-min-degree", type=int, default=4,
                   help="Rawr: fallback connectivity floor per token (>= 1)")
    a.add_argument("--rawr-dict", default=None,
                   help="Rawr: extra dictionary file (one word per line) on top of built-ins")
    a.add_argument("--rawr-graph-out", default=None,
                   help="Rawr: also export the connectivity graph JSON here")
    a.add_argument("--rawr-max-docs", type=int, default=2000,
                   help="Rawr: max corpus docs sampled for graph edges (0 = unlimited)")
    a.add_argument("--rawr-max-tokens-per-doc", type=int, default=1024)
    a.add_argument("--precision", choices=("fp8", "fp32"), default="fp8")
    a.add_argument("--ctx", type=int, default=256)
    a.add_argument("--batch", type=int, default=2)
    a.add_argument("--steps", type=int, default=1000)
    a.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate (--lr is learning_rate; --wd is weight_decay)")
    a.add_argument("--wd", type=float, default=0.01)
    a.add_argument("--optimizer", choices=("lion", "smaul"), default="smaul",
                   help="Optimizer: smaul (default, SmaulOpt v2.1) or lion "
                        "(resume-free, fixed betas)")
    a.add_argument("--beta-m", dest="beta_m", type=float, default=0.9,
                   help="SmaulOpt beta_m (momentum decay); Lion keeps its built-in betas")
    a.add_argument("--beta-v", dest="beta_v", type=float, default=0.999,
                   help="SmaulOpt beta_v (magnitude-EMA decay)")
    a.add_argument("--epsilon", type=float, default=1e-8,
                   help="SmaulOpt epsilon (positive)")
    a.add_argument("--state-dtype", dest="state_dtype", choices=SmaulOpt._STATE_DTYPES,
                   default="bf16",
                   help="SmaulOpt state storage width; update math is always FP32. "
                        "bf16 (2B, default: ~0.07%% error vs fp32) | fp16 (2B) | "
                        "fp32 (4B, lossless reference)")
    a.add_argument("--grad-dtype", dest="grad_dtype", choices=("bf16", "fp16", "fp32"),
                   default="bf16",
                   help="SmaulOpt: dtype gradients are stored in after backward. "
                        "bf16 (default) halves gradient memory; fp32 keeps them "
                        "wide. Update math is FP32 either way. Lion ignores this.")
    a.add_argument("--factor-v", dest="factor_v", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="SmaulOpt: store v factored (row/col marginals) for 2-D "
                        "parameters instead of full-size. Default on; --no-factor-v "
                        "gives the full-v comparison mode.")
    a.add_argument("--grad_clip", type=float, default=1.0)
    a.add_argument("--log_every", type=int, default=10)
    a.add_argument("--save_every", type=int, default=200)
    a.add_argument("--tok_records", type=int, default=200000,
                   help="Max records for automatic tokenizer training (0 = unlimited)")
    a.add_argument("--threads", type=int, default=2)
    args = a.parse_args()
    if args.list_presets:
        for _name in sorted(PRESETS):
            _p = PRESETS[_name]
            _est = estimate_params(_p["vocab"], _p["d"], _p["layers"], _p.get("ffn_mult", 2.5))
            print(f"{_name}: vocab={_p['vocab']} d={_p['d']} layers={_p['layers']} "
                  f"heads={_p['heads']} ffn_mult={_p.get('ffn_mult', 2.5)} ~{_est:,} params")
        return
    apply_preset(args)
    _validate_args(args)
    if args.tok_records < 0:
        raise ValueError(f"--tok_records must be non-negative, got {args.tok_records}")
    if args.grad_clip <= 0:
        raise ValueError(f"--grad_clip must be positive, got {args.grad_clip}")
    # Configure threads through the backend (sets OMP/MKL before torch init
    # where possible) instead of duplicating logic here.
    try:
        from kernel.compute import get_backend
        get_backend().configure(args.threads)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"invalid --threads {args.threads}: {exc}") from exc
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok, tok_path = _tok(args, out)
    # The tokenizer file is written only after the first successful step (see
    # the training loop): writing it here would clobber a good checkpoint's
    # tokenizer on a run that completes zero steps, while the step == 0 path
    # below reports the checkpoint was not overwritten. The sha uses the exact
    # serialization save() writes, so the recorded fingerprint still matches
    # the file once written.
    import hashlib as _hashlib
    tok_sha = _hashlib.sha256(json.dumps(
        tok.data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    try:
        ds_fp = _dataset_fingerprint(discover_files(Path(args.data)))
    except (OSError, ValueError, RuntimeError):
        ds_fp = ""
    arch = getattr(args, "architecture", "rawr") or "rawr"
    storage = getattr(args, "embedding_storage", "ram") or "ram"
    if arch not in ("rawr", "plain"):
        raise ValueError(f"--architecture must be rawr/plain, got {arch!r}")
    if storage not in ("ram", "mmap"):
        raise ValueError(f"--embedding-storage must be ram/mmap, got {storage!r}")
    rawr_graph = None
    if arch == "rawr":
        from rawr_graph import build_graph, print_stats, save_graph

        extra_words = None
        if getattr(args, "rawr_dict", None):
            extra_words = [ln.strip() for ln in Path(args.rawr_dict).read_text(
                encoding="utf-8-sig", errors="replace").splitlines() if ln.strip()]
        data_files = discover_files(Path(args.data))
        max_docs = int(getattr(args, "rawr_max_docs", 2000) or 0)
        max_tpd = int(getattr(args, "rawr_max_tokens_per_doc", 1024) or 0)
        corpus = (t for t, _, _ in iter_texts(data_files))
        rawr_graph = build_graph(tok, corpus_texts=corpus, dict_words=extra_words,
                                 window=1, min_degree=int(args.rawr_min_degree),
                                 max_docs=max_docs, max_tokens_per_doc=max_tpd or 4096)
        print(f"[rawr] graph digest={rawr_graph.digest} "
              f"edges={len(rawr_graph.edges)} min_deg={args.rawr_min_degree}")
        print_stats(rawr_graph)
        if getattr(args, "rawr_graph_out", None):
            save_graph(rawr_graph, Path(args.rawr_graph_out))
    cfg = LinearConfig(vocab_size=args.vocab, d_model=args.d, n_layer=args.layers, n_heads=args.heads,
                       ffn_mult=getattr(args, "ffn_mult", 2.5),
                       precision=args.precision, tokenizer_sha256=tok_sha, dataset_fingerprint=ds_fp,
                       architecture=arch, embedding_storage=storage,
                       rawr_sparsity=float(getattr(args, "rawr_sparsity", 0.9)),
                       rawr_min_degree=int(getattr(args, "rawr_min_degree", 4)))
    emb_path = (out / "embeddings.dat") if storage == "mmap" else None
    model = SmaulLinear(cfg, rawr_graph=rawr_graph, emb_path=emb_path)
    if arch == "rawr":
        # The graph stats printed above describe the token graph, not the
        # model. Print what the built model actually executes, so the headline
        # sparsity figure is not read as the model's. See
        # rawr_graph.print_model_compute.
        from rawr_graph import print_model_compute

        print_model_compute(model.compute_profile())
    opt_name = getattr(args, "optimizer", "lion") or "lion"
    if opt_name == "smaul":
        opt = SmaulOpt(list(model.parameters()), lr=args.lr,
                       beta_m=getattr(args, "beta_m", 0.9),
                       beta_v=getattr(args, "beta_v", 0.999),
                       epsilon=getattr(args, "epsilon", 1e-8),
                       weight_decay=args.wd, clip=args.grad_clip,
                       state_dtype=getattr(args, "state_dtype", "bf16"),
                       factor_v=getattr(args, "factor_v", True),
                       grad_dtype=getattr(args, "grad_dtype", "bf16"))
    else:
        opt = Lion(list(model.parameters()), lr=args.lr, wd=args.wd, clip=args.grad_clip)
    wrap = TokenizerWrapper(tok)
    stream = PretrainStream(Path(args.data), wrap, args.ctx)
    model.train()
    bx, by, step, toks, t0, since = [], [], 0, 0, time.perf_counter(), 0
    bad_steps = 0
    for x, y, _ in stream:
        if STOP or step >= args.steps:
            break
        bx.append(x)
        by.append(y)
        if len(bx) < args.batch:
            continue
        xb, yb = torch.stack(bx), torch.stack(by)
        bx, by = [], []
        opt.zero_grad(model)
        _, loss = model(xb, yb)
        if not torch.isfinite(loss):
            bad_steps += 1
            print(f"[warn] non-finite loss, skip step {step} ({bad_steps} consecutive)")
            opt.zero_grad(model)
            if bad_steps >= 50:
                print("[error] 50 consecutive non-finite losses; stopping to avoid infinite loop")
                break
            continue
        loss.backward()
        # Release the FP32 gradient buffers before the step; the update math is
        # still FP32 (it widens per block). No-op for Lion.
        if hasattr(opt, "narrow_grads_"):
            opt.narrow_grads_(model)
        # Sampled here, not at log time: opt.step() clears every FP8 _gw, so
        # afterwards the live-gradient figure would only see p.grad and
        # understate the peak by the largest single allocation in the model.
        grad_bytes = (sum(p.grad.numel() * p.grad.element_size()
                          for p in model.parameters() if p.grad is not None)
                      + sum(m._gw.numel() * m._gw.element_size()
                            for _, m in fp8_modules(model) if m._gw is not None))
        norm = opt.step(model)
        if norm == float("inf"):
            bad_steps += 1
            print(f"[warn] non-finite grads, skip step {step} ({bad_steps} consecutive)")
            if bad_steps >= 50:
                print("[error] 50 consecutive non-finite grads; stopping")
                break
            continue
        bad_steps = 0
        step += 1
        if step == 1:
            tok.save(str(tok_path))
        toks += xb.numel()
        since += xb.numel()
        if step % args.log_every == 0:
            el = time.perf_counter() - t0
            mem = sum(p.numel() * p.element_size() for p in model.parameters())
            mem += sum(b.numel() * b.element_size() for b in model.buffers())
            # Include optimizer state: otherwise the metric understates OOM risk.
            mem += sum(v.numel() * v.element_size() for v in opt.m.values())
            for _store in ("v", "v_row", "v_col"):
                # v is the full-size form; v_row/v_col are the factored
                # marginals, and with --factor-v (the default) v is empty, so
                # counting only v reported almost none of the state.
                _s = getattr(opt, _store, None)
                if _s:
                    mem += sum(v.numel() * v.element_size() for v in _s.values())
            print(f"step {step} loss {loss.item():.4f} {since / max(el, 1e-9):.1f} tok/s "
                  f"stored {mem / 1048576:.1f}MiB (+{grad_bytes / 1048576:.1f}MiB live grads)")
            t0, since = time.perf_counter(), 0
        if step % args.save_every == 0:
            model.save_pretrained(out)
            _save_optimizer(out, opt, model)
            print(f"[save] {out}")
    if STOP and (bx or by):
        print(f"[stop] dropped {len(bx)} buffered sample(s) from partial batch")
    if step == 0:
        print("[done] no steps completed; checkpoint not overwritten")
        return
    model.save_pretrained(out)
    _save_optimizer(out, opt, model)
    print(f"[done] steps={step} tokens={toks}")

if __name__ == "__main__":
    main()

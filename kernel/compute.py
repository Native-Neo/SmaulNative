#!/usr/bin/env python3
"""Minimal compute-backend boundary for SmaulNative.

The model (smaul_linear.py) and FP8 autograd (fp8_tile.py) call into a
backend selected with get_backend(); they never touch extensions directly.
The CPU backend is default, fully functional, and independently testable.
Future backends (e.g. AMD FP16/ROCm) register via register_backend()
without changing model code. No fake backends: unknown names raise.
"""
import os
import platform
import threading
import warnings
from pathlib import Path

import torch

# Cached reference fallbacks. Resolved lazily so the smaul_linear <-> compute
# import cycle is only paid once, and only if the native extensions are absent.
_attn_ref_fn = None
_attn_ref_bwd_fn = None


def _get_attn_ref():
    global _attn_ref_fn
    if _attn_ref_fn is None:
        from smaul_linear import _attn_reference
        _attn_ref_fn = _attn_reference
    return _attn_ref_fn


def _get_attn_ref_bwd():
    global _attn_ref_bwd_fn
    if _attn_ref_bwd_fn is None:
        from smaul_linear import _attn_reference_backward
        _attn_ref_bwd_fn = _attn_reference_backward
    return _attn_ref_bwd_fn


def _native_cflags() -> list:
    # Ivy Bridge AVX1 flags only on x86_64; elsewhere use portable -O3 so ARM
    # / non-AVX builds fall back cleanly instead of failing silently.
    base = ["-O3", "-ffp-contract=off"]
    if platform.machine().lower() in ("x86_64", "amd64", "x64"):
        base += ["-mavx", "-mf16c", "-msse4.2", "-mno-avx2", "-mno-avx512f"]
    return base

_BACKENDS = {}


def register_backend(backend):
    _BACKENDS[backend.name] = backend
    return backend


def get_backend(name=None):
    name = name or os.environ.get("SMAUL_BACKEND", "cpu")
    try:
        return _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown compute backend {name!r}; registered: {sorted(_BACKENDS)}") from None


class CpuBackend:
    """Ivy Bridge-safe CPU: native AVX1 E4M3 kernels, torch tiled fallback."""

    name = "cpu"

    def __init__(self):
        self._ext = None
        self._attn = None
        self._sparse = None
        self._quant = None
        self._lock = threading.Lock()
        self._warned_fallback = set()

    def configure(self, threads=None):
        if threads is None:
            env = os.environ.get("SMAUL_CPU_THREADS")
            if env:
                try:
                    threads = int(float(env))
                except (TypeError, ValueError):
                    raise ValueError(f"SMAUL_CPU_THREADS must be an integer, got {env!r}")
            else:
                threads = max(1, (os.cpu_count() or 2) // 2)
        try:
            threads = int(threads)
        except (TypeError, ValueError):
            raise ValueError(f"threads must be an integer, got {threads!r}")
        threads = max(1, threads)
        # Explicit assignment (not setdefault): an explicit --threads / configure()
        # call must win over a stale exported OMP/MKL value, otherwise torch
        # runs at `threads` while native OpenMP runs at the old value.
        # Call configure() before the first backend load for this to affect OpenMP.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        return threads

    @property
    def has_native(self):
        return self._load() is not None

    def _load(self):
        if self._ext is not None:
            return None if self._ext is False else self._ext
        with self._lock:
            if self._ext is not None:
                return None if self._ext is False else self._ext
            try:
                from torch.utils.cpp_extension import load
                root = Path(__file__).resolve().parent
                self._ext = load(name="smaul_fp8_ivb", sources=[str(root / "fp8_cpu.cpp")],
                    extra_cflags=_native_cflags(), verbose=False)
            except Exception as exc:
                self._ext = False
                warnings.warn(f"FP8 native ext unavailable; torch tiled fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._ext is False else self._ext

    def fp8_forward(self, x, w, s, in_f, out_f, tile):
        if tile <= 0 or in_f <= 0 or out_f <= 0:
            raise ValueError(f"tile/in_f/out_f must be positive, got {tile}/{in_f}/{out_f}")
        e = self._load()
        if e is not None and x.device.type == "cpu":
            xc = x if x.is_contiguous() else x.contiguous()
            wc = w.reshape(-1).contiguous()
            sc = s.reshape(-1).contiguous()
            return e.fp8_forward(xc, wc, sc, in_f, out_f, tile)
        self._warn_fallback_once("fp8_forward torch fallback (native missing or non-CPU input)")
        return _torch_forward(x, w, s, in_f, out_f, tile)

    def _warn_fallback_once(self, msg: str) -> None:
        # Keyed per message, not one flag for all three fallbacks: FP8 and
        # sparse have very different performance profiles, so suppressing two
        # of the three diagnostics hid which path was slow.
        if msg not in self._warned_fallback:
            self._warned_fallback.add(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def fp8_backward_input(self, g, w, s, in_f, out_f, tile):
        if tile <= 0 or in_f <= 0 or out_f <= 0:
            raise ValueError(f"tile/in_f/out_f must be positive, got {tile}/{in_f}/{out_f}")
        e = self._load()
        if e is not None and g.device.type == "cpu":
            gc = g if g.is_contiguous() else g.contiguous()
            return e.fp8_backward_input(gc, w.reshape(-1).contiguous(), s.reshape(-1).contiguous(), in_f, out_f, tile)
        self._warn_fallback_once("fp8_backward torch fallback (native missing or non-CPU input)")
        return _torch_backward_input(g, w, s, in_f, out_f, tile)

    def _load_attn(self):
        if self._attn is not None:
            return None if self._attn is False else self._attn
        with self._lock:
            if self._attn is not None:
                return None if self._attn is False else self._attn
            try:
                from torch.utils.cpp_extension import load
                root = Path(__file__).resolve().parent
                self._attn = load(name="smaul_attn", sources=[str(root / "attn_cpu.cpp")],
                    extra_cflags=_native_cflags(), verbose=False)
            except Exception as exc:
                self._attn = False
                warnings.warn(f"linear-attention native ext unavailable; python reference fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._attn is False else self._attn

    @property
    def has_attn_native(self):
        return self._load_attn() is not None

    def _load_sparse(self):
        if self._sparse is not None:
            return None if self._sparse is False else self._sparse
        with self._lock:
            if self._sparse is not None:
                return None if self._sparse is False else self._sparse
            try:
                from torch.utils.cpp_extension import load
                root = Path(__file__).resolve().parent
                self._sparse = load(name="smaul_sparse",
                    sources=[str(root / "sparse_cpu.cpp")],
                    extra_cflags=_native_cflags(), verbose=False)
            except Exception as exc:
                self._sparse = False
                warnings.warn(f"sparse native ext unavailable; torch chunked fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._sparse is False else self._sparse

    @property
    def has_sparse_native(self):
        return self._load_sparse() is not None

    def sparse_grad_v(self, dout, x, cols, values, budget=1 << 22):
        """grad_v[o, m] = sum_r dout[r, o] * x[r, cols[o, m]] (into a new tensor).

        ``dout`` is [rows, out_f] and ``x`` is [rows, in_f], both float32; the
        kernel wants them transposed to [out_f, rows] / [in_f, rows] so the
        inner product is contiguous. The torch fallback gathers instead, which
        materialises rows*out_f*K elements (836 MiB for an 8000x512 head at
        rows=512, K=51) and is ~100x slower; ``budget`` caps that temporary.
        """
        rows = int(dout.shape[0])
        out_f = int(dout.shape[1])
        k = int(values.shape[1])
        if rows <= 0 or out_f <= 0 or k == 0:
            return torch.empty_like(values)
        e = self._load_sparse()
        if (e is not None and dout.device.type == "cpu"
                and x.device.type == "cpu"
                and dout.dtype == torch.float32 and x.dtype == torch.float32
                and values.dtype == torch.float32
                and cols.dtype == torch.int64 and cols.is_contiguous()
                and values.is_contiguous()):
            dT = dout.t().contiguous()
            xT = x.t().contiguous()
            return e.sparse_grad_v(dT, xT, cols, torch.empty_like(values))
        self._warn_fallback_once("sparse_grad_v torch fallback (native missing or unsupported dtype)")
        out = torch.empty_like(values)
        b = max(1, min(out_f, int(budget) // max(1, rows * k * 4)))
        for o0 in range(0, out_f, b):
            o1 = min(o0 + b, out_f)
            g = x.index_select(1, cols[o0:o1].reshape(-1)).view(rows, o1 - o0, k)
            out[o0:o1] = (dout[:, o0:o1].unsqueeze(-1) * g).sum(0)
        return out

    def _load_quant(self):
        if self._quant is not None:
            return None if self._quant is False else self._quant
        with self._lock:
            if self._quant is not None:
                return None if self._quant is False else self._quant
            try:
                from torch.utils.cpp_extension import load
                root = Path(__file__).resolve().parent
                self._quant = load(name="smaul_quant",
                    sources=[str(root / "quant_cpu.cpp")],
                    extra_cflags=_native_cflags(), verbose=False)
            except Exception as exc:
                self._quant = False
                warnings.warn(f"fp8 quantizer native ext unavailable; torch fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._quant is False else self._quant

    def fp8_quantize_tiles(self, w32, tile):
        """Tiled E4M3 quantize, or None to let the caller use the torch path.

        Returns (codes uint8 [out_f, nt*tile] padded, scales float32
        [out_f, nt], non-finite input count). Bit-exact with the torch
        reference by construction: the codebook permutation and midpoints are
        taken from the same cached ``_tables`` rather than rebuilt in C++,
        because E4M3 has two codes for +448 and two for -448 and the tie order
        comes from torch.argsort.
        """
        e = self._load_quant()
        if e is not None and w32.device.type == "cpu" \
                and w32.dtype == torch.float32 and w32.is_contiguous():
            from kernel.fp8_tile import _quant_tables
            order, bounds = _quant_tables(w32.device)
            codes, sc, nf = e.fp8_quantize_tiles(w32, tile, order, bounds)
            return codes, sc, nf
        return None

    def attn_forward(self, Q, K, V, eps, need_den, need_state=False):
        """Returns (Y, DEN, S, z).

        S and z are the final recurrent state. With the native extension they
        are ``[B, H, D, D]`` / ``[B, H, D]`` **only when ``need_state`` is
        set**; otherwise they are zero-element placeholders, because the
        training path never reads them and materializing them costs
        ``B*H*D*D*4`` bytes of memset per call (4 MiB at B4/H16/D128, so
        32 MiB/step over 8 layers). The torch reference fallback has no state
        to hand back and returns ``None, None``.

        Callers that want the state must pass ``need_state=True``; treat the
        return value as usable only in that case.
        """
        e = self._load_attn()
        if (e is not None and Q.device.type == "cpu" and Q.dtype == torch.float32
                and K.dtype == torch.float32 and V.dtype == torch.float32):
            Qc = Q if Q.is_contiguous() else Q.contiguous()
            Kc = K if K.is_contiguous() else K.contiguous()
            Vc = V if V.is_contiguous() else V.contiguous()
            Y, DEN, S, z = e.attn_forward(Qc, Kc, Vc, float(eps), bool(need_den),
                                           bool(need_state))
            return Y, (DEN if need_den else None), S, z
        Y = _get_attn_ref()(Q, K, V, eps)
        return Y, None, None, None

    def attn_step(self, S, z, q, k, v, eps):
        """One decode step, advancing the carried (S, z) in place.

        Returns (y, S, z) with y [B, H, D]. This is what turns inference from
        O(N*T) into O(T + N*D^2): without it every generated token re-runs the
        whole prefix, so generating N tokens is quadratic in N.
        """
        e = self._load_attn()
        if (e is not None and S.device.type == "cpu"
                and S.dtype == torch.float32 and z.dtype == torch.float32
                and q.dtype == torch.float32 and k.dtype == torch.float32
                and v.dtype == torch.float32
                and S.is_contiguous() and z.is_contiguous()
                and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            return e.attn_step(S, z, q, k, v, float(eps))
        # torch fallback: the same recurrence, vectorised over (b, h).
        kn = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        S.add_(kn.unsqueeze(-1) * v.unsqueeze(-2))
        z.add_(kn)
        num = (q.unsqueeze(-2) @ S).squeeze(-2)
        den = (q * z).sum(-1, keepdim=True).clamp_min(eps)
        return num / den, S, z

    def attn_backward(self, dY, Q, K, V, Y, DEN, eps):
        e = self._load_attn()
        if (e is not None and DEN is not None and dY.device.type == "cpu"
                and dY.dtype == torch.float32 and Q.dtype == torch.float32
                and K.dtype == torch.float32 and V.dtype == torch.float32
                and Y.dtype == torch.float32):
            args = [a if a.is_contiguous() else a.contiguous() for a in (dY, Q, K, V, Y, DEN)]
            return e.attn_backward(*args, float(eps))
        return _get_attn_ref_bwd()(dY, Q, K, V, eps)


def _torch_forward(x, w, s, in_f, out_f, tile):
    # Honors the layer's tile; _OB stays fixed because it is only an output-row
    # blocking factor for cache reuse, not a quantization boundary.
    from kernel.fp8_tile import _OB, decode_tile
    with torch.no_grad():
        y = torch.zeros(x.shape[0], out_f, dtype=torch.float32, device=x.device)
        OB, nt = _OB, (in_f + tile - 1) // tile
        for o0 in range(0, out_f, OB):
            o1 = min(o0 + OB, out_f)
            acc = torch.zeros(x.shape[0], o1 - o0, dtype=torch.float32, device=x.device)
            for t in range(nt):
                k1 = min(in_f, (t + 1) * tile)
                acc += x[:, t * tile:k1] @ decode_tile(w, s, o0, o1, t, tile).T
            y[:, o0:o1] = acc
        return y


def _torch_backward_input(g, w, s, in_f, out_f, tile):
    from kernel.fp8_tile import _OB, decode_tile
    with torch.no_grad():
        gx = torch.zeros(g.shape[0], in_f, dtype=torch.float32, device=g.device)
        OB, nt = _OB, (in_f + tile - 1) // tile
        for o0 in range(0, out_f, OB):
            o1 = min(o0 + OB, out_f)
            gb = g[:, o0:o1]
            for t in range(nt):
                gx[:, t * tile:min(in_f, (t + 1) * tile)] += gb @ decode_tile(w, s, o0, o1, t, tile)
        return gx


register_backend(CpuBackend())

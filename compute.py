#!/usr/bin/env python3
"""Minimal compute-backend boundary for SmaulNative.

The model (smaul_linear.py) and FP8 autograd (fp8_tile.py) call into a
backend selected with get_backend(); they never touch extensions directly.
The CPU backend is default, fully functional, and independently testable.
Future backends (e.g. AMD FP16/ROCm) register via register_backend()
without changing model code. No fake backends: unknown names raise.
"""
import os
import warnings
from pathlib import Path

import torch

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

    def configure(self, threads=None):
        if threads is None:
            env = os.environ.get("SMAUL_CPU_THREADS")
            threads = int(env) if env else max(1, (os.cpu_count() or 2) // 2)
        threads = max(1, threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(threads))
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
        try:
            from torch.utils.cpp_extension import load
            root = Path(__file__).resolve().parent
            self._ext = load(name="smaul_fp8_ivb", sources=[str(root / "fp8_cpu.cpp")],
                extra_cflags=["-O3", "-mavx", "-mf16c", "-msse4.2", "-mno-avx2", "-mno-avx512f", "-ffp-contract=off"],
                verbose=False)
        except Exception as exc:
            self._ext = False
            warnings.warn(f"FP8 native ext unavailable; torch tiled fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._ext is False else self._ext

    def fp8_forward(self, x, w, s, in_f, out_f, tile):
        e = self._load()
        if e is not None and x.device.type == "cpu":
            return e.fp8_forward(x, w.reshape(-1), s.reshape(-1), in_f, out_f, tile)
        return _torch_forward(x, w, s, in_f, out_f, tile)

    def fp8_backward_input(self, g, w, s, in_f, out_f, tile):
        e = self._load()
        if e is not None and g.device.type == "cpu":
            return e.fp8_backward_input(g, w.reshape(-1).contiguous(), s.reshape(-1).contiguous(), in_f, out_f, tile)
        return _torch_backward_input(g, w, s, in_f, out_f, tile)

    def _load_attn(self):
        if self._attn is not None:
            return None if self._attn is False else self._attn
        try:
            from torch.utils.cpp_extension import load
            root = Path(__file__).resolve().parent
            self._attn = load(name="smaul_attn", sources=[str(root / "attn_cpu.cpp")],
                extra_cflags=["-O3", "-mavx", "-mf16c", "-msse4.2", "-mno-avx2", "-mno-avx512f", "-ffp-contract=off"],
                verbose=False)
        except Exception as exc:
            self._attn = False
            warnings.warn(f"linear-attention native ext unavailable; python reference fallback ({type(exc).__name__})", RuntimeWarning, stacklevel=2)
        return None if self._attn is False else self._attn

    @property
    def has_attn_native(self):
        return self._load_attn() is not None

    def attn_forward(self, Q, K, V, eps, need_den):
        e = self._load_attn()
        if (e is not None and Q.device.type == "cpu" and Q.dtype == torch.float32
                and K.dtype == torch.float32 and V.dtype == torch.float32):
            Qc = Q if Q.is_contiguous() else Q.contiguous()
            Kc = K if K.is_contiguous() else K.contiguous()
            Vc = V if V.is_contiguous() else V.contiguous()
            Y, DEN = e.attn_forward(Qc, Kc, Vc, float(eps), bool(need_den))
            return Y, (DEN if need_den else None)
        from smaul_linear import _attn_reference
        return _attn_reference(Q, K, V, eps), None

    def attn_backward(self, dY, Q, K, V, Y, DEN, eps):
        e = self._load_attn()
        if (e is not None and DEN is not None and dY.device.type == "cpu"
                and dY.dtype == torch.float32 and Q.dtype == torch.float32
                and K.dtype == torch.float32 and V.dtype == torch.float32
                and Y.dtype == torch.float32):
            args = [a if a.is_contiguous() else a.contiguous() for a in (dY, Q, K, V, Y, DEN)]
            return e.attn_backward(*args, float(eps))
        from smaul_linear import _attn_reference_backward
        return _attn_reference_backward(dY, Q, K, V, eps)


def _torch_forward(x, w, s, in_f, out_f, tile):
    from fp8_tile import decode_tile
    y = torch.zeros(x.shape[0], out_f, dtype=torch.float32, device=x.device)
    OB, nt = 64, (in_f + tile - 1) // tile
    for o0 in range(0, out_f, OB):
        o1 = min(o0 + OB, out_f)
        acc = torch.zeros(x.shape[0], o1 - o0, dtype=torch.float32, device=x.device)
        for t in range(nt):
            k1 = min(in_f, (t + 1) * tile)
            acc += x[:, t * tile:k1] @ decode_tile(w, s, o0, o1, t, tile).T
        y[:, o0:o1] = acc
    return y


def _torch_backward_input(g, w, s, in_f, out_f, tile):
    from fp8_tile import decode_tile
    gx = torch.zeros(g.shape[0], in_f, dtype=torch.float32, device=g.device)
    OB, nt = 64, (in_f + tile - 1) // tile
    for o0 in range(0, out_f, OB):
        o1 = min(o0 + OB, out_f)
        gb = g[:, o0:o1]
        for t in range(nt):
            gx[:, t * tile:min(in_f, (t + 1) * tile)] += gb @ decode_tile(w, s, o0, o1, t, tile)
    return gx


register_backend(CpuBackend())

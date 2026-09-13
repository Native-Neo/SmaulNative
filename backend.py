#!/usr/bin/env python3

import ctypes.util
import os


def _library_available(*names):
    return any(ctypes.util.find_library(name) for name in names)


def _cuda_available(torch):
    return bool(torch.cuda.is_available() and torch.version.cuda)


def _hip_available(torch):
    return bool(torch.cuda.is_available() and torch.version.hip)


def _aocl_available():
    return _library_available("amdblis", "blis", "flame", "amdlibm")


def _mkl_available(torch):
    try:
        config = torch.__config__.show()
    except Exception:
        config = ""
    return "mkl" in config.lower() or _library_available("mkl_rt", "mkl_core")


def detect_backend(torch, force_cpu=False):
    if not force_cpu and _hip_available(torch):
        return "hip"
    if not force_cpu and _cuda_available(torch):
        return "cuda"
    if _aocl_available():
        return "aocl"
    if _mkl_available(torch):
        return "mkl"
    return None


def require_backend(torch, force_cpu=False):
    backend = detect_backend(torch, force_cpu)
    if backend is None:
        print("ERR: Device NOT supported!")
        raise RuntimeError("Device NOT supported")
    return backend


def backend_device(torch, backend):
    if backend in ("hip", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")

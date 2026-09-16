#!/usr/bin/env python3

import ctypes.util


def _library_available(*names):
    return any(ctypes.util.find_library(name) for name in names)


def _torch_gpu_backend(torch):
    if not torch.cuda.is_available():
        return None
    if torch.version.hip:
        return "hip"
    if torch.version.cuda:
        return "cuda"
    return None


def _gpu_supported(torch, backend):
    if backend == "cuda":
        try:
            major, minor = torch.cuda.get_device_capability(0)
            return (major, minor) >= (6, 0)
        except Exception:
            return False
    if backend == "hip":
        try:
            arch = str(torch.cuda.get_device_properties(0).gcnArchName).lower()
        except Exception:
            return False
        if not arch.startswith("gfx"):
            return False
        try:
            return int(arch[3:5]) >= 8
        except ValueError:
            return False
    return False


def _mkl_available(torch):
    try:
        config = torch.__config__.show().lower()
    except Exception:
        config = ""
    return "mkl" in config or _library_available("mkl_rt", "mkl_core")


def _aocl_available():
    return _library_available("amdblis")


def detect_backend(torch, force_cpu=False):
    if not force_cpu:
        gpu_backend = _torch_gpu_backend(torch)
        if gpu_backend and _gpu_supported(torch, gpu_backend):
            return gpu_backend
    if _aocl_available():
        return "aocl"
    if _mkl_available(torch):
        return "mkl"
    return None


def backend_device(torch, backend):
    if backend in ("hip", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


def backend_name(backend):
    return {
        "hip": "HIP",
        "cuda": "CUDA",
        "aocl": "AOCL",
        "mkl": "MKL",
    }.get(backend, backend.upper())


def require_backend(torch, force_cpu=False):
    backend = detect_backend(torch, force_cpu)
    if backend is None:
        print("ERR: Device NOT supported!")
        raise RuntimeError("Device NOT supported")
    if backend in ("aocl", "mkl"):
        try:
            from cpu import configure
            configure()
        except Exception as exc:
            raise RuntimeError(f"failed to configure native CPU backend: {exc}") from exc
    print(f"[BACKEND] {backend_name(backend)}")
    return backend

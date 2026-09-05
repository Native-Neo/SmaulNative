import os
from pathlib import Path

import torch

from cpu_backend import NativeLion, configure as _configure

_WKV_EXT = None
_WKV_ORIG = None


def _load_wkv():
    global _WKV_EXT
    if _WKV_EXT is None:
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parent
        _WKV_EXT = load(
            name="smaulnative_wkv",
            sources=[str(root / "wkv_kernel.cpp")],
            extra_cflags=["-O3", "-march=native", "-mavx"],
            verbose=False,
        )
    return _WKV_EXT


class _NativeWKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, w, k, v, kk, a, r):
        ctx.save_for_backward(state, w, k, v, kk, a, r)
        ext = _load_wkv()
        out_state, y = ext.wkv_forward(
            state.contiguous(), w.float().contiguous(), k.float().contiguous(),
            v.float().contiguous(), kk.float().contiguous(), a.float().contiguous(),
            r.float().contiguous()
        )
        return out_state, y.to(dtype=r.dtype)

    @staticmethod
    def backward(ctx, grad_state, grad_y):
        saved = ctx.saved_tensors
        with torch.enable_grad():
            inputs = [x.detach().requires_grad_(True) for x in saved]
            out_state, y = _WKV_ORIG(*inputs)
            if grad_state is None:
                grad_state = torch.zeros_like(out_state)
            if grad_y is None:
                grad_y = torch.zeros_like(y)
            grads = torch.autograd.grad(
                (out_state, y), inputs, (grad_state, grad_y), allow_unused=True
            )
        return grads


def _native_wkv(state, w, k, v, kk, a, r):
    return _NativeWKV.apply(state, w, k, v, kk, a, r)


def configure(threads=None):
    global _WKV_ORIG
    threads = _configure(threads)
    torch.set_float32_matmul_precision("high")
    try:
        import rwkv_x_core
        if _WKV_ORIG is None:
            _WKV_ORIG = rwkv_x_core._wkv_run_chunk
            rwkv_x_core._wkv_run_chunk = _native_wkv
        old_init = rwkv_x_core.RWKVXConfig.__init__
        if not getattr(old_init, "_smaul_cpu", False):
            chunk = int(os.environ.get("SMAUL_WKV_CHUNK", "128"))
            checkpoint_ffn = os.environ.get("SMAUL_CHECKPOINT_FFN", "0") not in {"0", "false", "False"}

            def init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                self.wkv_chunk_size = chunk
                self.checkpoint_ffn = checkpoint_ffn

            init._smaul_cpu = True
            rwkv_x_core.RWKVXConfig.__init__ = init
    except Exception:
        pass
    return threads


__all__ = ["NativeLion", "configure"]

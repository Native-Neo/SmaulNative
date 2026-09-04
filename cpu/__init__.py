import os
import torch

from cpu_backend import NativeLion, configure as _configure


def configure(threads=None):
    threads = _configure(threads)
    torch.set_float32_matmul_precision("high")
    try:
        import rwkv_x_core
        old_init = rwkv_x_core.RWKVXConfig.__init__
        if not getattr(old_init, "_smaul_cpu", False):
            chunk = int(os.environ.get("SMAUL_WKV_CHUNK", "128"))

            def init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                self.wkv_chunk_size = chunk

            init._smaul_cpu = True
            rwkv_x_core.RWKVXConfig.__init__ = init
    except Exception:
        pass
    return threads

__all__ = ["NativeLion", "configure"]

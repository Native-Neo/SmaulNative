import os
from pathlib import Path

import torch
from torch.optim import Optimizer

_EXT = None


def configure(threads=None):
    threads = threads or int(os.environ.get("SMAUL_CPU_THREADS", min(os.cpu_count() or 1, 4)))
    threads = max(1, threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return threads


def _load():
    global _EXT
    if _EXT is not None:
        return _EXT
    from torch.utils.cpp_extension import load
    root = Path(__file__).resolve().parent
    _EXT = load(
        name="smaulnative_cpu",
        sources=[str(root / "cpu_kernels.cpp")],
        extra_cflags=["-O3", "-march=native"],
        verbose=False,
    )
    return _EXT


class NativeLion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        self._ext = _load()

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32 or not p.is_contiguous() or not p.grad.is_contiguous():
                    self._fallback(p, p.grad, lr, b1, b2, wd)
                    continue
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                self._ext.lion_step(p, p.grad, state["exp_avg"], lr, b1, b2, wd)
        return loss

    @staticmethod
    def _fallback(p, g, lr, b1, b2, wd):
        state = getattr(p, "_smaul_lion_state", None)
        if state is None:
            state = torch.zeros_like(p)
            p._smaul_lion_state = state
        state.mul_(b1).add_(g, alpha=1 - b1)
        if wd:
            p.mul_(1 - lr * wd)
        p.add_(state.sign(), alpha=-lr)
        state.lerp_(g, 1 - b2)

import os
from pathlib import Path

import torch
import torch.nn.functional as F

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
        state, w, k, v, kk, a, r = ctx.saved_tensors
        ext = _load_wkv()
        state_f = state.float().contiguous()
        w_f = w.float().contiguous()
        k_f = k.float().contiguous()
        v_f = v.float().contiguous()
        kk_f = kk.float().contiguous()
        a_f = a.float().contiguous()
        r_f = r.float().contiguous()
        if grad_state is None:
            grad_state = torch.zeros_like(state_f)
        else:
            grad_state = grad_state.float().contiguous()
        if grad_y is None:
            grad_y = torch.zeros_like(r_f)
        else:
            grad_y = grad_y.float().contiguous()
        grads = ext.wkv_backward(
            state_f, w_f, k_f, v_f, kk_f, a_f, r_f, grad_state, grad_y
        )
        return tuple(
            None if g is None else g.to(dtype=src.dtype)
            for g, src in zip(grads, (state, w, k, v, kk, a, r))
        )


def _native_wkv(state, w, k, v, kk, a, r):
    return _NativeWKV.apply(state, w, k, v, kk, a, r)


def _linear_memory_attention(self, x):
    B, T, C = x.shape
    H, N = self.n_head, C // self.n_head
    q = self.receptance(x).view(B, T, H, N).transpose(1, 2)
    k = self.key(x).view(B, T, H, N).transpose(1, 2)
    v = self.value(x).view(B, T, H, N).transpose(1, 2)
    phi_q = F.elu(q) + 1.0
    phi_k = F.elu(k) + 1.0
    state = torch.zeros(B, H, N, N, dtype=q.dtype, device=x.device)
    norm = torch.zeros(B, H, N, dtype=q.dtype, device=x.device)
    ys = []
    local = max(1, self.chunk_size)
    for lo in range(0, T, local):
        hi = min(lo + local, T)
        q_i = q[:, :, lo:hi]
        k_i = k[:, :, lo:hi]
        v_i = v[:, :, lo:hi]
        local_y = F.scaled_dot_product_attention(q_i, k_i, v_i, is_causal=True)
        pk = phi_k[:, :, lo:hi]
        pv = v_i
        kv = torch.einsum("bhtn,bhtm->bhtnm", pk, pv).cumsum(dim=2)
        z = pk.cumsum(dim=2)
        state_i = state.unsqueeze(2) + kv
        norm_i = norm.unsqueeze(2) + z
        global_y = torch.einsum("bhtn,bhtnm->bhtm", phi_q[:, :, lo:hi], state_i)
        denom = (phi_q[:, :, lo:hi] * norm_i).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        global_y = global_y / denom
        state = state_i[:, :, -1]
        norm = norm_i[:, :, -1]
        ys.append((local_y + global_y) * 0.5)
    y = torch.cat(ys, dim=2)
    return self.output(y.transpose(1, 2).contiguous().view(B, T, C))


def configure(threads=None):
    global _WKV_ORIG
    threads = _configure(threads)
    torch.set_float32_matmul_precision("high")
    import rwkv_x_core
    if _WKV_ORIG is None:
        _WKV_ORIG = rwkv_x_core._wkv_run_chunk
        rwkv_x_core._wkv_run_chunk = _native_wkv
    old_att_forward = rwkv_x_core.CausalSelfAttention.forward
    if not getattr(old_att_forward, "_smaul_linear", False):
        use_linear = os.environ.get("SMAUL_LINEAR_MEMORY", "1") not in {"0", "false", "False"}

        def att_forward(self, x):
            if use_linear:
                return _linear_memory_attention(self, x)
            return old_att_forward(self, x)

        att_forward._smaul_linear = True
        rwkv_x_core.CausalSelfAttention.forward = att_forward
    old_init = rwkv_x_core.RWKVXConfig.__init__
    if not getattr(old_init, "_smaul_cpu", False):
        chunk = int(os.environ.get("SMAUL_WKV_CHUNK", "256"))
        moba_chunk = int(os.environ.get("SMAUL_MOBA_CHUNK", "128"))
        checkpoint_ffn = os.environ.get("SMAUL_CHECKPOINT_FFN", "0") not in {"0", "false", "False"}

        def init(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            if self.head_size > 128:
                raise ValueError(f"--cpu native WKV needs head_size <= 128, got {self.head_size}")
            self.wkv_chunk_size = chunk
            self.moba_chunk_size = moba_chunk
            self.checkpoint_ffn = checkpoint_ffn

        init._smaul_cpu = True
        rwkv_x_core.RWKVXConfig.__init__ = init
    return threads


__all__ = ["NativeLion", "configure"]

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from kernel.compute import get_backend
from kernel.fp8_tile import FP8Linear, fp8_modules

BYTE_VOCAB_SIZE = 256


@dataclass
class LinearConfig:
    vocab_size: int = BYTE_VOCAB_SIZE
    d_model: int = 512
    n_layer: int = 8
    n_heads: int = 8
    ffn_mult: float = 2.5
    eps: float = 1e-6
    tile: int = 64
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
    # Weight of the Switch-style load-balancing aux loss for the MoE router.
    moe_balance_weight: float = 0.01
    precision: str = "fp8"
    tokenizer_sha256: str = ""
    dataset_fingerprint: str = ""
    # Architecture selector: "plain" (dense baseline, historical behavior) or
    # "rawr" (sparse structural variant; both use Linear Attention).
    # Code-level default stays "plain" so existing checkpoints/tests that
    # build LinearConfig without this field keep historical behavior; the
    # train/infer CLIs default to "rawr" explicitly.
    architecture: str = "plain"
    # Embedding storage: "ram" (nn.Embedding) or "mmap" (file-backed).
    embedding_storage: str = "ram"
    # Rawr structural sparsity: fraction of hidden/head connections omitted.
    rawr_sparsity: float = 0.5
    rawr_min_degree: int = 4
    rawr_graph_hash: str = ""
    rawr_edge_count: int = 0

    def __post_init__(self):
        if self.precision not in ("fp8", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}; expected 'fp8' or 'fp32'")
        if self.architecture not in ("plain", "rawr"):
            raise ValueError(f"unknown architecture {self.architecture!r}; expected 'plain' or 'rawr'")
        if self.embedding_storage not in ("ram", "mmap"):
            raise ValueError(
                f"unknown embedding_storage {self.embedding_storage!r}; expected 'ram' or 'mmap'")
        for name in ("vocab_size", "d_model", "n_layer", "n_heads", "tile", "num_experts",
                     "num_experts_per_tok"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if not isinstance(self.ffn_mult, (int, float)) or not math.isfinite(self.ffn_mult) \
                or self.ffn_mult <= 0:
            raise ValueError(f"ffn_mult must be positive finite, got {self.ffn_mult!r}")
        if not isinstance(self.eps, float) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError(f"eps must be positive finite, got {self.eps!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.is_moe and self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({self.num_experts_per_tok}) > num_experts ({self.num_experts})")
        if not isinstance(self.moe_balance_weight, (int, float)) \
                or not math.isfinite(self.moe_balance_weight) \
                or self.moe_balance_weight < 0:
            raise ValueError(
                f"moe_balance_weight must be non-negative finite, got {self.moe_balance_weight!r}")
        if not isinstance(self.rawr_sparsity, (int, float)) \
                or not 0.0 <= float(self.rawr_sparsity) < 1.0:
            raise ValueError(f"rawr_sparsity must be in [0, 1), got {self.rawr_sparsity!r}")
        if not isinstance(self.rawr_min_degree, int) or self.rawr_min_degree < 1:
            raise ValueError(f"rawr_min_degree must be a positive int, got {self.rawr_min_degree!r}")
        if not isinstance(self.rawr_edge_count, int) or self.rawr_edge_count < 0:
            raise ValueError(f"rawr_edge_count must be non-negative, got {self.rawr_edge_count!r}")

    def save(self, p: Path): Path(p).write_text(json.dumps(asdict(self), indent=2))
    @classmethod
    def load(cls, p: Path):
        try:
            data = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"could not load config {p}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"config {p} must contain a JSON object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(f"invalid config {p}: {exc}") from exc

def _linear(cfg: LinearConfig, in_f: int, out_f: int, bias: bool = False) -> nn.Module:
    """Precision-selected linear layer: tiled-E4M3 FP8 or plain FP32."""
    if in_f <= 0 or out_f <= 0:
        raise ValueError(f"in_f/out_f must be positive, got {in_f}/{out_f}")
    if cfg.precision == "fp8":
        # FP8Linear supports bias; honor it so fp8/fp32 modes do not diverge.
        return FP8Linear(in_f, out_f, cfg.tile, bias)
    if cfg.precision == "fp32":
        return _DenseLinear(in_f, out_f, bias=bias)
    raise ValueError(f"unknown precision {cfg.precision!r}; expected 'fp8' or 'fp32'")

class _DenseLinearFn(torch.autograd.Function):
    """FP32 dense projection backed by SmaulNative's AVX1 SGEMM."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        x2 = x.float().reshape(-1, weight.shape[1]).contiguous()
        y2 = get_backend().sgemm_bt(x2, weight.contiguous())
        ctx.save_for_backward(x2, weight)
        ctx.xshape = tuple(x.shape)
        ctx.xdtype = x.dtype
        ctx.has_bias = bias is not None
        if bias is not None:
            y2 = y2 + bias
        return y2.reshape(*x.shape[:-1], weight.shape[0]).to(
            x.dtype if x.is_floating_point() else torch.float32
        )

    @staticmethod
    def backward(ctx, grad):
        x2, weight = ctx.saved_tensors
        g2 = grad.float().reshape(-1, weight.shape[0]).contiguous()
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = get_backend().sgemm(g2, weight.contiguous()).reshape(ctx.xshape).to(ctx.xdtype)
        if ctx.needs_input_grad[1]:
            gw = get_backend().sgemm(g2.transpose(0, 1).contiguous(), x2)
        if ctx.needs_input_grad[2] and ctx.has_bias:
            gb = g2.sum(0)
        return gx, gw, gb


class _DenseLinear(nn.Module):
    """Plain-FP32 linear using the Ivy Bridge AVX1 backend."""

    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return _DenseLinearFn.apply(x, self.lin.weight, self.lin.bias)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"RMSNorm eps must be positive finite, got {eps!r}")
        self.weight = nn.Parameter(torch.ones(d, dtype=torch.float32))
        self.eps = eps
    def forward(self, x):
        # NOTE: the obvious allocation saving here is to scale in place
        # (measured 2.554 -> 1.851 ms on [2,256,512] bf16, bit-identical), but
        # `x.float()` is an autograd graph output, so mul_ on it is rejected.
        # Recovering it needs a custom autograd Function with a hand-written
        # RMSNorm backward; at ~0.4% of a step that is not worth the risk.
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(v + self.eps) * self.weight).to(x.dtype if x.is_floating_point() else torch.float32)

class LinearAttention(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        d, self.nh = cfg.d_model, cfg.n_heads
        if d % self.nh != 0:
            raise ValueError(f"d_model ({d}) must be divisible by n_heads ({self.nh})")
        self.hd = d // self.nh
        self.eps = cfg.eps
        self.q = _linear(cfg, d, d)
        self.k = _linear(cfg, d, d)
        self.v = _linear(cfg, d, d)
        self.o = _linear(cfg, d, d)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.nh, self.hd
        q = self.q(x).float().view(B, T, H, D)
        k = self.k(x).float().view(B, T, H, D)
        v = self.v(x).float().view(B, T, H, D)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        y = _LinearAttnFn.apply(q, k, v, self.eps).reshape(B, T, -1)
        return self.o(y.to(x.dtype if x.is_floating_point() else torch.float32))

    def _qkv(self, x, B, T):
        H, D = self.nh, self.hd
        q = F.elu(self.q(x).float().view(B, T, H, D)) + 1.0
        k = F.elu(self.k(x).float().view(B, T, H, D)) + 1.0
        v = self.v(x).float().view(B, T, H, D)
        return q, k, v

    def prefill(self, x):
        """Run the whole prefix once and return (out, _AttnState).

        Inference only. Training uses ``forward`` (autograd); this path calls
        the backend directly and carries the O(D^2) recurrence state forward so
        each generated token costs one ``step`` instead of a full re-forward.
        """
        B, T, _ = x.shape
        q, k, v = self._qkv(x, B, T)
        y, _DEN, S, z = get_backend().attn_forward(q, k, v, self.eps, False,
                                                    need_state=True)
        y = y.reshape(B, T, -1).to(x.dtype if x.is_floating_point() else torch.float32)
        return self.o(y), _AttnState(S, z, self.eps)

    def step(self, x, st):
        """One decode step for a single position, advancing ``st`` in place.

        Mirrors ``prefill``: returns (out, state).
        """
        B = x.shape[0]
        q, k, v = self._qkv(x, B, 1)
        q, k, v = q[:, 0], k[:, 0], v[:, 0]
        y, st.S, st.z = get_backend().attn_step(st.S, st.z, q, k, v, st.eps)
        y = y.reshape(B, 1, -1).to(x.dtype if x.is_floating_point() else torch.float32)
        return self.o(y), st


def _attn_reference(Q, K, V, eps):
    """Pure-torch linear-attention recurrence (exact math reference).

    Same formulas as the native kernel: elu+1 feature map applied by the
    caller, per-step key normalization, FP32 S/z state, causal steps, no
    softmax, no QK^T.
    """
    B, T, H, D = Q.shape
    S = torch.zeros(B, H, D, D, dtype=torch.float32, device=Q.device)
    z = torch.zeros(B, H, D, dtype=torch.float32, device=Q.device)
    ys = []
    for t in range(T):
        kt, vt, qt = K[:, t], V[:, t], Q[:, t]
        kt = kt / kt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        S = S + kt.unsqueeze(-1) @ vt.unsqueeze(-2)
        z = z + kt
        num = (qt.unsqueeze(-2) @ S).squeeze(-2)
        den = (qt * z).sum(-1, keepdim=True).clamp_min(eps)
        ys.append(num / den)
    return torch.stack(ys, 1)


def _attn_reference_backward(dY, Q, K, V, eps):
    """Gradients via autograd through the reference recurrence."""
    wants = (Q.requires_grad, K.requires_grad, V.requires_grad)
    Qr = Q.detach().requires_grad_(wants[0])
    Kr = K.detach().requires_grad_(wants[1])
    Vr = V.detach().requires_grad_(wants[2])
    with torch.enable_grad():
        Yr = _attn_reference(Qr, Kr, Vr, eps)
        outs = [t for t, w in zip((Qr, Kr, Vr), wants) if w]
        grads = torch.autograd.grad(Yr, outs, dY, allow_unused=True) if outs else []
    it = iter(grads)
    return tuple(next(it) if w else None for w in wants)


class _LinearAttnFn(torch.autograd.Function):
    """Linear-attention op: native AVX1 kernel with reference fallback.

    Forward runs the backend kernel (native when available). Backward runs the
    backend two-pass kernel, or autograd through the reference recurrence when
    native is unavailable. Q/K/V must already carry the elu+1 feature map; key
    normalization happens inside the op. Tensors are saved for backward only
    when grads are needed.
    """

    @staticmethod
    def forward(ctx, Q, K, V, eps):
        be = get_backend()
        need = Q.requires_grad or K.requires_grad or V.requires_grad
        Y, DEN = be.attn_forward(Q, K, V, eps, need)[:2]
        if need:
            ctx.save_for_backward(Q, K, V, Y, DEN) if DEN is not None else ctx.save_for_backward(Q, K, V, Y)
            ctx.has_den = DEN is not None
            ctx.eps = eps
        return Y

    @staticmethod
    def backward(ctx, dY):
        saved = ctx.saved_tensors
        Q, K, V, Y = saved[0], saved[1], saved[2], saved[3]
        DEN = saved[4] if ctx.has_den else None
        be = get_backend()
        dQ, dK, dV = be.attn_backward(dY, Q, K, V, Y, DEN, ctx.eps)
        return (dQ if Q.requires_grad else None,
                dK if K.requires_grad else None,
                dV if V.requires_grad else None, None)

class _AttnState:
    """Carried linear-attention state for one attention layer.

    The recurrence state is O(D^2) per (batch, head) and independent of T, so
    a prefill plus one ``step`` per generated token replaces re-running the
    whole prefix every time. ``inference.LinearInference`` drives this; it is
    not used in training.
    """

    __slots__ = ("S", "z", "eps")

    def __init__(self, S, z, eps):
        self.S, self.z, self.eps = S, z, eps


class SwiFFN(nn.Module):
    def __init__(self, cfg: LinearConfig):
        super().__init__()
        h = int(cfg.d_model * cfg.ffn_mult)
        self.gate = _linear(cfg, cfg.d_model, h)
        self.up = _linear(cfg, cfg.d_model, h)
        self.down = _linear(cfg, h, cfg.d_model)
    def forward(self, x):
        return self.down((F.silu(self.gate(x).float()) * self.up(x).float()).to(x.dtype if x.is_floating_point() else torch.float32))


def _csr_T_parts(cols, in_f):
    """(crow_indices, perm) of the CSR form of S^T (crow over in_f).

    d/dx is ``dout @ S``, and the sparse way to compute that is
    ``(S^T @ dout^T)^T`` -- so it wants S's *transpose*, not S. Rebuilding the
    transposed layout on every call is what made it 1.4-4.5x slower than the
    forward product (191 ms vs 63 ms for the 8000x512 head), and only the
    indices can be cached: ``values`` is rewritten by the optimizer every step,
    so the permuted value array is re-gathered per backward (see ``_csr_T``).

    ``perm`` is the argsort of the flattened column indices, so S^T's
    col_indices are ``perm // K`` (recomputed per call -- it is one cheap
    division) and its values are ``values.reshape(-1)[perm]``. Only ``perm``
    and ``crow`` are retained, and perm is int32 when it fits, so the retained
    extra is ~4 bytes per nonzero against ``cols``' 8.
    """
    flat = cols.reshape(-1)
    perm = torch.argsort(flat, stable=True)
    if perm.numel() < (1 << 31):
        perm = perm.to(torch.int32)
    counts = torch.bincount(flat, minlength=int(in_f))
    crow = torch.cat([torch.zeros(1, dtype=torch.long, device=flat.device),
                      counts.cumsum(0)])
    return crow, perm


def _csr_T(crow, perm, values, out_f, in_f):
    """Sparse CSR of S^T, re-gathering the (per-step-updated) values."""
    col = torch.div(perm, values.shape[1], rounding_mode="floor")
    return torch.sparse_csr_tensor(
        crow, col, values.reshape(-1)[perm.long()],
        size=(int(in_f), int(out_f)))


def _csr(cols, values, out_f, in_f):
    """CSR view of a (out_f, in_f) matrix with K nonzeros per row.

    This aliases storage that the module already holds: ``cols`` is
    [out_f, K], so ``cols.reshape(-1)`` is already in crow_indices order, and
    ``values.reshape(-1)`` is already the value array. Only the indptr is
    synthesized (out_f+1 int64, ~2% of the ``cols`` tensor it describes), and
    construction is ~27 us, so it is rebuilt per call rather than cached.
    """
    k = values.shape[1]
    return torch.sparse_csr_tensor(
        torch.arange(0, int(out_f) * k + 1, k, dtype=torch.int64,
                     device=values.device),
        cols.reshape(-1), values.reshape(-1),
        size=(int(out_f), int(in_f)))


class _SparseLinearFn(torch.autograd.Function):
    """out[..., o] = sum_j x[..., cols[o, j]] * values[o, j]

    The forward and d/dx are dispatched as sparse matrix products against a
    CSR view of ``cols``/``values`` (see ``_csr``). This is what makes the
    sparsity pay off: the previous gather-based formulation
    (``(x[..., cols] * values).sum(-1)``) touched ``rows * out_f * K``
    elements three times -- gather, multiply, reduce -- for one MAC each,
    i.e. ~2 FLOP per 8 bytes loaded, and a scatter-add with ~65 index
    collisions per column on the way back. Measured on an 8000x512 head at
    K=51, that was **23x slower than the dense FP32 GEMM it replaces**
    despite doing 10x less arithmetic, and it dominated the step (35.5 s,
    8.3 tok/s, with the head alone at 41%).

    As a sparse product the same arithmetic is 1.9-3.2x *faster* than that
    dense GEMM, because each nonzero is visited once and the input column is
    reused across every output row that selects it (out_f*K/in_f ~ 65 of
    them here) instead of being re-gathered per output row. Only the output
    is allocated, so the peak no longer scales with K either -- the
    ``[rows, out_f, K]`` block and its transient-budget machinery are
    gone from the forward.

    d/dvalues is the one product that cannot be a sparse product:
    ``grad_v[o, j] = values[o, j] * <dout[:, o], x[:, cols[o, j]]>`` needs the
    gathered activation, which in torch means materialising
    ``rows * out_f * K`` elements -- 209M fp32 (836 MiB) for the 8000x512
    head -- and reading it back twice, so the product runs at memory
    bandwidth (measured 1004 ms). ``backend.sparse_grad_v`` fuses it into one
    pass over dout with x cache-resident: ~19 MiB of traffic instead of
    ~3.3 GiB. Without the extension it falls back to the gather in row
    blocks bounded by ``_GRAD_V_BUDGET``.
    """

    @staticmethod
    def forward(ctx, x, cols, values, out_f, grad_v_budget, csr_t):
        ctx.save_for_backward(x, cols, values)
        ctx.out_f = int(out_f)
        ctx.grad_v_budget = int(grad_v_budget)
        # Non-tensor argument, hence not save_for_backward (which is for tensors)
        # and correspondingly None in backward's return. It is derived from
        # `cols` rather than being an input or output.
        ctx.csr_t_crow, ctx.csr_t_perm = csr_t
        in_f = x.shape[-1]
        xf = x.reshape(-1, in_f).float()
        y = torch.sparse.mm(_csr(cols, values, out_f, in_f), xf.t()).t()
        return y.contiguous().reshape(*x.shape[:-1], int(out_f))

    @staticmethod
    def backward(ctx, dout):
        x, cols, values = ctx.saved_tensors
        out_f = ctx.out_f
        d = x.shape[-1]
        xf = x.reshape(-1, d).float()
        dof = dout.reshape(-1, out_f).float()
        # d/dx = dout @ S, i.e. (S^T @ dout^T)^T, using the cached transposed
        # index layout (see _csr_T_parts). Same arithmetic as s.t() inline,
        # 1.4-4.5x faster because the transpose is not rebuilt per call.
        st = _csr_T(ctx.csr_t_crow, ctx.csr_t_perm, values, out_f, d)
        grad_x = torch.sparse.mm(st, dof.t()).t()
        grad_v = get_backend().sparse_grad_v(dof, xf, cols, values,
                                            ctx.grad_v_budget)
        return grad_x.reshape(x.shape), None, grad_v, None, None, None


class SparseLinear(nn.Module):
    """Fixed fan-in sparse FP32 linear driven by the Rawr graph.

    Only ``values`` (out_f x K) are stored as parameters; column indices come
    from the shared Rawr graph (kept as a non-persistent buffer, rebuilt from
    ``rawr_graph.json`` on load, so per-layer storage is values-only).
    Omitted connections consume neither storage nor FLOPs.

    ``cols`` is read-only after construction, so one tensor may be shared by
    several ``SparseLinear`` instances (as ``RawrFFN`` does for gate/up).
    """

    def __init__(self, in_f: int, out_f: int, cols):
        super().__init__()

        if in_f <= 0 or out_f <= 0:
            raise ValueError(f"in_f/out_f must be positive, got {in_f}/{out_f}")
        cols = torch.as_tensor(cols, dtype=torch.long)
        if cols.dim() != 2 or cols.shape[0] != out_f:
            raise ValueError(f"cols must be [out_f, K], got {tuple(cols.shape)}")
        if int(cols.min()) < 0 or int(cols.max()) >= in_f:
            raise ValueError("cols indices out of range")
        self.in_f, self.out_f = in_f, out_f
        self.register_buffer("cols", cols, persistent=False)
        self.values = nn.Parameter(torch.empty(out_f, cols.shape[1], dtype=torch.float32))
        nn.init.kaiming_uniform_(self.values, a=math.sqrt(5))

    # Cap the d/dvalues gather temporary in bytes, for the torch fallback only
    # (the native kernel fuses it and allocates just its output). Measured
    # fastest at 1.6 MiB (1004 ms) and worst at 102 MiB (1695 ms) on the head.
    _GRAD_V_BUDGET = 1 << 22

    def _csr_t_parts(self):
        """Cached S^T index layout, rebuilt if ``cols`` is ever replaced.

        Keyed on the buffer's address, shape, device and dtype -- not its
        identity, despite what an earlier version of this comment claimed. An
        address can be reused after the old tensor is freed, so a replaced
        ``cols`` of identical shape/dtype on the same device could hit a stale
        layout; in practice ``cols`` is built once in ``__init__`` and only
        ``.to(device)`` (new address, and usually a new device) exercises the
        rebuild, which is picked up rather than silently reusing a stale
        layout on the wrong device.

        The cache additionally refuses to cross the inference-mode boundary.
        ``SmaulLinear.prefill``/``step`` run under ``torch.inference_mode()``,
        so a prefill builds the layout out of *inference* tensors and caches
        them here; a later training forward would then reuse inference tensors,
        which are rejected in places normal tensors are accepted
        (``create_graph=True``, saved-for-backward, version-counter bumps). The
        ordinary forward/backward path happens to tolerate them, so this fails
        far from its cause -- hence an explicit check. Rebuilding is cheap
        relative to a step (measured 1.1 ms for the 8000x512 head) and only
        happens on the first forward after a mode change.
        """
        c = self.cols
        key = (c.data_ptr(), tuple(c.shape), c.device, c.dtype)
        cache = getattr(self, "_csr_t", None)
        if cache is not None and cache[0] == key and \
                cache[3] == torch.is_inference_mode_enabled():
            return cache[1], cache[2]
        parts = _csr_T_parts(c, self.in_f)
        self._csr_t = (key, parts[0], parts[1], torch.is_inference_mode_enabled())
        return parts

    def forward(self, x):
        out = _SparseLinearFn.apply(x, self.cols, self.values, self.out_f,
                                   self._GRAD_V_BUDGET, self._csr_t_parts())
        return out.to(x.dtype if x.is_floating_point() else torch.float32)


class RawrFFN(nn.Module):
    """SwiGLU FFN with graph-sparse projections (Rawr architecture)."""

    def __init__(self, cfg: LinearConfig, graph):
        super().__init__()
        if graph is None:
            raise ValueError("RawrFFN needs a Rawr graph (pass rawr_graph=)")
        from rawr_graph import hidden_cols

        h = int(cfg.d_model * cfg.ffn_mult)
        d = cfg.d_model
        mp = max(1, min(cfg.rawr_min_degree, d))
        mp_h = max(1, min(cfg.rawr_min_degree, h))
        # gate and up have identical (out_f, in_f, sparsity, min_degree), so
        # they get identical columns. Derive them once and share the tensor:
        # it is read-only (see SparseLinear.cols) and non-persistent, so this
        # halves the index RAM for the pair without touching state_dict.
        gate_up_cols = hidden_cols(h, d, graph, cfg.rawr_sparsity, mp)
        self.gate = SparseLinear(d, h, gate_up_cols)
        self.up = SparseLinear(d, h, gate_up_cols)
        self.down = SparseLinear(h, d, hidden_cols(d, h, graph, cfg.rawr_sparsity, mp_h))

    def forward(self, x):
        return self.down(
            (F.silu(self.gate(x).float()) * self.up(x).float()).to(
                x.dtype if x.is_floating_point() else torch.float32))

class SwiFFN_MoE(nn.Module):
    """Sparse MoE over the RAWR trunk: router picks top-k experts per token.

    Experts are ``RawrFFN`` (graph-sparse) on architecture='rawr' and
    ``SwiFFN`` (dense) on 'plain', so MoE reuses the same FFN the dense
    trunk would run rather than a parallel implementation. Only experts
    with at least one assigned token execute; the rest are skipped entirely
    (no forward, hence no backward and no gradient).
    """

    def __init__(self, cfg: LinearConfig, rawr_graph=None):
        super().__init__()
        if cfg.num_experts < 1 or cfg.num_experts_per_tok < 1:
            raise ValueError("num_experts and num_experts_per_tok must be >= 1")
        if cfg.num_experts_per_tok > cfg.num_experts:
            raise ValueError(
                f"num_experts_per_tok ({cfg.num_experts_per_tok}) > num_experts ({cfg.num_experts})")
        # No silent clamp: misconfig must fail fast instead of training a
        # different MoE than requested.
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.architecture = cfg.architecture
        if cfg.architecture == "rawr":
            if rawr_graph is None:
                from rawr_graph import fallback_graph

                rawr_graph = fallback_graph(cfg.vocab_size, cfg.rawr_min_degree)
            self.experts = nn.ModuleList(
                [RawrFFN(cfg, rawr_graph) for _ in range(cfg.num_experts)])
        else:
            self.experts = nn.ModuleList([SwiFFN(cfg) for _ in range(cfg.num_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        # Routing diagnostics (plain attributes, not state): refreshed every
        # forward, read by training logs and retention reports.
        self.last_prob = None
        self.last_topi = None
        self.last_executed = [False] * cfg.num_experts
        self.expert_token_counts = [0] * cfg.num_experts

    def balance_loss(self, prob: torch.Tensor) -> torch.Tensor:
        """Switch-Transformer load-balancing aux loss over the router softmax.

        Without it top-k routing collapses onto one or two experts. Add
        ``1e-2 * moe.balance_loss(prob)`` to the training loss, where ``prob``
        is the pre-topk router softmax -- ``forward`` computes it but does not
        return it, so a caller wiring this up must capture it from
        ``last_prob``.
        """
        density = prob.mean(0)
        return (density * density * self.num_experts).sum()

    def forward(self, x):
        prob = torch.softmax(self.gate(x.float()), -1)
        # The topk gather is differentiable in the selected values, so the gate
        # receives gradient through topv and none through topi: unselected
        # experts get exactly zero. That is the collapse mechanism
        # balance_loss() exists to counteract.
        self.last_prob = prob
        topv, topi = torch.topk(prob, self.top_k, -1)
        self.last_topi = topi.detach()
        denom = topv.sum(-1, keepdim=True)
        # Guard tiny denominators (would explode weights); fall back to uniform.
        tiny = denom.squeeze(-1) < 1e-6
        topv = torch.where(tiny.unsqueeze(-1), torch.full_like(topv, 1.0 / self.top_k),
                           topv / denom.clamp_min(1e-9))
        out = torch.zeros_like(x.float())
        counts = [0] * self.num_experts
        executed = [False] * self.num_experts
        for e, expert in enumerate(self.experts):
            w = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            m = (w.squeeze(-1) > 0)
            n = int(m.sum().item()) if m.numel() else 0
            counts[e] = n
            if m.any():
                executed[e] = True
                xm = x[m].contiguous()
                out[m] += expert(xm).float() * w[m]
        self.last_executed = executed
        self.expert_token_counts = counts
        return out.to(x.dtype if x.is_floating_point() else torch.float32)

    def routing_stats(self) -> dict:
        """Load-balancing diagnostics for the most recent forward."""
        counts = list(self.expert_token_counts)
        total = sum(counts)
        usage = [(c / total) if total else 0.0 for c in counts]
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "token_counts": counts,
            "usage_frac": usage,
            "executed": list(self.last_executed),
            "inactive_experts": sum(1 for c in counts if c == 0),
            "max_usage_frac": max(usage) if usage else 0.0,
        }

    def reset_routing_stats(self) -> None:
        self.last_prob = None
        self.last_topi = None
        self.last_executed = [False] * self.num_experts
        self.expert_token_counts = [0] * self.num_experts

class Block(nn.Module):
    def __init__(self, cfg, rawr_graph=None):
        super().__init__()
        d = cfg.d_model
        self.n1 = RMSNorm(d, cfg.eps)
        self.att = LinearAttention(cfg)
        self.n2 = RMSNorm(d, cfg.eps)
        self.n3 = RMSNorm(d, cfg.eps)
        if cfg.is_moe:
            self.ffn = SwiFFN_MoE(cfg, rawr_graph if cfg.architecture == "rawr" else None)
        elif cfg.architecture == "rawr":
            self.ffn = RawrFFN(cfg, rawr_graph)
        else:
            self.ffn = SwiFFN(cfg)
        self.n4 = RMSNorm(d, cfg.eps)
        self.n5 = RMSNorm(d, cfg.eps)
    def forward(self, x):
        ck = self.training and torch.is_grad_enabled()
        n1x = self.n1(x)
        # Attention is deliberately NOT checkpointed. Saving Q, K, V, Y and DEN costs
        # 5*rows*d*4 bytes; checkpointing avoids that by re-running the q/k/v/o
        # projections in backward for ~rows*d^2 MACs, so the trade worsens as d
        # grows rather than staying flat. Measured 1124.6 ms -> 899.1 ms per
        # block step for 5 MiB/layer at d=512, i.e. checkpointing measured
        # faster, so this is a measured call and not an assumed one. Revisit if
        # the block ever runs at a d where the re-run cost dominates.
        ax = self.att(n1x)
        a = self.n2(ax.float())
        x = self.n3((x.float() + a.float()).to(x.dtype))
        # The FFN checkpoint is kept: what it saves (the SwiGLU intermediates)
        # scales with h = d_model*ffn_mult, so it earns its keep on wide FFNs.
        fx = torch.utils.checkpoint.checkpoint(self.ffn, x, use_reentrant=False) if ck else self.ffn(x)
        f = self.n4(fx.float())
        return self.n5((x.float() + f.float()).to(x.dtype))

    def _tail(self, x, att_out):
        """n2..n5 given the attention output; shared by forward/prefill/step."""
        a = self.n2(att_out.float())
        x = self.n3((x.float() + a.float()).to(x.dtype))
        fx = self.ffn(x)
        f = self.n4(fx.float())
        return self.n5((x.float() + f.float()).to(x.dtype))

    def prefill(self, x):
        """Full-prefix pass that captures the attention state. Inference only."""
        ax, st = self.att.prefill(self.n1(x))
        return self._tail(x, ax), st

    def step(self, x, st):
        """One decode step for a single position, advancing st in place."""
        ax, st = self.att.step(self.n1(x), st)
        return self._tail(x, ax), st


def _check_legacy_tokenizer_file(d: Path, cfg) -> None:
    """Refuse old word-level checkpoints explicitly instead of misloading them.

    A pre-byte checkpoint carries a tokenizer.json with a string-keyed word
    vocabulary (no ``kind == "byte"``). Its embedding rows and graph columns
    index a different token space, so loading its tensors under a byte-level
    config would silently produce a broken model. There is no faithful
    migration (the token spaces are unrelated), hence a clear error telling
    the user to retrain. Checkpoints without a tokenizer file (unit-test
    fixtures, fresh saves) load normally.
    """
    tp = Path(d) / "tokenizer.json"
    if not tp.exists():
        return
    try:
        data = json.loads(tp.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return
    if not isinstance(data, dict):
        return
    if data.get("kind") == "byte":
        return
    if isinstance(data.get("vocab"), dict):
        raise ValueError(
            f"checkpoint {d} bundles a legacy word-level tokenizer "
            f"(version={data.get('version')!r}, vocab={len(data['vocab'])}); "
            "byte-level models (vocab 256) cannot reuse its embedding rows. "
            "Retrain the model; do not copy the old tokenizer.json forward.")


class SmaulLinear(nn.Module):
    def __init__(self, cfg, rawr_graph=None, emb_path=None):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = LinearConfig(**cfg)
        self.cfg = cfg
        if cfg.architecture == "rawr" and rawr_graph is None:
            from rawr_graph import fallback_graph

            rawr_graph = fallback_graph(cfg.vocab_size, cfg.rawr_min_degree)
        if rawr_graph is not None and rawr_graph.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"Rawr graph vocab {rawr_graph.vocab_size} != config vocab {cfg.vocab_size}")
        self.rawr_graph = rawr_graph
        from embeddings import create_embedding

        # Common embedding interface: RAM (nn.Embedding behavior) or mmap
        # (file-backed, OS-paged). FP32 table in both cases so Lion steps
        # do not stagnate; cast to bf16 on forward.
        self.emb = create_embedding(cfg.vocab_size, cfg.d_model,
                                    cfg.embedding_storage, emb_path)
        self.n0 = RMSNorm(cfg.d_model, cfg.eps)
        self.blocks = nn.ModuleList([Block(cfg, rawr_graph) for _ in range(cfg.n_layer)])
        self.nf = RMSNorm(cfg.d_model, cfg.eps)
        if cfg.architecture == "rawr":
            from rawr_graph import hidden_cols

            mp = max(1, min(cfg.rawr_min_degree, cfg.d_model))
            self.head = SparseLinear(cfg.d_model, cfg.vocab_size,
                                     hidden_cols(cfg.vocab_size, cfg.d_model, rawr_graph,
                                                 cfg.rawr_sparsity, mp))
        else:
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            with torch.no_grad():
                nn.init.normal_(self.head.weight, 0, 0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, labels=None, last_only=False):
        """Full-sequence forward (training) or last-token-only (inference).

        ``last_only=True`` runs the identical trunk (embedding, norms, Linear
        Attention blocks) over the whole context but projects only the final
        position through the LM head, returning ``[B, 1, V]`` instead of
        ``[B, T, V]``. Head math is position-independent (dense ``nn.Linear``
        or ``SparseLinear`` row gather), so the returned row matches the last
        row of the full computation. Training (``labels=...``) always uses
        the full path; combining it with ``last_only`` is rejected.
        """
        if last_only and labels is not None:
            raise ValueError("last_only inference cannot compute a loss (labels given)")
        x = self.emb(idx).to(torch.bfloat16)
        x = self.n0(x.float()).to(torch.bfloat16)
        for b in self.blocks:
            x = b(x)
        x = self.nf(x.float())
        if last_only:
            # Keep the dim so downstream [0, -1] indexing is unchanged.
            x = x[:, -1:, :]
        logits = self.head(x.float())
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) if labels is not None else None
        return logits, loss

    # ------------------------------------------------------------------
    # Incremental decoding (inference only).
    #
    # Linear attention's state is O(D^2) per (batch, head) and independent of
    # the sequence length, so a prefill plus one step per generated token
    # replaces re-running the entire prefix for every token. Without this,
    # generating N tokens costs N full forwards over the whole window, i.e.
    # quadratic in N. Measured end to end through inference.py, 2 threads, 32
    # generated tokens after a 1024-token prompt: 74.2 s -> 3.45 s (21.5x) at
    # V8000 d512 L8.
    #
    # Semantics are unchanged: prefill consumes the same window
    # (ids[-MODEL_WINDOW:]) that the old re-forward path used, and the caller
    # re-prefills if the window would slide, which is what the old code did by
    # simply truncating. Training never calls these.
    #
    # A step is not bit-identical to a batched forward, and cannot be: torch
    # picks a different GEMM/SpMM kernel for a [1, d] input than a [T, d] one,
    # so the same row of an activation sums in a different order (~5e-07
    # relative, measured). Test the logits against a tolerance, never the text.
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def prefill(self, idx):
        """Run the whole prefix; returns (logits [B,1,V], states)."""
        if idx.shape[1] == 0:
            raise ValueError("prefill needs at least one token")
        x = self.emb(idx).to(torch.bfloat16)
        x = self.n0(x.float()).to(torch.bfloat16)
        states = []
        for b in self.blocks:
            x, st = b.prefill(x)
            states.append(st)
        x = self.nf(x.float())[:, -1:, :]
        return self.head(x.float()), states

    @torch.inference_mode()
    def step(self, idx, states):
        """One token: returns (logits [B,1,V], states) with states advanced.

        Every block must be stepped before the final norm and head: the loop
        body has to be the *whole* body. It briefly did not -- the norm, head
        and return were indented inside the loop, so this ran block 0 only and
        advanced the remaining states never. It is not a rounding bug: the
        result was uncorrelated with the equivalent full forward (~1.0 relative,
        against ~1e-7 for the fp32 kernel-selection noise a per-token step
        legitimately introduces). The equivalence tests in tests/test_last_token.py
        pin it, and they caught it when the loop body was briefly wrong.
        """
        if idx.shape[1] != 1:
            raise ValueError(f"step takes exactly one token, got {idx.shape[1]}")
        if len(states) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} states, got {len(states)}")
        x = self.emb(idx).to(torch.bfloat16)
        x = self.n0(x.float()).to(torch.bfloat16)
        for i, b in enumerate(self.blocks):
            x, states[i] = b.step(x, states[i])
        x = self.nf(x.float())
        return self.head(x.float()), states

    def compute_profile(self) -> dict:
        """Per-token multiply-accumulates actually executed, by layer class.

        Counterpart to ``RawrGraph.stats()``, which reports the *graph's*
        sparsity: how many of the ``vocab**2`` possible token-to-token edges
        exist. That number is real but it is not the model's sparsity, and
        quoting it as one is badly misleading -- at the 32M preset
        (vocab 8000, d 512, 8 layers, ffn 2.5) the graph reports 99.9 % while
        97.7 % of the arithmetic the model performs is *dense*, because the
        graph only drives the FFN and the LM head.

        Everything is measured by walking the built model, so this cannot
        drift from the architecture: the counts are the actual ``in_f *
        out_f`` of a dense ``FP8Linear`` and the actual ``out_f * K`` of a
        ``SparseLinear``.

        Keys (all per token; ``*_dense_equiv`` is what the same layers would
        cost with no sparsity at all):

        ==================  ==================================================
        key                 meaning
        ==================  ==================================================
        fp8_dense_mac       dense FP8Linear MAC (attention q/k/v/o, and the
                            SwiFFN projections on architecture='plain')
        fp32_dense_mac      dense MAC from precision='fp32', where the
                            projections are _DenseLinear rather than FP8Linear
        sparse_nnz_mac      nonzero MAC of every SparseLinear
        sparse_dense_mac    what those SparseLinears would cost dense
        dense_head_mac      a dense LM head (architecture='plain' uses a
                            plain nn.Linear, which is dense work too and must
                            not fall through the cracks)
        emb_dense_mac       embedding table entries, reported for context
                            only: a lookup reads rows, it does not multiply,
                            so it is deliberately NOT part of the sparsity
                            ratio below (it does dominate gradient and
                            optimizer memory)
        mac_per_token       dense MAC + sparse_nnz_mac
        dense_share         dense MAC / mac_per_token
        model_sparsity      1 - (nnz + dense) / (dense_equiv of every
                            projection), so 0 for a fully dense model
        trainable_values    total ``p.numel()`` with requires_grad
        ==================  ==================================================
        """
        fp8 = [(n, m) for n, m in self.named_modules() if isinstance(m, FP8Linear)]
        sp = [(n, m) for n, m in self.named_modules() if isinstance(m, SparseLinear)]
        # _DenseLinear is what _linear() builds under precision='fp32'. It is a
        # bare nn.Module wrapper, not an nn.Linear subclass, so it has to be
        # matched explicitly or the whole fp32 model reports zero attention and
        # FFN MACs and a wildly wrong dense_share.
        dl = [(n, m) for n, m in self.named_modules() if isinstance(m, _DenseLinear)]
        fp8_dense = sum(m.in_f * m.out_f for _, m in fp8)
        fp32_dense = sum(m.lin.in_features * m.lin.out_features for _, m in dl)
        sparse_nnz = sum(m.out_f * m.cols.shape[1] for _, m in sp)
        sparse_dense = sum(m.out_f * m.in_f for _, m in sp)
        # A head that is none of the above is a plain dense nn.Linear and is
        # real arithmetic, so it belongs in the dense bucket. (SparseLinear,
        # FP8Linear and _DenseLinear heads are already counted above; this must
        # not double count them, hence the isinstance checks.)
        dense_head = 0
        if not isinstance(self.head, (SparseLinear, FP8Linear, _DenseLinear)):
            dense_head = int(self.head.in_features * self.head.out_features)
        emb = 0
        w = getattr(self.emb, "weight", None)
        if w is not None:
            emb = int(w.numel())
        dense = fp8_dense + fp32_dense + dense_head
        mac = dense + sparse_nnz
        # Projections only -- see the emb_dense_mac note above.
        total_dense = sparse_dense + dense
        moe = [m for _, m in self.named_modules() if isinstance(m, SwiFFN_MoE)]
        return {
            "fp8_modules": len(fp8),
            "sparse_modules": len(sp),
            "moe_blocks": len(moe),
            "moe_experts": sum(m.num_experts for m in moe),
            "moe_top_k": self.cfg.num_experts_per_tok if self.cfg.is_moe else 0,
            "fp8_dense_mac": fp8_dense + fp32_dense,
            "fp32_modules": len(dl),
            "fp32_dense_mac": fp32_dense,
            "sparse_nnz_mac": sparse_nnz,
            "sparse_dense_mac": sparse_dense,
            "dense_head_mac": dense_head,
            "emb_dense_mac": emb,
            "mac_per_token": mac,
            "dense_share": (dense / mac) if mac else 0.0,
            "model_sparsity": (1.0 - mac / total_dense) if total_dense else 0.0,
            "trainable_values": sum(p.numel() for p in self.parameters()
                                    if p.requires_grad),
            "rawr_sparsity": self.cfg.rawr_sparsity,
        }


    def save_pretrained(self, out: Path):
        from safetensors.torch import save_file
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if self.cfg.architecture == "rawr" and self.rawr_graph is not None:
            self.cfg.rawr_graph_hash = self.rawr_graph.digest
            self.cfg.rawr_edge_count = len(self.rawr_graph.edges)
            from rawr_graph import save_graph

            save_graph(self.rawr_graph, out / "rawr_graph.json")
        if self.cfg.embedding_storage == "mmap":
            from embeddings import EMBEDDING_FILE

            emb = self.emb
            dest = out / EMBEDDING_FILE
            if getattr(emb, "path", None) is not None and Path(emb.path) == dest:
                emb.flush()
            else:
                # Chunked file-to-file copy; never holds two full tables.
                import numpy as np

                src = np.memmap(str(emb.path), dtype=np.float32, mode="r",
                                shape=(self.cfg.vocab_size, self.cfg.d_model)) \
                    if hasattr(emb, "path") else emb.weight.detach().cpu().float().numpy()
                tmp_e = dest.with_suffix(".dat.tmp")
                dst = np.memmap(str(tmp_e), dtype=np.float32, mode="w+",
                                shape=(self.cfg.vocab_size, self.cfg.d_model))
                for r0 in range(0, self.cfg.vocab_size, 1024):
                    r1 = min(self.cfg.vocab_size, r0 + 1024)
                    dst[r0:r1] = src[r0:r1]
                dst.flush()
                del dst, src
                os.replace(tmp_e, dest)
            sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
                  if not k.startswith("emb.")}
        else:
            sd = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        tmp = out / "model.safetensors.tmp"
        save_file(sd, str(tmp))
        os.replace(tmp, out / "model.safetensors")
        self.cfg.save(out / "config.json")

    @classmethod
    def from_pretrained(cls, d: Path, device: str = "cpu", architecture=None,
                        embedding_storage=None):
        from safetensors.torch import load_file
        d = Path(d)
        cfg = LinearConfig.load(d / "config.json")
        _check_legacy_tokenizer_file(d, cfg)
        if architecture is not None and architecture != cfg.architecture:
            raise ValueError(
                f"checkpoint architecture is {cfg.architecture!r}, "
                f"but {architecture!r} was requested; refusing to misinterpret "
                f"(rawr <-> plain weights are not interchangeable)")
        storage = embedding_storage or cfg.embedding_storage
        if storage not in ("ram", "mmap"):
            raise ValueError(f"unknown embedding_storage {storage!r}")
        graph = None
        if cfg.architecture == "rawr":
            from rawr_graph import load_graph

            gp = d / "rawr_graph.json"
            if not gp.exists():
                raise FileNotFoundError(
                    f"Rawr checkpoint {d} is missing rawr_graph.json")
            graph = load_graph(gp)
            if graph.vocab_size != cfg.vocab_size:
                raise ValueError(
                    f"Rawr graph vocab {graph.vocab_size} != config vocab {cfg.vocab_size}")
            if cfg.rawr_graph_hash and graph.digest != cfg.rawr_graph_hash:
                raise ValueError(
                    f"Rawr graph hash {graph.digest} != checkpoint record "
                    f"{cfg.rawr_graph_hash}; refusing to load mismatched graph")
        if storage == "mmap":
            from embeddings import EMBEDDING_FILE

            ep = d / EMBEDDING_FILE
            if not ep.exists():
                raise FileNotFoundError(
                    f"mmap checkpoint {d} is missing {EMBEDDING_FILE}")
            m = cls(cfg, rawr_graph=graph, emb_path=ep)
        else:
            m = cls(cfg, rawr_graph=graph)
        try:
            sd = load_file(str(d / "model.safetensors"), device=device)
        except RuntimeError as exc:
            raise RuntimeError(f"could not load checkpoint {d}: {exc}") from exc
        try:
            if storage == "mmap":
                missing, unexpected = m.load_state_dict(sd, strict=False)
                missing = set(missing)
                if unexpected:
                    raise RuntimeError(f"unexpected keys: {sorted(unexpected)}")
                # emb.weight lives in embeddings.dat, not the safetensors file.
                ok_missing = {k for k in missing if k.startswith("emb.")}
                if missing - ok_missing:
                    raise RuntimeError(
                        f"checkpoint {d} ({cfg.architecture}) missing keys: "
                        f"{sorted(missing - ok_missing)}")
            else:
                m.load_state_dict(sd, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint incompatible with config (arch={cfg.architecture} "
                f"fp8<->fp32 key change w8/sc vs weight?): {exc}"
            ) from exc
        # A rawr checkpoint loaded as plain (or vice versa) can never reach
        # here: key shapes/names differ and strict/explicit checks fail first.
        return m


# Named model-size presets: preset name -> dict(vocab, d, layers, heads, ffn_mult).
# Vocabulary is always the 256 byte values; width/depth carry the capacity.
# Effective fp32-equivalent params ~= 2*vocab*d + (5*layers+2)*d
#   + layers*(4*d*d + 3*d*int(d*ffn_mult)). FP8 per-tile scales add ~1-2% on top.
# Presets are added one per commit, largest first.
PRESETS: dict = {
    # ~1,024M params (1.024B target).
    "1B": {"vocab": 256, "d": 2048, "layers": 18, "heads": 16, "ffn_mult": 2.0},
    # ~508M params (512M target).
    "512M": {"vocab": 256, "d": 1536, "layers": 13, "heads": 12, "ffn_mult": 2.0},
    # ~260M params (256M target).
    "256M": {"vocab": 256, "d": 1024, "layers": 12, "heads": 8, "ffn_mult": 2.0},
    # ~132M params (128M target).
    "128M": {"vocab": 256, "d": 1024, "layers": 11, "heads": 8, "ffn_mult": 2.0},
    # ~65M params (64M target).
    "64M": {"vocab": 256, "d": 768, "layers": 9, "heads": 12, "ffn_mult": 2.0},
    # ~32M params (32M target, matches previous defaults).
    "32M": {"vocab": 256, "d": 512, "layers": 8, "heads": 8, "ffn_mult": 2.5},
    # ~16M params (16M target).
    "16M": {"vocab": 256, "d": 512, "layers": 4, "heads": 8, "ffn_mult": 2.5},
    # ~7.8M params (8M target).
    "8M": {"vocab": 256, "d": 448, "layers": 3, "heads": 7, "ffn_mult": 2.0},
    # ~4.2M params (4M target).
    "4M": {"vocab": 256, "d": 256, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~2.1M params (2M target).
    "2M": {"vocab": 256, "d": 256, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~1.05M params (1M target).
    "1M": {"vocab": 256, "d": 128, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~526K params (512K target).
    "512K": {"vocab": 256, "d": 128, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~256K params (256K target).
    "256K": {"vocab": 256, "d": 64, "layers": 6, "heads": 4, "ffn_mult": 2.0},
    # ~132K params (128K target).
    "128K": {"vocab": 256, "d": 64, "layers": 3, "heads": 4, "ffn_mult": 2.0},
    # ~67K params (64K target).
    "64K": {"vocab": 256, "d": 56, "layers": 2, "heads": 4, "ffn_mult": 2.0},
    # ~33K params (32K target).
    "32K": {"vocab": 256, "d": 32, "layers": 3, "heads": 2, "ffn_mult": 2.0},
    # ~16.6K params (16K target).
    "16K": {"vocab": 256, "d": 32, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~8.2K params (8K target).
    "8K": {"vocab": 256, "d": 24, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~4.2K params (4K target).
    "4K": {"vocab": 256, "d": 16, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~2.1K params (2K target).
    "2K": {"vocab": 256, "d": 12, "layers": 1, "heads": 2, "ffn_mult": 2.0},
    # ~1.08K params (1K target).
    "1K": {"vocab": 256, "d": 8, "layers": 1, "heads": 2, "ffn_mult": 2.0},
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


def build_model(args, tok, out, tokenizer_sha256="", dataset_fingerprint=""):
    """Assemble LinearConfig, the optional Rawr graph, and SmaulLinear from CLI args.

    Moved verbatim from train.main; train.py keeps the optimizer, stream,
    and loop. Needs tok for graph edges and out for the mmap table path.
    """
    from dataset import discover_files, iter_texts

    arch = getattr(args, "architecture", "rawr") or "rawr"
    storage = getattr(args, "embedding_storage", "ram") or "ram"
    if arch not in ("rawr", "plain"):
        raise ValueError(f"--architecture must be rawr/plain, got {arch!r}")
    if storage not in ("ram", "mmap"):
        raise ValueError(f"--embedding-storage must be ram/mmap, got {storage!r}")
    opt_name = getattr(args, "optimizer", "smaul") or "smaul"
    if arch != "rawr":
        # The whole rawr block below is skipped, so non-default rawr flags
        # would silently do nothing. Warn rather than raise: defaults flow
        # through here on every plain run and must stay quiet.
        for _flag, _v, _dflt in (
                ("--rawr-sparsity", getattr(args, "rawr_sparsity", 0.9), 0.9),
                ("--rawr-min-degree", getattr(args, "rawr_min_degree", 4), 4),
                ("--rawr-dict", getattr(args, "rawr_dict", None), None),
                ("--rawr-graph-out", getattr(args, "rawr_graph_out", None), None),
                ("--rawr-max-docs", getattr(args, "rawr_max_docs", 2000), 2000),
                ("--rawr-max-tokens-per-doc", getattr(args, "rawr_max_tokens_per_doc", 1024), 1024)):
            if _v != _dflt:
                print(f"[WARN] {_flag}={_v} has no effect with --architecture {arch}")
    if opt_name != "smaul":
        # Lion keeps fixed built-in betas and no narrow state: SmaulOpt-only
        # flags are validated above and then discarded. Same warn-not-raise.
        for _flag, _v, _dflt in (
                ("--beta-m", getattr(args, "beta_m", 0.9), 0.9),
                ("--beta-v", getattr(args, "beta_v", 0.999), 0.999),
                ("--epsilon", getattr(args, "epsilon", 1e-8), 1e-8),
                ("--state-dtype", getattr(args, "state_dtype", "bf16"), "bf16"),
                ("--grad-dtype", getattr(args, "grad_dtype", "bf16"), "bf16"),
                ("--factor-v/--no-factor-v", getattr(args, "factor_v", True), True)):
            if _v != _dflt:
                print(f"[WARN] {_flag}={_v} has no effect with --optimizer {opt_name}")
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
    _ne = getattr(args, "moe_experts", 1)
    _ne = 1 if _ne is None else int(_ne)
    _tk = getattr(args, "moe_top_k", 1)
    _tk = 1 if _tk is None else int(_tk)
    cfg = LinearConfig(vocab_size=args.vocab, d_model=args.d, n_layer=args.layers, n_heads=args.heads,
                       ffn_mult=getattr(args, "ffn_mult", 2.5),
                       precision=args.precision, tokenizer_sha256=tokenizer_sha256,
                       dataset_fingerprint=dataset_fingerprint,
                       architecture=arch, embedding_storage=storage,
                       rawr_sparsity=float(getattr(args, "rawr_sparsity", 0.9)),
                       rawr_min_degree=int(getattr(args, "rawr_min_degree", 4)),
                       is_moe=_ne > 1,
                       num_experts=_ne,
                       num_experts_per_tok=_tk,
                       moe_balance_weight=float(getattr(args, "moe_balance_weight", 0.01)))
    emb_path = (out / "embeddings.dat") if storage == "mmap" else None
    model = SmaulLinear(cfg, rawr_graph=rawr_graph, emb_path=emb_path)
    if arch == "rawr":
        # The graph stats printed above describe the token graph, not the
        # model. Print what the built model actually executes, so the headline
        # sparsity figure is not read as the model's. See
        # rawr_graph.print_model_compute.
        from rawr_graph import print_model_compute

        print_model_compute(model.compute_profile())
    return model

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
from kernel.fp8_tile import FP8Linear

@dataclass
class LinearConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layer: int = 8
    n_heads: int = 8
    ffn_mult: float = 2.5
    eps: float = 1e-6
    tile: int = 64
    is_moe: bool = False
    num_experts: int = 1
    num_experts_per_tok: int = 1
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
        if self.is_moe and self.architecture == "rawr":
            raise ValueError("MoE upcycling is only supported with architecture='plain'")
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

class _DenseLinear(nn.Module):
    """Plain-FP32 linear with FP8Linear-compatible dtype behavior.

    Computes in float32 and returns the input dtype, so the surrounding model
    code (bf16 activations, fp32 attention core) is identical in both modes.
    """

    def __init__(self, in_f: int, out_f: int, bias: bool = False):
        super().__init__()
        self.lin = nn.Linear(in_f, out_f, bias=bias)

    def forward(self, x):
        return self.lin(x.float()).to(x.dtype if x.is_floating_point() else torch.float32)

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

        Keyed on the buffer identity so ``.to(device)`` (which gives a new
        ``cols`` tensor) is picked up rather than silently reusing a stale
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
    def __init__(self, cfg: LinearConfig):
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
        self.experts = nn.ModuleList([SwiFFN(cfg) for _ in range(cfg.num_experts)])
        self.gate = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)

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
        denom = topv.sum(-1, keepdim=True)
        # Guard tiny denominators (would explode weights); fall back to uniform.
        tiny = denom.squeeze(-1) < 1e-6
        topv = torch.where(tiny.unsqueeze(-1), torch.full_like(topv, 1.0 / self.top_k),
                           topv / denom.clamp_min(1e-9))
        out = torch.zeros_like(x.float())
        for e, expert in enumerate(self.experts):
            w = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            m = (w.squeeze(-1) > 0)
            if m.any():
                xm = x[m].contiguous()
                out[m] += expert(xm).float() * w[m]
        return out.to(x.dtype if x.is_floating_point() else torch.float32)

class Block(nn.Module):
    def __init__(self, cfg, rawr_graph=None):
        super().__init__()
        d = cfg.d_model
        self.n1 = RMSNorm(d, cfg.eps)
        self.att = LinearAttention(cfg)
        self.n2 = RMSNorm(d, cfg.eps)
        self.n3 = RMSNorm(d, cfg.eps)
        if cfg.is_moe:
            self.ffn = SwiFFN_MoE(cfg)
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
        return {
            "fp8_modules": len(fp8),
            "sparse_modules": len(sp),
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

# Run this on your machine (where torch is installed) before training with the
# updated RWKV_CMix_MoE. Confirms the sparse-gather rewrite == old dense-mask math.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rwkv_x_core import RWKVXConfig, RWKV_CMix_MoE  # no dots   

torch.manual_seed(0)
cfg = RWKVXConfig(n_embd=128, num_experts=8, num_experts_per_tok=2, is_moe=True, n_layer=4)
moe = RWKV_CMix_MoE(cfg, 0)
x = torch.randn(2, 64, 128)

def dense_ref(moe, x, x_prev_last=None):
    B, T, C = x.shape
    logits = moe.gate(x)
    top_val, top_idx = torch.topk(logits, k=moe.top_k, dim=-1)
    top_w = torch.softmax(top_val, dim=-1)
    out = torch.zeros_like(x)
    for e_id, expert in enumerate(moe.experts):
        mask = (top_idx == e_id)
        if not torch.any(mask):
            continue
        e_out, _ = expert(x, x_prev_last)
        weight = torch.where(mask, top_w, torch.zeros_like(top_w)).sum(dim=-1, keepdim=True)
        out = out + e_out * weight
    return out

new_out, _ = moe(x)
ref_out = dense_ref(moe, x)
print("max abs diff:", (new_out - ref_out).abs().max().item())  # should be ~0 (fp rounding only)

import time
x3 = torch.randn(2, 512, 128)
t0 = time.perf_counter()
for _ in range(20): moe(x3)
t_sparse = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(20): dense_ref(moe, x3)
t_dense = time.perf_counter() - t0
print(f"sparse: {t_sparse:.3f}s  dense: {t_dense:.3f}s  speedup: {t_dense/t_sparse:.2f}x")

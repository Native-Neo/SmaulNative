import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rwkv_x_core import RWKVXConfig, RWKV_CMix_MoE


def test_sparse_moe_matches_dense_reference():
    torch.manual_seed(0)
    cfg = RWKVXConfig(n_embd=128, num_experts=8, num_experts_per_tok=2, is_moe=True, n_layer=4)
    moe = RWKV_CMix_MoE(cfg, 0)
    x = torch.randn(2, 64, 128, requires_grad=True)

    def dense_ref(moe, x, x_prev_last=None):
        logits = moe.gate(x)
        top_val, top_idx = torch.topk(logits, k=moe.top_k, dim=-1)
        top_w = torch.softmax(top_val, dim=-1)
        out = torch.zeros_like(x)
        prev = torch.cat([x_prev_last.unsqueeze(1) if x_prev_last is not None else torch.zeros_like(x[:, :1]), x[:, :-1]], 1)
        for e_id, expert in enumerate(moe.experts):
            weight = torch.where(top_idx == e_id, top_w, torch.zeros_like(top_w)).sum(dim=-1, keepdim=True)
            out = out + expert.forward_selected(x, prev) * weight
        return out

    new_out, _ = moe(x)
    ref_out = dense_ref(moe, x)
    assert (new_out - ref_out).abs().max().item() < 1e-5
    new_out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(expert.key.weight.grad is not None for expert in moe.experts)
    assert moe.gate.weight.grad is not None

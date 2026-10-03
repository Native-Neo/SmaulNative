import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from smaul_linear import LinearConfig, SwiFFN_MoE


def test_sparse_moe_matches_dense_reference():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=64, d_model=64, n_layer=2, n_heads=4, is_moe=True, num_experts=4, num_experts_per_tok=2)
    moe = SwiFFN_MoE(cfg)
    x = torch.randn(2, 16, 64, requires_grad=True)

    def dense_ref(moe, x):
        prob = torch.softmax(moe.gate(x.float()), -1)
        topv, topi = torch.topk(prob, moe.top_k, -1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x.float())
        for e, expert in enumerate(moe.experts):
            w = torch.where(topi == e, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            out = out + expert(x).float() * w
        return out

    new_out = moe(x)
    ref_out = dense_ref(moe, x)
    assert (new_out.float() - ref_out).abs().max().item() < 1e-5
    new_out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert moe.gate.weight.grad is not None

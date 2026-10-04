import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from model import LinearConfig, RawrFFN, SmaulLinear, SwiFFN, SwiFFN_MoE
from rawr_graph import fallback_graph


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


def _rawr_moe_cfg(**over):
    kw = dict(vocab_size=64, d_model=32, n_layer=1, n_heads=2, precision="fp32",
              architecture="rawr", is_moe=True, num_experts=4,
              num_experts_per_tok=2, rawr_sparsity=0.5, rawr_min_degree=2)
    kw.update(over)
    return LinearConfig(**kw)


def test_rawr_moe_uses_sparse_experts():
    torch.manual_seed(0)
    cfg = _rawr_moe_cfg()
    moe = SwiFFN_MoE(cfg, fallback_graph(64, 2))
    assert all(isinstance(e, RawrFFN) for e in moe.experts)
    x = torch.randn(2, 8, 32)
    out = moe(x)
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_rawr_moe_without_a_graph_falls_back():
    torch.manual_seed(0)
    moe = SwiFFN_MoE(_rawr_moe_cfg())
    assert len(moe.experts) == 4
    assert torch.isfinite(moe(torch.randn(1, 4, 32))).all()


def test_plain_moe_uses_dense_experts():
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                       is_moe=True, num_experts=3, num_experts_per_tok=1)
    moe = SwiFFN_MoE(cfg)
    assert all(isinstance(e, SwiFFN) for e in moe.experts)


def test_router_output_is_a_valid_distribution():
    torch.manual_seed(1)
    moe = SwiFFN_MoE(_rawr_moe_cfg(), fallback_graph(64, 2))
    x = torch.randn(2, 8, 32)
    moe(x)
    prob = moe.last_prob
    assert prob.shape == (2, 8, 4)
    assert torch.allclose(prob.sum(-1), torch.ones(2, 8), atol=1e-5)
    assert bool(((prob >= 0) & (prob <= 1)).all())


def test_top_k_selection_picks_exactly_k_per_token():
    torch.manual_seed(2)
    moe = SwiFFN_MoE(_rawr_moe_cfg(num_experts_per_tok=2), fallback_graph(64, 2))
    moe(torch.randn(3, 10, 32))
    assert moe.last_topi.shape == (3, 10, 2)
    for tok in moe.last_topi.reshape(-1, 2).tolist():
        assert len(set(tok)) == 2  # distinct experts


def test_only_selected_experts_execute():
    torch.manual_seed(3)
    moe = SwiFFN_MoE(_rawr_moe_cfg(), fallback_graph(64, 2))
    calls = [0] * 4
    for e, expert in enumerate(moe.experts):
        orig = expert.forward
        def spy(x, _e=e, _orig=orig):
            calls[_e] += 1
            return _orig(x)
        expert.forward = spy
    moe(torch.randn(2, 8, 32))
    assert calls == [int(v) for v in moe.last_executed]
    assert sum(calls) <= 4 and sum(calls) >= 1
    # Tokens are 16 with top-2: at most all experts, at least one.
    assert any(c == 0 for c in calls) or sum(moe.expert_token_counts) == 32


def test_selected_experts_receive_gradients():
    torch.manual_seed(4)
    moe = SwiFFN_MoE(_rawr_moe_cfg(), fallback_graph(64, 2))
    x = torch.randn(2, 8, 32)
    moe(x).sum().backward()
    assert moe.gate.weight.grad is not None and torch.isfinite(moe.gate.weight.grad).all()
    assert any(moe.last_executed), "no expert executed, nothing to assert"
    for e, expert in enumerate(moe.experts):
        grads = [p.grad for p in expert.parameters() if p.grad is not None]
        if moe.last_executed[e]:
            assert grads and all(torch.isfinite(g).all() for g in grads)


def test_routing_stats_report_usage_and_inactive():
    torch.manual_seed(5)
    moe = SwiFFN_MoE(_rawr_moe_cfg(), fallback_graph(64, 2))
    moe(torch.randn(2, 8, 32))
    st = moe.routing_stats()
    assert st["num_experts"] == 4 and st["top_k"] == 2
    assert sum(st["token_counts"]) == 2 * 8 * 2
    assert abs(sum(st["usage_frac"]) - 1.0) < 1e-6
    assert st["inactive_experts"] == sum(1 for c in st["token_counts"] if c == 0)


def test_balance_loss_is_nonnegative_and_finite():
    torch.manual_seed(6)
    moe = SwiFFN_MoE(_rawr_moe_cfg(), fallback_graph(64, 2))
    moe(torch.randn(2, 8, 32))
    aux = moe.balance_loss(moe.last_prob)
    assert torch.isfinite(aux) and float(aux) >= 1.0 - 1e-6


@pytest.mark.parametrize("n,k", [(1, 1), (2, 1), (4, 2), (8, 3)])
def test_configurable_expert_counts_and_top_k(n, k):
    torch.manual_seed(7)
    moe = SwiFFN_MoE(_rawr_moe_cfg(num_experts=n, num_experts_per_tok=k),
                     fallback_graph(64, 2))
    out = moe(torch.randn(1, 6, 32))
    assert out.shape == (1, 6, 32)
    assert moe.routing_stats()["top_k"] == k
    assert sum(moe.expert_token_counts) == 6 * k


def test_misconfigured_moe_fails_fast():
    with pytest.raises(ValueError):
        SwiFFN_MoE(_rawr_moe_cfg(num_experts=2, num_experts_per_tok=3),
                   fallback_graph(64, 2))


def test_rawr_moe_checkpoint_round_trip(tmp_path):
    torch.manual_seed(8)
    m = SmaulLinear(_rawr_moe_cfg(), rawr_graph=fallback_graph(64, 2))
    m.save_pretrained(tmp_path)
    m2 = SmaulLinear.from_pretrained(tmp_path)
    assert m2.cfg.is_moe and m2.cfg.num_experts == 4
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        assert torch.equal(m(ids)[0], m2(ids)[0])

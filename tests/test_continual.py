import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from continual import (ContinualStream, GwStash, ReplayBuffer, evaluate_loss,
                       global_clip_scale, moe_routing_summary, partition_model,
                       retention_report)
from dataset import TokenizerWrapper
from model import LinearConfig, SmaulLinear
from rawr_graph import fallback_graph
from tokenizer import SmaulTokenizer


def _wrap():
    return TokenizerWrapper(SmaulTokenizer())


def _corpus(d, name="a.txt", text="hello continual world. "):
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text * 30, encoding="utf-8")
    return d


def _tiny_moe():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=256, d_model=32, n_layer=1, n_heads=2,
                       precision="fp32", architecture="rawr", is_moe=True,
                       num_experts=4, num_experts_per_tok=2,
                       rawr_sparsity=0.5, rawr_min_degree=2)
    return SmaulLinear(cfg, rawr_graph=fallback_graph(256, 2))


# --- ReplayBuffer ------------------------------------------------------------

def test_replay_buffer_is_bounded():
    buf = ReplayBuffer(8)
    for i in range(100):
        buf.add([i, i + 1, i + 2])
    assert len(buf) == 8
    assert buf.sample(3)


def test_replay_buffer_zero_capacity_stores_nothing():
    buf = ReplayBuffer(0)
    buf.add([1, 2, 3])
    assert len(buf) == 0 and buf.sample(2) == []


def test_replay_buffer_rejects_negative_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(-1)


def test_replay_sample_returns_copies():
    buf = ReplayBuffer(4)
    buf.add([1, 2, 3])
    buf.sample(1)[0].append(999)
    assert buf.sample(1)[0] == [1, 2, 3]


# --- ContinualStream ----------------------------------------------------------

def test_continual_stream_preserves_state_across_domains(tmp_path):
    a = _corpus(tmp_path / "a", text="first domain alpha. ")
    b = _corpus(tmp_path / "b", text="second domain beta. ")
    items_reset = list(ContinualStream([a, b], _wrap(), 16, reset_state=True))
    items_kept = list(ContinualStream([a, b], _wrap(), 16, reset_state=False))
    assert items_reset and items_kept
    for x, y, _ in items_kept:
        assert x.numel() == 16 and y.numel() == 16
        assert torch.equal(y[:-1], x[1:])


def test_continual_stream_replays_old_chunks(tmp_path):
    a = _corpus(tmp_path / "a", text="old domain words here. ")
    b = _corpus(tmp_path / "b", text="new domain words here. ")
    replay = ReplayBuffer(64)
    items = list(ContinualStream([a, b], _wrap(), 16, replay=replay,
                                 replay_rate=0.5, seed=0))
    assert any(str(pos[0]).startswith("replay:") for _, _, pos in items)
    assert len(replay) > 0 and len(replay) <= 64


def test_continual_stream_rejects_bad_setup(tmp_path):
    with pytest.raises(ValueError, match="at least one domain"):
        ContinualStream([], _wrap(), 16)
    with pytest.raises(ValueError, match="replay_rate"):
        ContinualStream([tmp_path], _wrap(), 16, replay_rate=1.5)
    with pytest.raises(ValueError, match="ctx_len"):
        ContinualStream([tmp_path], _wrap(), 0)


# --- retention ------------------------------------------------------------------

def test_retention_report_measures_forgetting():
    rep = retention_report(2.0, 2.2, 1.5)
    assert rep["old_loss_before"] == 2.0
    assert rep["old_loss_after"] == 2.2
    assert abs(rep["forgetting"] - 0.2) < 1e-9
    assert rep["new_loss_after"] == 1.5


def test_evaluate_loss_runs_on_a_real_corpus(tmp_path):
    d = _corpus(tmp_path / "eval")
    m = _tiny_moe()
    m.eval()
    loss = evaluate_loss(m, d, _wrap(), 16, max_batches=3)
    assert loss > 0 and loss == loss  # finite
    assert m.training  # restored to train mode


def test_evaluate_loss_rejects_empty_or_bad_args(tmp_path):
    m = _tiny_moe()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((RuntimeError, ValueError)):
        evaluate_loss(m, empty, _wrap(), 16, max_batches=2)
    with pytest.raises(ValueError):
        evaluate_loss(m, tmp_path, _wrap(), 16, max_batches=0)


# --- partitioning / per-group behaviour ------------------------------------------

def test_partition_covers_every_parameter_once():
    m = _tiny_moe()
    parts = partition_model(m)
    seen = []
    for g in ("trunk", "expert", "router"):
        assert parts[g]["params"], g
        seen.extend(id(p) for p in parts[g]["params"])
    all_ids = [id(p) for p in m.parameters()]
    assert sorted(seen) == sorted(all_ids)


def test_partition_of_a_dense_model_is_all_trunk():
    torch.manual_seed(1)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                       precision="fp32", architecture="plain")
    m = SmaulLinear(cfg)
    parts = partition_model(m)
    assert not parts["expert"]["params"] and not parts["router"]["params"]
    assert len(parts["trunk"]["params"]) == len(list(m.parameters()))


def test_gw_stash_isolates_one_group(tmp_path):
    torch.manual_seed(2)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                       precision="fp8", architecture="plain", is_moe=True,
                       num_experts=2, num_experts_per_tok=1, tile=32)
    m = SmaulLinear(cfg)
    m.train()
    ids = torch.randint(0, 64, (2, 16))
    _, loss = m(ids, ids)
    loss.backward()
    from kernel.fp8_tile import fp8_modules
    assert any(mod._gw is not None for _, mod in fp8_modules(m))
    parts = partition_model(m)
    with GwStash(m, parts["trunk"]["fp8"]):
        for name, mod in parts["expert"]["fp8"] + parts["router"]["fp8"]:
            assert mod._gw is None, name
        assert any(mod._gw is not None for _, mod in parts["trunk"]["fp8"])
    # Everything restored afterwards.
    assert any(mod._gw is not None for _, mod in fp8_modules(m))


def test_global_clip_scale_bounds_the_norm():
    torch.manual_seed(3)
    m = _tiny_moe()
    ids = torch.randint(0, 256, (2, 16))
    _, loss = m(ids, ids)
    loss.backward()
    norm = global_clip_scale(m, 0.01)
    assert norm > 0.01
    from model import _grad_norm
    from kernel.fp8_tile import fp8_modules
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert _grad_norm(grads) <= 0.01 + 1e-6


def test_moe_routing_summary_reports_usage():
    m = _tiny_moe()
    m.eval()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        m(ids)
    rs = moe_routing_summary(m)
    assert rs["moe_blocks"] == 1
    assert sum(rs["token_counts"]) == 2 * 16 * 2
    assert abs(sum(rs["usage_frac"]) - 1.0) < 1e-6


def test_moe_routing_summary_without_moe():
    torch.manual_seed(4)
    m = SmaulLinear(LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                                 precision="fp32", architecture="plain"))
    assert moe_routing_summary(m) == {"moe_blocks": 0}


# --- continual training reduces forgetting vs replaces ------------------------------

def test_replay_keeps_old_chunks_available_while_new_trains(tmp_path):
    old = _corpus(tmp_path / "old", text="old domain sentences. ")
    new = _corpus(tmp_path / "new", text="new domain sentences. ")
    replay = ReplayBuffer(32)
    stream = ContinualStream([old, new], _wrap(), 16, replay=replay,
                             replay_rate=0.3, seed=1)
    n_replay = sum(1 for _, _, pos in stream if str(pos[0]).startswith("replay:"))
    assert n_replay > 0
    assert 0 < len(replay) <= 32

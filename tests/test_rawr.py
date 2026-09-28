"""Focused tests for the experimental Rawr architecture (spec items 1-12).

Covers: embedding ram/mmap equivalence (+batches), deterministic graph, id
validity, fallback connectivity, rawr/plain forward, checkpoint roundtrips
(+plain legacy compat), tiny training runs, and all four arch/storage combos.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from embeddings import MmapEmbedding, RamEmbedding
from rawr_graph import (RawrGraph, build_graph, fallback_graph, hidden_cols,
                         load_graph, save_graph)
from smaul_linear import LinearConfig, SmaulLinear, SparseLinear
from tokenizer import SmaulTokenizer


def _mini_tokenizer():
    data = {
        "vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
                  "hello": 4, "world": 5, " ": 6, "!": 7, "namaste": 8,
                  "shabd": 9, "12": 10, "def": 11},
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": ["<cap>", "<upper>"],
        "unk_id": 1,
        "stats": {"vocab_size": 12},
    }
    return SmaulTokenizer(data)


def _tiny_cfg(arch="plain", storage="ram", vocab=48, d=16):
    return LinearConfig(vocab_size=vocab, d_model=d, n_layer=1, n_heads=2,
                        precision="fp32", architecture=arch,
                        embedding_storage=storage, rawr_sparsity=0.5,
                        rawr_min_degree=2)


# 1. RAM and mmap embedding lookup numerical equivalence.
def test_embedding_ram_mmap_equivalence(tmp_path):
    torch.manual_seed(0)
    ram = RamEmbedding(48, 16)
    torch.manual_seed(0)
    mm = MmapEmbedding(48, 16, tmp_path / "e.dat")
    with torch.no_grad():
        mm.weight.copy_(ram.weight)
    ids = torch.tensor([0, 5, 47, 12])
    assert torch.equal(ram(ids), mm(ids))


# 2. RAM and mmap embeddings with batches.
def test_embedding_ram_mmap_batches(tmp_path):
    torch.manual_seed(1)
    ram = RamEmbedding(48, 16)
    torch.manual_seed(1)
    mm = MmapEmbedding(48, 16, tmp_path / "e.dat")
    with torch.no_grad():
        mm.weight.copy_(ram.weight)
    ids = torch.randint(0, 48, (4, 10))
    assert torch.equal(ram(ids), mm(ids))
    # mmap file holds exactly vocab*d float32, nothing more.
    assert (tmp_path / "e.dat").stat().st_size == 48 * 16 * 4


# 3. Deterministic Rawr graph generation.
def test_graph_deterministic():
    tok = _mini_tokenizer()
    kw = dict(corpus_texts=["hello world", "namaste shabd 12", "def hello"],
              dict_words=None, window=1, min_degree=2)
    g1 = build_graph(tok, **kw)
    g2 = build_graph(tok, **kw)
    assert g1.digest == g2.digest
    assert g1.edges == g2.edges


# 4. No invalid token IDs in the graph.
def test_graph_no_invalid_ids():
    tok = _mini_tokenizer()
    v = tok.get_vocab_size()
    g = build_graph(tok, corpus_texts=["hello world !", "namaste 12 def"],
                    window=1, min_degree=2)
    for a, b in g.edges:
        assert 0 <= a < v and 0 <= b < v, (a, b)


# 5. Fallback connectivity (unseen tokens stay reachable).
def test_graph_fallback_connectivity():
    tok = _mini_tokenizer()
    v = tok.get_vocab_size()
    # Empty corpus + empty dict: fallbacks alone must satisfy min_degree.
    g = build_graph(tok, corpus_texts=[], dict_words=[], window=1, min_degree=3)
    deg = g.out_degree()
    assert min(deg) >= 3, deg
    # Fallback-only graph also deterministic and valid.
    f = fallback_graph(v, 3)
    assert min(f.out_degree()) >= 3
    # hidden_cols works even at high sparsity (ring fallback fills rows).
    cols = hidden_cols(8, 6, g, sparsity=0.9, min_per_row=1)
    assert cols.shape == (8, 1)
    assert bool(((cols >= 0) & (cols < 6)).all())


# 6. Rawr forward pass.
def test_rawr_forward():
    torch.manual_seed(0)
    m = SmaulLinear(_tiny_cfg("rawr", "ram"))
    m.eval()
    ids = torch.randint(0, 48, (2, 8))
    logits, loss = m(ids, ids)
    assert logits.shape == (2, 8, 48)
    assert torch.isfinite(loss)


# 7. Plain forward pass.
def test_plain_forward():
    torch.manual_seed(0)
    m = SmaulLinear(_tiny_cfg("plain", "ram"))
    m.eval()
    ids = torch.randint(0, 48, (2, 8))
    logits, loss = m(ids, ids)
    assert logits.shape == (2, 8, 48)
    assert torch.isfinite(loss)


# 8. Rawr checkpoint save/load.
def test_rawr_checkpoint(tmp_path):
    torch.manual_seed(0)
    m = SmaulLinear(_tiny_cfg("rawr", "ram"))
    m.save_pretrained(tmp_path)
    assert (tmp_path / "rawr_graph.json").exists()
    m2 = SmaulLinear.from_pretrained(tmp_path)
    assert m2.cfg.architecture == "rawr"
    ids = torch.randint(0, 48, (1, 6))
    with torch.no_grad():
        a, _ = m(ids)
        b, _ = m2(ids)
    assert torch.equal(a, b)


# 9. Plain checkpoint compatibility (legacy configs without arch keys).
def test_plain_checkpoint_compat(tmp_path):
    torch.manual_seed(0)
    m = SmaulLinear(_tiny_cfg("plain", "ram"))
    m.save_pretrained(tmp_path)
    # Strip the new keys to simulate a pre-Rawr checkpoint file.
    cfg_path = tmp_path / "config.json"
    data = json.loads(cfg_path.read_text())
    for k in ("architecture", "embedding_storage", "rawr_sparsity",
              "rawr_min_degree", "rawr_graph_hash", "rawr_edge_count"):
        data.pop(k, None)
    cfg_path.write_text(json.dumps(data))
    m2 = SmaulLinear.from_pretrained(tmp_path)
    assert m2.cfg.architecture == "plain"
    assert m2.cfg.embedding_storage == "ram"
    ids = torch.randint(0, 48, (1, 6))
    with torch.no_grad():
        assert torch.equal(m(ids)[0], m2(ids)[0])
    # Cross-architecture loads must fail clearly, not silently.
    with pytest.raises(ValueError, match="architecture"):
        SmaulLinear.from_pretrained(tmp_path, architecture="rawr")


def _tiny_train_step(arch, storage, tmp_path):
    from train import Lion

    torch.manual_seed(0)
    cfg = _tiny_cfg(arch, storage)
    kw = {"emb_path": tmp_path / f"emb_{arch}_{storage}.dat"} if storage == "mmap" else {}
    m = SmaulLinear(cfg, **kw)
    m.train()
    opt = Lion(list(m.parameters()), lr=1e-4)
    ids = torch.randint(0, 48, (2, 8))
    before = [p.detach().clone() for p in m.parameters() if p.requires_grad]
    for _ in range(3):
        opt.zero_grad(m)
        _, loss = m(ids, ids)
        assert torch.isfinite(loss)
        loss.backward()
        opt.step(m)
    _, loss = m(ids, ids)
    assert torch.isfinite(loss)
    changed = any(not torch.equal(a, b.detach())
                  for a, b in zip(before, [p for p in m.parameters() if p.requires_grad]))
    assert changed, "training step did not update parameters"
    return float(loss.detach())


# 10. Tiny Rawr training run.
def test_tiny_rawr_train(tmp_path):
    _tiny_train_step("rawr", "ram", tmp_path)


# 11. Tiny plain training run.
def test_tiny_plain_train(tmp_path):
    _tiny_train_step("plain", "ram", tmp_path)


# 12. All four architecture/storage combinations.
@pytest.mark.parametrize("arch,storage", [("rawr", "ram"), ("rawr", "mmap"),
                                          ("plain", "ram"), ("plain", "mmap")])
def test_all_four_combos(tmp_path, arch, storage):
    torch.manual_seed(0)
    cfg = _tiny_cfg(arch, storage)
    kw = {"emb_path": tmp_path / f"e_{arch}_{storage}.dat"} if storage == "mmap" else {}
    m = SmaulLinear(cfg, **kw)
    m.save_pretrained(tmp_path / f"ckpt_{arch}_{storage}")
    m2 = SmaulLinear.from_pretrained(tmp_path / f"ckpt_{arch}_{storage}")
    assert (m2.cfg.architecture, m2.cfg.embedding_storage) == (arch, storage)
    ids = torch.randint(0, 48, (2, 8))
    m2.eval()
    logits, loss = m2(ids, ids)
    assert logits.shape == (2, 8, 48) and torch.isfinite(loss)
    _tiny_train_step(arch, storage, tmp_path)


def test_graph_save_load_roundtrip(tmp_path):
    tok = _mini_tokenizer()
    g = build_graph(tok, corpus_texts=["hello world"], window=1, min_degree=2)
    save_graph(g, tmp_path / "g.json")
    g2 = load_graph(tmp_path / "g.json")
    assert g2.edges == g.edges and g2.digest == g.digest


# ---------------------------------------------------------------------------
# hidden_cols: the column selection is what an existing Rawr checkpoint is
# interpreted through, so the optimized selection must stay bit-identical to
# the original "score every j, sort, truncate" implementation.
# ---------------------------------------------------------------------------

def _hidden_cols_reference(out_f, in_f, graph, sparsity, min_per_row=1):
    """The pre-optimization implementation, verbatim, as the oracle."""
    v = graph.vocab_size
    directed = graph.directed_set()
    k = int(round(in_f * (1.0 - sparsity)))
    k = max(min_per_row, min(in_f, k))
    cols = []
    for i in range(out_f):
        vi = i % v
        scored = []
        for j in range(in_f):
            s = 1 if (vi, j % v) in directed else 0
            scored.append((-s, abs(i - j), j))
        scored.sort()
        cols.append([j for _, _, j in scored[:k]])
    return torch.tensor(cols, dtype=torch.long)


_GRAPHS = [
    fallback_graph(16, 4),
    fallback_graph(64, 4),
    fallback_graph(7, 3),                       # vocab < in_f in some shapes
    RawrGraph(vocab_size=5, edges=[(0, 1), (1, 2), (2, 2), (3, 4)]),
    RawrGraph(vocab_size=64, edges=[(0, 0), (5, 9), (63, 1), (30, 30), (2, 61)]),
]


@pytest.mark.parametrize("gi", range(len(_GRAPHS)))
def test_hidden_cols_matches_full_sort_reference(gi):
    g = _GRAPHS[gi]
    for out_f, in_f in [(1, 1), (4, 8), (8, 4), (16, 16), (33, 7), (64, 32),
                        (9, 9), (20, 5), (40, 3)]:
        for sparsity in (0.0, 0.5, 0.9, 0.99):
            for min_per_row in (1, 2, 4, 8):
                want = _hidden_cols_reference(out_f, in_f, g, sparsity, min_per_row)
                got = hidden_cols(out_f, in_f, g, sparsity, min_per_row)
                assert got.shape == want.shape, (gi, out_f, in_f, sparsity, min_per_row)
                assert torch.equal(got, want), (gi, out_f, in_f, sparsity, min_per_row)


def test_hidden_cols_row_length_is_uniform_and_bounded():
    """min_per_row > in_f must not produce ragged rows."""
    g = fallback_graph(32, 4)
    c = hidden_cols(16, 8, g, 0.5, 99)
    assert c.shape == (16, 8)
    assert c.shape[1] <= 8
    for row in c:
        assert len(set(row.tolist())) == row.numel()   # no duplicate columns


def test_hidden_cols_rejects_bad_arguments():
    g = fallback_graph(16, 4)
    with pytest.raises(ValueError):
        hidden_cols(0, 8, g, 0.5, 1)
    with pytest.raises(ValueError):
        hidden_cols(8, 0, g, 0.5, 1)
    with pytest.raises(ValueError):
        hidden_cols(8, 8, g, 1.0, 1)
    with pytest.raises(ValueError):
        hidden_cols(8, 8, g, -0.1, 1)
    with pytest.raises(ValueError):
        hidden_cols(8, 8, g, 0.5, 0)


def test_rawr_gate_and_up_share_one_cols_tensor():
    """Identical shapes must not pay for the derivation twice."""
    g = fallback_graph(64, 4)
    cfg = LinearConfig(vocab_size=64, d_model=32, n_layer=1, n_heads=2,
                       architecture="rawr", rawr_sparsity=0.9)
    m = SmaulLinear(cfg, rawr_graph=g)
    assert m.blocks[0].ffn.gate.cols is m.blocks[0].ffn.up.cols
    assert torch.equal(m.blocks[0].ffn.gate.cols, m.blocks[0].ffn.up.cols)
    # Still independent state/parameters, and still a real forward.
    assert m.blocks[0].ffn.gate.values is not m.blocks[0].ffn.up.values
    x = torch.randint(0, 64, (2, 8))
    _, loss = m(x, x)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# SparseLinear gradients. The audit found no gradient check for
# _SparseLinearFn, which let a total error in d/dvalues (the gather was
# dropped from the multiplicand, leaving dout*values) pass unnoticed because
# the loss still went down. All three gradients are pinned to a dense
# reference here.
# ---------------------------------------------------------------------------

def _dense_reference(sl, x):
    """out = x @ W.T with W nonzero only on the cols support."""
    W = torch.zeros(sl.out_f, sl.in_f)
    W.scatter_(1, sl.cols, sl.values.detach())
    return x @ W.t(), W


@pytest.mark.parametrize("rows,out_f,in_f,K", [
    (2, 4, 8, 3), (7, 5, 3, 2), (16, 33, 17, 5), (128, 320, 256, 26),
])
def test_sparse_linear_gradients_match_dense_reference(rows, out_f, in_f, K):
    torch.manual_seed(rows)
    sl = SparseLinear(in_f, out_f,
                      torch.stack([torch.randperm(in_f)[:K] for _ in range(out_f)]))
    X = torch.randn(rows, in_f)
    G = torch.randn(rows, out_f)

    xr = X.clone().requires_grad_(True)
    yr, W = _dense_reference(sl, xr)
    yr.backward(G)

    xn = X.clone().requires_grad_(True)
    yn = sl(xn)
    yn.backward(G)

    scale = max(float(xr.grad.abs().max()), 1.0)
    assert float((yn - yr).abs().max()) / scale < 1e-5
    assert float((xn.grad - xr.grad).abs().max()) / scale < 1e-5
    # d/dvalues is the dense grad restricted to the support, not the whole one.
    W2 = torch.zeros(out_f, in_f, requires_grad=True)
    with torch.no_grad():
        W2.copy_(W)
    (X @ W2.t()).backward(G)
    support = torch.arange(out_f)[:, None].expand_as(sl.cols)
    gv_ref = W2.grad[support, sl.cols]
    assert sl.values.grad.shape == (out_f, K)
    assert float((sl.values.grad - gv_ref).abs().max()) / max(
        1e-30, float(gv_ref.abs().max())) < 1e-5


def test_sparse_linear_backward_covers_every_output_row():
    """A dropped gather or a short block must not leave rows un-updated."""
    torch.manual_seed(7)
    out_f, in_f, K = 40, 24, 5
    sl = SparseLinear(in_f, out_f,
                      torch.stack([torch.randperm(in_f)[:K] for _ in range(out_f)]))
    x = torch.randn(9, in_f, requires_grad=True)
    sl(x).sum().backward()
    # Every output row contributes dout = 1, so every grad_v is the column sum.
    expect = x.detach()[:, sl.cols.reshape(-1)].view(9, out_f, K).sum(0)
    assert torch.allclose(sl.values.grad, expect, atol=1e-5)
    assert (sl.values.grad.abs().sum(1) > 0).all(), "some rows got no gradient"


def test_sparse_linear_bf16_input_gradients_finite():
    torch.manual_seed(3)
    sl = SparseLinear(64, 96, torch.stack([torch.randperm(64)[:9] for _ in range(96)]))
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16, requires_grad=True)
    y = sl(x)
    assert y.dtype is torch.bfloat16
    y.float().pow(2).sum().backward()
    assert x.grad.dtype is torch.bfloat16
    assert sl.values.grad.dtype is torch.float32
    assert torch.isfinite(x.grad).all() and torch.isfinite(sl.values.grad).all()


def test_sparse_grad_v_native_matches_torch_fallback():
    """The fused d/dvalues kernel must agree with the gather it replaces.

    Forces both backend paths: ``_sparse`` is set to False to take the
    chunked torch fallback and reset to None to rebuild/reuse the extension.
    """
    from kernel.compute import get_backend
    be = get_backend()
    native = be.has_sparse_native
    try:
        for rows, out_f, in_f, K in [(64, 128, 96, 12), (16, 33, 17, 5), (2, 4, 8, 3)]:
            torch.manual_seed(rows)
            cols = torch.stack([torch.randperm(in_f)[:K] for _ in range(out_f)])
            vals = torch.randn(out_f, K) * 0.1
            xf = torch.randn(rows, in_f)
            dof = torch.randn(rows, out_f)
            be._sparse = False
            fb = be.sparse_grad_v(dof, xf, cols, vals, 1 << 22)
            be._sparse = None
            if native:
                nat = be.sparse_grad_v(dof, xf, cols, vals, 1 << 22)
                scale = max(float(fb.abs().max()), 1.0)
                assert float((nat - fb).abs().max()) / scale < 1e-5, (rows, out_f, in_f, K)
            # And both must match an explicit reference.
            ref = torch.empty_like(vals)
            for o0 in range(0, out_f, 16):
                o1 = min(o0 + 16, out_f)
                g = xf.index_select(1, cols[o0:o1].reshape(-1)).view(rows, o1 - o0, K)
                ref[o0:o1] = (dof[:, o0:o1].unsqueeze(-1) * g).sum(0)
            scale = max(float(ref.abs().max()), 1.0)
            assert float((fb - ref).abs().max()) / scale < 1e-5
    finally:
        be._sparse = None


def test_sparse_grad_v_fallback_handles_degenerate_shapes():
    from kernel.compute import get_backend
    be = get_backend()
    be._sparse = False          # force the torch path regardless of the extension
    try:
        for rows, out_f, in_f, K in [(1, 1, 1, 1), (4, 3, 7, 2)]:
            cols = torch.stack([torch.randperm(in_f)[:K] for _ in range(out_f)])
            vals = torch.randn(out_f, K)
            out = be.sparse_grad_v(torch.randn(rows, out_f), torch.randn(rows, in_f),
                                   cols, vals, 1 << 22)
            assert out.shape == (out_f, K) and torch.isfinite(out).all()
    finally:
        be._sparse = None

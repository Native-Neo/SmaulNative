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
from rawr_graph import build_graph, fallback_graph, hidden_cols, load_graph, save_graph
from smaul_linear import LinearConfig, SmaulLinear
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

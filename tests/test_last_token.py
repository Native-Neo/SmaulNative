"""Regression tests for last-token-only LM-head inference.

Proves the generation path never materializes full [B, T, V] logits while
returning the same next-token distribution as the full computation.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from inference import LinearInference
from smaul_linear import LinearConfig, SmaulLinear
from tokenizer import SmaulTokenizer

_VOCAB = 64
_D = 32


def _plain_cfg():
    return LinearConfig(vocab_size=_VOCAB, d_model=_D, n_layer=2, n_heads=2,
                        precision="fp32", architecture="plain")


def _rawr_cfg():
    return LinearConfig(vocab_size=_VOCAB, d_model=_D, n_layer=2, n_heads=2,
                        precision="fp32", architecture="rawr",
                        rawr_sparsity=0.5, rawr_min_degree=2)


def _mini_tokenizer():
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3,
             "hello": 4, "world": 5, " ": 6, "!": 7}
    # Pad out to _VOCAB single-char tokens so checkpoint/tokenizer vocabs agree.
    filler = [c for c in "abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-_/\\|@#$%*+=<>()[]{}"
              if c not in vocab]
    i = len(vocab)
    for c in filler:
        if i >= _VOCAB:
            break
        vocab[c] = i
        i += 1
    assert len(vocab) == _VOCAB, len(vocab)
    data = {
        "vocab": vocab,
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": ["<cap>", "<upper>"],
        "unk_id": 1,
        "stats": {"vocab_size": _VOCAB},
    }
    return SmaulTokenizer(data)


def _model_dir(tmp_path, arch):
    torch.manual_seed(0)
    cfg = _plain_cfg() if arch == "plain" else _rawr_cfg()
    m = SmaulLinear(cfg)
    d = tmp_path / f"ckpt_{arch}"
    m.save_pretrained(d)
    _mini_tokenizer().save(d / "tokenizer.json")
    return d


class _HeadSpy(torch.nn.Module):
    """Wraps a head, recording the sequence length it was asked to project."""

    def __init__(self, head, budget):
        super().__init__()
        self.head = head
        self.budget = budget
        self.seen_rows = []

    def forward(self, x):
        rows = x.shape[-2]
        self.seen_rows.append(rows)
        assert rows <= self.budget, (
            f"head asked to project {rows} rows (budget {self.budget}); "
            f"full-sequence logits would allocate here")
        return self.head(x)


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_last_only_shape(arch):
    torch.manual_seed(0)
    cfg = _plain_cfg() if arch == "plain" else _rawr_cfg()
    m = SmaulLinear(cfg).eval()
    for batch, t in ((1, 7), (2, 7), (1, 1)):
        ids = torch.randint(0, _VOCAB, (batch, t))
        with torch.inference_mode():
            logits, loss = m(ids, last_only=True)
        assert logits.shape == (batch, 1, _VOCAB)
        assert loss is None


@pytest.mark.parametrize("arch", ["plain", "rawr"])
@pytest.mark.parametrize("batch", [1, 2])
def test_last_only_matches_full_row(arch, batch):
    torch.manual_seed(0)
    cfg = _plain_cfg() if arch == "plain" else _rawr_cfg()
    m = SmaulLinear(cfg).eval()
    ids = torch.randint(0, _VOCAB, (batch, 9))
    with torch.inference_mode():
        full, _ = m(ids)
        last, _ = m(ids, last_only=True)
    assert last.shape == (batch, 1, _VOCAB)
    torch.testing.assert_close(last[:, 0, :], full[:, -1, :], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_no_full_logits_allocated(arch):
    torch.manual_seed(0)
    cfg = _plain_cfg() if arch == "plain" else _rawr_cfg()
    m = SmaulLinear(cfg).eval()
    m.head = _HeadSpy(m.head, budget=8)
    ids = torch.randint(0, _VOCAB, (2, 64))
    with torch.inference_mode():
        logits, _ = m(ids, last_only=True)
    assert logits.shape == (2, 1, _VOCAB)
    assert m.head.seen_rows == [1]


def test_last_only_rejects_labels():
    torch.manual_seed(0)
    m = SmaulLinear(_plain_cfg()).eval()
    ids = torch.randint(0, _VOCAB, (1, 4))
    with pytest.raises(ValueError, match="last_only"):
        m(ids, ids, last_only=True)


def test_training_path_unchanged():
    torch.manual_seed(0)
    m = SmaulLinear(_plain_cfg())
    m.train()
    ids = torch.randint(0, _VOCAB, (2, 8))
    logits, loss = m(ids, ids)
    assert logits.shape == (2, 8, _VOCAB)
    assert torch.isfinite(loss)


def test_256k_safe_with_stubbed_trunk():
    """256K-shaped input through last_only without a ~64GB allocation.

    The trunk is stubbed to identity so this tests the head-path plumbing
    (shapes, batching, budget), not 256K steps of attention math.
    """
    torch.manual_seed(0)
    m = SmaulLinear(_plain_cfg()).eval()
    ident = torch.nn.Identity()
    m.n0 = ident
    m.nf = ident
    for b in m.blocks:
        b.forward = lambda x: x
    m.head = _HeadSpy(m.head, budget=8)
    T = 262144
    for batch in (1, 2):
        ids = torch.randint(0, _VOCAB, (batch, T))
        with torch.inference_mode():
            logits, _ = m(ids, last_only=True)
        assert logits.shape == (batch, 1, _VOCAB)
    # Single-token prompt matches the full path exactly here.
    m.head = m.head.head
    ids = torch.tensor([[5]])
    with torch.inference_mode():
        assert torch.equal(m(ids, last_only=True)[0], m(ids)[0])


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_generation_matches_full_logits_path(tmp_path, monkeypatch, arch):
    """Greedy generation identical with last-only vs full-logits forward."""
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    engine.eos_id = -1  # never trigger EOS: compare fixed-length outputs

    def _forward_full(self, tokens):
        import torch as _torch

        ids = _torch.tensor([tokens], dtype=_torch.long, device=self.device)
        with _torch.inference_mode():
            logits, _ = self.model(ids)
        return logits, None

    out_last = engine.generate("hello world", max_new_tokens=6, temperature=0.0,
                               top_k=0, top_p=1.0)
    monkeypatch.setattr(engine, "_forward",
                        types.MethodType(_forward_full, engine))
    out_full = engine.generate("hello world", max_new_tokens=6, temperature=0.0,
                               top_k=0, top_p=1.0)
    assert out_last == out_full


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_generation_works(tmp_path, arch):
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    text = engine.generate("hello", max_new_tokens=4, temperature=0.0, top_k=0)
    assert isinstance(text, str)
    chunks = list(engine.stream("hello", max_new_tokens=4, temperature=0.0, top_k=0))
    assert "".join(chunks) == text

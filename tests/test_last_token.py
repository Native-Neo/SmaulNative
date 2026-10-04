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
from model import LinearConfig, SmaulLinear
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


class _NoPrefillModel:
    """Proxy exposing only the last-only forward, i.e. pre-incremental-decoding.

    Used to drive ``LinearInference``'s re-forward fallback, so the two decode
    paths can be compared against each other rather than each against itself.
    """

    def __init__(self, model):
        self._m = model

    def __call__(self, *a, **kw):
        return self._m(*a, **kw)


def _rel(a, b):
    d = float((a - b).abs().max())
    s = max(float(b.abs().max()), 1e-30)
    return d / s


def _logits_via_reforward(engine, seq):
    """Reference: the last position's logits from one batched forward."""
    with torch.inference_mode():
        return engine.model(torch.tensor([seq], dtype=torch.long))[0][:, -1:, :]


def _logits_via_decoder(engine, prompt, tail):
    """Same positions, produced by prefill + one step per appended token."""
    out = []
    with torch.inference_mode():
        lg, states = engine.model.prefill(torch.tensor([prompt], dtype=torch.long))
        out.append(lg)
        for t in tail:
            lg, states = engine.model.step(torch.tensor([[t]], dtype=torch.long), states)
            out.append(lg)
    return out


@pytest.mark.parametrize("arch", ["plain", "rawr"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_decode_matches_the_per_token_re_forward_path(tmp_path, arch, seed):
    """The prefill/step path must reproduce the old loop's logits.

    The old loop re-ran the whole window for every generated token; the new one
    carries the O(D^2) recurrence state. They are only interchangeable if the
    state is stepped exactly while it covers the same window a re-forward would
    have used, and re-prefilled the moment the window slides -- which is what
    the window test below pins. Here we pin the arithmetic.

    The bound is *relative* and loose on purpose. A per-token step is not
    bit-exact with a batched forward, and cannot be: torch selects a different
    GEMM/ SpMM kernel for a [1, d] input than for a [T, d] one, so the same
    row of an activation sums in a different order. Measured on this repo, the
    same row of x @ W.T differs by 4.7e-07 relative at T=12 and 4.1e-07 at
    T=128 (d=512, V=8000), and the Rawr SparseLinear head differs from its dense
    equivalent by 6.0e-07. 1e-4 is two to three orders of magnitude above that
    noise floor and still far below any real defect (the window-sliding bug
    this guards against produced 1.6e-01).
    """
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    torch.manual_seed(seed)
    prompt = engine.encode("hello world")
    tail = [int(t) for t in torch.randint(0, _VOCAB, (4,))]
    got = _logits_via_decoder(engine, prompt, tail)
    with torch.inference_mode():
        batched = engine.model(torch.tensor([prompt + tail], dtype=torch.long))[0]
    for i, g in enumerate(got):
        pos = len(prompt) - 1 + i
        ref = batched[:, pos:pos + 1, :]
        assert _rel(g, ref) < 1e-4, (arch, seed, i, _rel(g, ref))


def _count_decoder_calls(engine, prompt, max_new_tokens, window):
    import inference as _inf
    old = _inf.MODEL_WINDOW
    _inf.MODEL_WINDOW = window
    calls = {"prefill": 0, "step": 0}
    real_prefill, real_step = engine.model.prefill, engine.model.step
    engine.model.prefill = lambda idx: (calls.__setitem__("prefill", calls["prefill"] + 1)
                                        or real_prefill(idx))
    engine.model.step = lambda idx, st: (calls.__setitem__("step", calls["step"] + 1)
                                         or real_step(idx, st))
    try:
        engine.eos_id = -1
        engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0, top_k=0)
    finally:
        _inf.MODEL_WINDOW = old
        engine.model.prefill, engine.model.step = real_prefill, real_step
    return calls


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_incremental_decode_uses_steps_when_the_window_fits(tmp_path, arch):
    """With room in the window the decoder must step, not re-run the prefix.

    The whole speedup depends on this: re-prefilling per token would be the old
    O(N * window) behaviour with extra bookkeeping.
    """
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    calls = _count_decoder_calls(engine, "hello world", 6, 1 << 20)
    assert calls["prefill"] == 1, calls
    assert calls["step"] == 6, calls


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_incremental_decode_re_prefills_instead_of_stepping_a_sliding_window(tmp_path, arch):
    """A window shorter than the context slides on every append.

    Stepping then would be wrong, not merely imprecise: the carried state
    would still hold tokens the windowed re-forward excludes. The recurrence is
    a sum over everything it has absorbed, so the decoder must re-prefill -- and
    must never step -- once the window is full.

    MODEL_WINDOW is patched to 4 with a 5-token prompt so the full branch runs
    without allocating a 262144-token context.
    """
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    window, n = 4, 6
    plen = len(engine.encode("hello world"))
    calls = _count_decoder_calls(engine, "hello world", n, window)
    # The window has room for `window - plen` appends before it is full; a step
    # is valid exactly while the state still covers the whole required window,
    # so that is precisely how many steps may happen. Every later token must
    # re-prefill, because the window has slid and the state still holds evicted
    # tokens.
    assert calls["step"] == window - plen, calls
    assert calls["prefill"] == 1 + (n - calls["step"]), calls


@pytest.mark.parametrize("arch", ["plain", "rawr"])
def test_generate_without_prefill_step_falls_back(tmp_path, arch):
    """A model with no prefill/step must still generate, unchanged."""
    d = _model_dir(tmp_path, arch)
    engine = LinearInference(str(d), device="cpu")
    engine.eos_id = -1
    engine.model = _NoPrefillModel(engine.model)
    out = engine.generate("hello", max_new_tokens=5, temperature=0.0, top_k=0)
    assert isinstance(out, str) and out

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer import (CASE, SPECIAL, SmaulTokenizer, _build, _guaranteed,
                        devanagari_units, read_texts)


def test_recursive_text_order_is_deterministic(tmp_path):
    (tmp_path / "z.txt").write_text("z")
    sub = tmp_path / "a"
    sub.mkdir()
    (sub / "y.txt").write_text("y")
    (sub / "b.txt").write_text("b")
    assert list(read_texts(tmp_path)) == ["b", "y", "z"]


def test_undecodable_bytes_are_not_silently_lost(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_bytes(b"ok\xff\n")
    try:
        texts = list(read_texts(tmp_path))
    except UnicodeDecodeError:
        return
    # errors=replace preserves content as U+FFFD instead of crashing the stream.
    assert texts and any("ok" in t for t in texts)
    assert any("\ufffd" in t for t in texts)


def test_empty_tokenizer_corpus_is_rejected():
    with pytest.raises(RuntimeError, match="no usable text records"):
        _build(iter(()), vocab_size=16, word_budget=8)


def test_csv_prompt_completion_records_are_read(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Prompt,Completion\nHello,World\n")
    assert list(read_texts(path)) == ["Hello\nWorld"]


def test_tiny_tokenizer_preserves_whitespace():
    # vocab_size has to be able to hold the guaranteed character set (8 special +
    # 2 case + 227 guaranteed = 237) for the letters to be in the vocabulary at
    # all. This used to ask for 16 and passed only because the guaranteed set
    # was emitted unbounded, so the real vocabulary was 237 anyway; the fix that
    # honours --vocab truncates the set instead, and at 16 there is no room left
    # for "a" or "b" and the round trip below would be <unk> <unk>. The point of
    # the test is that whitespace survives and text round-trips, and that holds
    # from 237 up. The truncation itself is covered by
    # test_vocab_size_below_the_guaranteed_set_is_truncated.
    data = _build(["a b"], vocab_size=240, word_budget=8)
    tok = SmaulTokenizer(data)
    assert " " in tok.vocab
    assert tok.decode(tok.encode("a b")) == "a b"


def test_devanagari_units_preserve_marks_and_joiners():
    assert devanagari_units("कि") == ["कि"]
    assert devanagari_units("क्ष") == ["क्ष"]
    assert devanagari_units("क्\u200dष") == ["क्\u200dष"]
    assert devanagari_units("क\u093c") == ["क\u093c"]


# ---------------------------------------------------------------------------
# The guaranteed character set is 227 entries and used to be emitted
# unconditionally, so any --vocab below 237 (8 special + 2 case + 227) was
# silently overshot and the failure appeared later as an unrelated-looking
# "Rawr graph vocab 237 != config vocab 48" from SmaulLinear.
# ---------------------------------------------------------------------------

_MIN_HONEST_VOCAB = len(SPECIAL) + len(CASE) + len(_guaranteed())


@pytest.mark.parametrize("vocab_size", [24, 48, 96, 200, _MIN_HONEST_VOCAB - 1])
def test_vocab_size_is_always_honoured(vocab_size):
    """The vocabulary comes out at exactly vocab_size, no overshoot."""
    data = _build(["a b"], vocab_size=vocab_size, word_budget=8)
    assert len(data["vocab"]) == vocab_size, (
        f"asked for {vocab_size}, got {len(data['vocab'])}")
    assert data["stats"]["vocab_size"] == vocab_size


@pytest.mark.parametrize("vocab_size", [24, 48, 96])
def test_vocab_size_below_the_guaranteed_set_is_truncated(capsys, vocab_size):
    """Below 237 the guaranteed set is cut, and the cut is announced.

    Deterministic rather than arbitrary: the set is sorted and deduped, so a
    prefix is stable across runs and machines.
    """
    _build(["a b"], vocab_size=vocab_size, word_budget=8)
    out = capsys.readouterr().out
    assert "guaranteed character set" in out, out
    assert f"vocab_size={vocab_size}" in out, out
    # And it says what would have been needed, so the fix is actionable.
    assert str(_MIN_HONEST_VOCAB) in out, out
    data = _build(["a b"], vocab_size=vocab_size, word_budget=8)
    again = _build(["a b"], vocab_size=vocab_size, word_budget=8)
    assert data["vocab"] == again["vocab"], "truncation is not deterministic"


def test_guaranteed_set_is_not_truncated_when_it_fits(capsys):
    """No notice, and no behaviour change, at or above the honest minimum."""
    for v in (_MIN_HONEST_VOCAB, _MIN_HONEST_VOCAB + 1, 512):
        capsys.readouterr()
        _build(["a b"], vocab_size=v, word_budget=8)
        assert "guaranteed character set" not in capsys.readouterr().out, v


def test_guaranteed_prefix_is_kept_in_order():
    """What survives is a prefix of the guaranteed set, not an arbitrary subset."""
    vocab_size = 100
    data = _build(["a b"], vocab_size=vocab_size, word_budget=8)
    budget = vocab_size - len(SPECIAL) - len(CASE)
    want = [g for g in _guaranteed()[:budget]]
    got = [t for t in data["vocab"] if t in set(_guaranteed())]
    assert got == want, (got[:5], want[:5])

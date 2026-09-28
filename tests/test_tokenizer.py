import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer import SmaulTokenizer, _build, devanagari_units, read_texts


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

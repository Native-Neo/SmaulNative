import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer import SmaulTokenizer, _build, read_texts


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
        list(read_texts(tmp_path))
    except UnicodeDecodeError:
        return
    raise AssertionError("invalid UTF-8 must not be silently discarded")


def test_empty_tokenizer_corpus_is_rejected():
    with pytest.raises(RuntimeError, match="no usable text records"):
        _build(iter(()), vocab_size=16, word_budget=8)


def test_csv_prompt_completion_records_are_read(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Prompt,Completion\nHello,World\n")
    assert list(read_texts(path)) == ["Hello\nWorld"]


def test_tiny_tokenizer_preserves_whitespace():
    data = _build(["a b"], vocab_size=9, word_budget=8)
    tok = SmaulTokenizer(data)
    assert " " in tok.vocab
    assert tok.decode(tok.encode("a b")) == "a b"

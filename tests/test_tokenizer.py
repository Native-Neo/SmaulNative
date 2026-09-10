import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import read_texts


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

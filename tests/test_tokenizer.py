import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from tokenizer import (BYTE_VOCAB_SIZE, IncrementalByteDecoder, SmaulTokenizer,
                       decode_bytes, encode_bytes, read_texts)


def test_vocab_is_exactly_256_byte_values():
    assert BYTE_VOCAB_SIZE == 256
    assert SmaulTokenizer().get_vocab_size() == 256
    # Every value 0-255 is a valid single id, and encoding only emits them.
    for i in range(256):
        assert isinstance(decode_bytes([i]), str)
    assert set(encode_bytes("hello UTF-8: héllo → 世界 🦀")) <= set(range(256))


def test_every_byte_value_round_trips():
    ids = list(range(256))
    assert encode_bytes(decode_bytes(ids, errors="replace")) is not None
    assert decode_bytes(ids, errors="replace")
    # Pure ASCII is identity.
    assert encode_bytes("hello world 123") == [104, 101, 108, 108, 111, 32, 119,
                                                111, 114, 108, 100, 32, 49, 50, 51]


def test_arbitrary_byte_sequences_never_crash():
    assert isinstance(decode_bytes([0, 255, 128, 200, 1]), str)
    assert isinstance(decode_bytes([0xFF, 0xFE, 0x00, 0x41]), str)


def test_utf8_round_trips():
    for text in ["hello world",
                 "नमस्ते दुनिया, यह हिन्दी है।",
                 "κόσμος café naïve",
                 "def f(x):\n    return x + 1  # code → ✓",
                 "emoji 🦀🔥 mixed with\twhitespace\n\nand \x00 null",
                 "क्ष ज्ञ त्र श्र (conjuncts) क्\u200dष"]:
        assert decode_bytes(encode_bytes(text)) == text, text


def test_hindi_devanagari_preserved_byte_exact():
    text = "हिन्दी भाषा शब्द संख्या"
    ids = encode_bytes(text)
    assert ids == list(text.encode("utf-8"))
    assert decode_bytes(ids) == text


def test_source_code_preserved():
    code = "def quick_sort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    assert decode_bytes(encode_bytes(code)) == code


def test_incomplete_utf8_is_replaced_not_dropped():
    full = "€".encode("utf-8")  # 3 bytes
    assert decode_bytes(list(full)) == "€"
    # A truncated tail decodes with a replacement marker, not silently empty.
    got = decode_bytes(list(full[:2]))
    assert got == "\ufffd" or "�" in got


def test_decode_rejects_out_of_range_ids():
    for bad in ([256], [-1], [True], ["a"]):
        with pytest.raises(ValueError):
            decode_bytes(bad)


def test_encode_rejects_non_strings():
    for bad in (None, 42, b"bytes", ["x"]):
        with pytest.raises(ValueError):
            encode_bytes(bad)


def test_incremental_decoder_buffers_split_multibyte_chars():
    text = "a€b verstärkt 🦀"
    raw = list(text.encode("utf-8"))
    d = IncrementalByteDecoder()
    out = "".join(d.push(i) for i in raw) + d.flush()
    assert out == text


def test_incremental_decoder_emits_nothing_for_an_incomplete_tail():
    tail = list("€".encode("utf-8"))
    d = IncrementalByteDecoder()
    assert d.push(tail[0]) == ""
    assert d.push(tail[1]) == ""
    assert d.push(tail[2]) == "€"


def test_incremental_decoder_flush_replaces_a_dangling_tail():
    d = IncrementalByteDecoder()
    d.push(list("€".encode("utf-8"))[0])
    assert d.flush() == "\ufffd"


def test_tokenizer_save_load_round_trip(tmp_path):
    p = tmp_path / "tok.json"
    SmaulTokenizer().save(p)
    tok = SmaulTokenizer.from_file(p)
    assert tok.get_vocab_size() == 256
    assert tok.decode(tok.encode("hi 👍")) == "hi 👍"


def test_tokenizer_refuses_a_legacy_word_level_file(tmp_path):
    p = tmp_path / "tok.json"
    p.write_text('{"version": 8, "vocab": {"hello": 4}, "unk_id": 1}')
    with pytest.raises(ValueError, match="legacy word-level"):
        SmaulTokenizer.from_file(p)


def test_train_tokenizer_writes_bytes_without_a_corpus(tmp_path):
    from tokenizer import train_tokenizer
    out = tmp_path / "tok.json"
    tok = train_tokenizer(None, out)
    assert out.exists() and tok.get_vocab_size() == 256


def test_train_tokenizer_rejects_a_non_byte_vocab(tmp_path):
    from tokenizer import train_tokenizer
    with pytest.raises(ValueError, match="exactly 256"):
        train_tokenizer(None, tmp_path / "tok.json", vocab_size=64)


def test_ensure_tokenizer_rewrites_a_legacy_file(tmp_path, capsys):
    from tokenizer import ensure_tokenizer
    out = tmp_path / "tok.json"
    out.write_text('{"version": 8, "vocab": {"a": 0}}')
    tok = ensure_tokenizer(out, None)
    assert tok.get_vocab_size() == 256
    assert "legacy" in capsys.readouterr().out


def test_ensure_tokenizer_reuses_a_byte_file(tmp_path, capsys):
    from tokenizer import ensure_tokenizer
    out = tmp_path / "tok.json"
    first = ensure_tokenizer(out, None)
    capsys.readouterr()
    second = ensure_tokenizer(out, None)
    assert "writing" not in capsys.readouterr().out
    assert second.get_vocab_size() == first.get_vocab_size() == 256


# --- corpus text extraction (kept helpers) ----------------------------------

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
    texts = list(read_texts(tmp_path))
    assert texts and any("ok" in t for t in texts)
    assert any("\ufffd" in t for t in texts)


def test_csv_prompt_completion_records_are_read(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("Prompt,Completion\nHello,World\n")
    assert list(read_texts(path)) == ["Hello\nWorld"]


def test_read_texts_splits_plain_files_into_paragraphs(tmp_path):
    (tmp_path / "a.txt").write_text("one\n\ntwo\n\n\n\nthree\n\n")
    assert list(read_texts(tmp_path)) == ["one", "two", "three"]


def test_read_texts_accepts_a_single_file_or_a_directory(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("only doc")
    assert list(read_texts(f)) == ["only doc"]
    assert list(read_texts(tmp_path)) == ["only doc"]


def test_read_texts_max_records_stops_early(tmp_path):
    (tmp_path / "a.txt").write_text("\n\n".join(f"p{i}" for i in range(50)))
    assert list(read_texts(tmp_path, 3)) == ["p0", "p1", "p2"]
    assert len(list(read_texts(tmp_path, 10_000))) == 50


def test_read_texts_skips_an_empty_or_whitespace_file(tmp_path):
    (tmp_path / "a.txt").write_text("   \n\n\t\n")
    (tmp_path / "b.txt").write_text("real")
    assert list(read_texts(tmp_path)) == ["real"]


def test_read_texts_skips_oversized_plain_text(tmp_path, capsys, monkeypatch):
    import tokenizer as tk
    monkeypatch.setattr(tk, "MAX_PLAIN_BYTES", 5)
    (tmp_path / "a.txt").write_text("x" * 50)
    (tmp_path / "b.txt").write_text("ok")
    assert list(read_texts(tmp_path)) == ["ok"]
    assert "oversized text file" in capsys.readouterr().out


def test_read_texts_skips_a_symlink_escaping_the_corpus_root(tmp_path):
    outside = tmp_path.parent / "outside-tok-corpus"
    outside.mkdir(exist_ok=True)
    (outside / "leak.txt").write_text("should not be read")
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "real.txt").write_text("wanted")
    try:
        (root / "link.txt").symlink_to(outside / "leak.txt")
    except OSError:
        pytest.skip("symlinks unavailable")
    assert list(read_texts(root)) == ["wanted"]


def test_read_texts_reads_csv(tmp_path):
    import csv as _csv
    with (tmp_path / "a.csv").open("w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["prompt", "completion"])
        w.writeheader()
        w.writerow({"prompt": "Q1", "completion": "A1"})
        w.writerow({"prompt": "Q2", "completion": "A2"})
    assert list(read_texts(tmp_path)) == ["Q1\nA1", "Q2\nA2"]


def test_read_texts_skips_a_headerless_csv(tmp_path):
    (tmp_path / "a.csv").write_text("")
    assert list(read_texts(tmp_path)) == []


def test_read_texts_reads_jsonl_and_json(tmp_path):
    import json as _json
    (tmp_path / "a.jsonl").write_text('{"text": "one"}\n{not json}\n{"text": "two"}\n')
    (tmp_path / "b.json").write_text(_json.dumps([{"text": "three"}, {"text": "four"}]))
    assert list(read_texts(tmp_path)) == ["one", "two", "three", "four"]


def test_read_texts_parquet_prompt_completion(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    pq.write_table(pa.table({"prompt": ["Q1", "Q2"], "completion": ["A1", None]}),
                   tmp_path / "a.parquet")
    assert list(read_texts(tmp_path)) == ["Q1\nA1", "Q2"]


def test_coerce_list_text_handles_every_shape():
    from tokenizer import _coerce_list_text
    assert _coerce_list_text("plain") == "plain"
    assert _coerce_list_text(["a", "  ", "b"]) == "a\nb"
    assert _coerce_list_text(42) == ""
    assert _coerce_list_text(None) == ""
    for key in ("value", "text", "content", "completion", "output", "answer"):
        assert _coerce_list_text([{key: f"from {key}"}]) == f"from {key}"


def test_string_values_pairs_prompt_and_completion():
    from tokenizer import _string_values
    assert list(_string_values({"prompt": "Q", "completion": "A"})) == ["Q\nA"]
    assert list(_string_values({"completion": "A"})) == ["A"]
    assert list(_string_values({"prompt": "", "completion": "A"})) == ["A"]


def test_record_text_matches_the_prompt_completion_pairing():
    from tokenizer import _record_text
    assert _record_text({"prompt": "Q", "completion": "A"}) == "Q\nA"
    assert _record_text({"completion": "A"}) == "A"
    assert _record_text({"text": "T"}) == "T"
    assert _record_text({"n": 1}) == ""


# --- the CLI ----------------------------------------------------------------

def test_cli_encode_from_text_and_from_file(tmp_path, monkeypatch, capsys):
    import tokenizer as tk
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode", "--text", "hello"])
    tk.main()
    from_text = capsys.readouterr().out.strip().splitlines()[-1]
    assert from_text == "104 101 108 108 111"

    f = tmp_path / "in.txt"
    f.write_text("hello")
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode",
                                     "--text-file", str(f)])
    tk.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == from_text


def test_cli_encode_requires_some_text(monkeypatch):
    import tokenizer as tk
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode"])
    with pytest.raises(ValueError, match="--text"):
        tk.main()


def test_cli_decode_round_trips_and_rejects_garbage(tmp_path, monkeypatch, capsys):
    import tokenizer as tk
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "decode", "--ids", "104 105"])
    tk.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == "hi"

    monkeypatch.setattr("sys.argv", ["tokenizer.py", "decode", "--ids", "1 two 3"])
    with pytest.raises(ValueError, match="space-separated"):
        tk.main()


def test_cli_requires_a_subcommand(monkeypatch):
    import tokenizer as tk
    monkeypatch.setattr("sys.argv", ["tokenizer.py"])
    with pytest.raises(SystemExit):
        tk.main()

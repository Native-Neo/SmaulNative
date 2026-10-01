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
        pytest.skip("read_texts decodes strictly; nothing to assert")
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


# ---------------------------------------------------------------------------
# read_texts and the CLI were the uncovered half of this file. read_texts is
# what decides what the tokenizer is trained on, so its failure mode is the
# quiet one: a format it cannot parse is skipped, and a smaller vocabulary gets
# built from whatever did parse.
# ---------------------------------------------------------------------------

import csv as _csv
import json as _json
import pyarrow as pa
import pyarrow.parquet as pq

from tokenizer import (_coerce_list_text, _record_text, _string_values,
                        ensure_tokenizer, train_tokenizer)


def test_read_texts_splits_plain_files_into_paragraphs(tmp_path):
    (tmp_path / "a.txt").write_text("one\n\ntwo\n\n\n\nthree\n\n")
    assert list(read_texts(tmp_path)) == ["one", "two", "three"]


def test_read_texts_accepts_a_single_file_or_a_directory(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("only doc")
    assert list(read_texts(f)) == ["only doc"]
    assert list(read_texts(tmp_path)) == ["only doc"]


def test_read_texts_max_records_stops_early(tmp_path):
    """The cap has to work mid-file, not just at a file boundary."""
    (tmp_path / "a.txt").write_text("\n\n".join(f"p{i}" for i in range(50)))
    assert list(read_texts(tmp_path, 3)) == ["p0", "p1", "p2"]
    assert len(list(read_texts(tmp_path, 10_000))) == 50


def test_read_texts_max_records_spans_multiple_files(tmp_path):
    (tmp_path / "a.txt").write_text("a1\n\na2\n\na3")
    (tmp_path / "b.txt").write_text("b1\n\nb2")
    assert list(read_texts(tmp_path, 4)) == ["a1", "a2", "a3", "b1"]


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
    (tmp_path / "a.jsonl").write_text('{"text": "one"}\n{not json}\n{"text": "two"}\n')
    (tmp_path / "b.json").write_text(_json.dumps([{"text": "three"}, {"text": "four"}]))
    assert list(read_texts(tmp_path)) == ["one", "two", "three", "four"]


def test_read_texts_skips_oversized_json(tmp_path, capsys):
    """The JSON cap is a hardcoded 50 MB, not MAX_PLAIN_BYTES, so it needs a
    genuinely large file. truncate makes that a sparse file: the size check
    sees 50 MB and the test costs nothing."""
    f = tmp_path / "a.json"
    f.write_text("{}")
    with f.open("r+") as fh:
        fh.truncate(50_000_001)
    assert f.stat().st_size > 50_000_000
    assert list(read_texts(tmp_path)) == []
    assert "oversized JSON" in capsys.readouterr().out


def test_read_texts_parquet_prompt_completion(tmp_path):
    pq.write_table(pa.table({"prompt": ["Q1", "Q2"], "completion": ["A1", None]}),
                   tmp_path / "a.parquet")
    assert list(read_texts(tmp_path)) == ["Q1\nA1", "Q2"]


def test_read_texts_parquet_prefers_a_text_key(tmp_path):
    pq.write_table(pa.table({"id": ["x", "y"], "text": ["first", "second"]}),
                   tmp_path / "a.parquet")
    assert list(read_texts(tmp_path)) == ["first", "second"]


def test_read_texts_parquet_falls_back_to_string_columns(tmp_path):
    """No recognized key: it must not silently yield nothing."""
    pq.write_table(pa.table({"weird": ["alpha beta", "gamma delta"]}),
                   tmp_path / "a.parquet")
    assert list(read_texts(tmp_path)) == ["alpha beta", "gamma delta"]


def test_read_texts_parquet_list_valued_prompt(tmp_path):
    """A list-valued prompt column is a real shape (multi-turn)."""
    # pyarrow requires a list column to be lists throughout.
    pq.write_table(pa.table({"prompt": [["u1", "u2"], ["single"]],
                             "completion": [["a1", "a2"], ["one"]]}), tmp_path / "a.parquet")
    assert list(read_texts(tmp_path)) == ["u1\nu2\na1\na2", "single\none"]


# --- the record coercers ----------------------------------------------------

def test_coerce_list_text_handles_every_shape():
    assert _coerce_list_text("plain") == "plain"
    assert _coerce_list_text(["a", "  ", "b"]) == "a\nb"
    assert _coerce_list_text(42) == ""
    assert _coerce_list_text(None) == ""
    for key in ("value", "text", "content", "completion", "output", "answer"):
        assert _coerce_list_text([{key: f"from {key}"}]) == f"from {key}"


def test_coerce_list_text_skips_unusable_dict_entries():
    assert _coerce_list_text([{"nope": "x"}, {"text": "kept"}]) == "kept"
    assert _coerce_list_text([{"text": "   "}, {"text": ""}]) == ""


def test_string_values_pairs_prompt_and_completion():
    assert list(_string_values({"prompt": "Q", "completion": "A"})) == ["Q\nA"]
    # A completion with no prompt is still usable.
    assert list(_string_values({"completion": "A"})) == ["A"]
    # And an empty prompt does not produce a leading newline.
    assert list(_string_values({"prompt": "", "completion": "A"})) == ["A"]


def test_string_values_falls_back_to_any_string_in_the_object():
    out = list(_string_values({"meta": {"nested": "deep value"}, "n": 1}))
    assert "deep value" in out


def test_string_values_yields_nothing_for_empty_input():
    assert list(_string_values({})) == []
    assert list(_string_values(None)) == []
    assert list(_string_values([])) == []


def test_record_text_matches_the_prompt_completion_pairing():
    assert _record_text({"prompt": "Q", "completion": "A"}) == "Q\nA"
    assert _record_text({"completion": "A"}) == "A"
    assert _record_text({"text": "T"}) == "T"
    assert _record_text({"n": 1}) == ""


# --- building a tokenizer ---------------------------------------------------

def test_train_tokenizer_from_a_directory(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world this is a corpus " * 40)
    out = tmp_path / "tok.json"
    tok = train_tokenizer(corpus, out, vocab_size=64)
    assert out.exists() and tok.get_vocab_size() <= 64
    assert _json.loads(out.read_text())["vocab"]


def test_train_tokenizer_accepts_a_text_stream(tmp_path):
    out = tmp_path / "tok.json"
    tok = train_tokenizer(None, out, vocab_size=64, texts=iter(["hello world " * 40]))
    assert out.exists() and tok.get_vocab_size() <= 64


def test_ensure_tokenizer_reuses_a_matching_file(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    first = ensure_tokenizer(out, corpus, vocab_size=64)
    capsys.readouterr()
    second = ensure_tokenizer(out, corpus, vocab_size=64)
    assert "creating" not in capsys.readouterr().out
    assert second.data["vocab"] == first.data["vocab"]


def test_ensure_tokenizer_rebuilds_on_a_vocab_mismatch(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    ensure_tokenizer(out, corpus, vocab_size=64)
    capsys.readouterr()
    ensure_tokenizer(out, corpus, vocab_size=48)
    assert "rebuild" in capsys.readouterr().out
    assert len(_json.loads(out.read_text())["vocab"]) <= 48


def test_ensure_tokenizer_rebuilds_on_a_version_mismatch(tmp_path, capsys):
    """A stale-format file must be rebuilt, not loaded and used."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    data = _json.loads(train_tokenizer(corpus, out, vocab_size=64).data and
                       out.read_text())
    data["version"] = 0
    out.write_text(_json.dumps(data))
    capsys.readouterr()
    ensure_tokenizer(out, corpus, vocab_size=64)
    assert "rebuild" in capsys.readouterr().out
    assert _json.loads(out.read_text())["version"] != 0


def test_ensure_tokenizer_rebuilds_an_unreadable_file(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    out.write_text("{not json")
    tok = ensure_tokenizer(out, corpus, vocab_size=64)
    assert "unreadable" in capsys.readouterr().out
    assert tok.get_vocab_size() > 0


def test_ensure_tokenizer_accepts_a_json_dataset_path(tmp_path):
    """Passing dataset.json should train on its directory, not the file."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    (corpus / "dataset.json").write_text(_json.dumps([{"text": "hi"}]))
    out = tmp_path / "tok.json"
    assert ensure_tokenizer(out, corpus / "dataset.json", vocab_size=64).get_vocab_size() > 0


def test_streaming_requires_a_record_cap():
    """An unbounded FineWeb download must be refused, not started."""
    with pytest.raises(ValueError, match="max-records"):
        train_tokenizer(None, "/tmp/should-not-exist.json", 64, stream_name="fineweb")


# --- the CLI ----------------------------------------------------------------

def _tok_file(tmp_path, vocab=64):
    corpus = tmp_path / "c"
    corpus.mkdir(exist_ok=True)
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    train_tokenizer(corpus, out, vocab_size=vocab)
    return out


def test_cli_train_writes_the_vocabulary_and_reports_it(tmp_path, monkeypatch, capsys):
    import tokenizer as tk
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "nested" / "tok.json"          # parent does not exist yet
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "train", "--fromdataset", str(corpus),
                                     "--vocab-size", "64", "--output", str(out)])
    tk.main()
    printed = capsys.readouterr().out
    assert "Vocabulary:" in printed and "Unique words:" in printed and "Saved:" in printed
    assert out.exists()


def test_cli_train_leaves_no_temp_file_behind(tmp_path, monkeypatch):
    import tokenizer as tk
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("hello world corpus " * 40)
    out = tmp_path / "tok.json"
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "train", "--fromdataset", str(corpus),
                                     "--vocab-size", "64", "--output", str(out)])
    tk.main()
    # The atomic write must clean up its temp file on success.
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_cli_train_rejects_bad_counts(tmp_path, monkeypatch):
    import tokenizer as tk
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "a.txt").write_text("x")
    for flag, val in (("--vocab-size", "0"), ("--vocab-size", "-5"), ("--word-budget", "-1"),
                      ("--max-records", "-1")):
        monkeypatch.setattr("sys.argv", ["tokenizer.py", "train", "--fromdataset", str(corpus),
                                         flag, val, "--output", str(tmp_path / "t.json")])
        with pytest.raises(ValueError):
            tk.main()


def test_cli_encode_from_text_and_from_file(tmp_path, monkeypatch, capsys):
    import tokenizer as tk
    tok = _tok_file(tmp_path)
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode", "--tokenizer", str(tok),
                                     "--text", "hello"])
    tk.main()
    # load() may print a vocabulary-coverage warning, so take the last line.
    from_text = capsys.readouterr().out.strip().splitlines()[-1]
    assert from_text and all(p.isdigit() for p in from_text.split())

    f = tmp_path / "in.txt"
    f.write_text("hello")
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode", "--tokenizer", str(tok),
                                     "--text-file", str(f)])
    tk.main()
    assert capsys.readouterr().out.strip().splitlines()[-1] == from_text


def test_cli_encode_requires_some_text(tmp_path, monkeypatch):
    import tokenizer as tk
    tok = _tok_file(tmp_path)
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode", "--tokenizer", str(tok)])
    with pytest.raises(ValueError, match="--text"):
        tk.main()


def test_cli_decode_round_trips_and_rejects_garbage(tmp_path, monkeypatch, capsys):
    import tokenizer as tk
    tok = _tok_file(tmp_path)
    monkeypatch.setattr("sys.argv", ["tokenizer.py", "encode", "--tokenizer", str(tok),
                                     "--text", "hello world"])
    tk.main()
    ids = capsys.readouterr().out.strip().splitlines()[-1]

    monkeypatch.setattr("sys.argv", ["tokenizer.py", "decode", "--tokenizer", str(tok),
                                     "--ids", ids])
    tk.main()
    assert capsys.readouterr().out.strip().splitlines()[-1]

    monkeypatch.setattr("sys.argv", ["tokenizer.py", "decode", "--tokenizer", str(tok),
                                     "--ids", "1 two 3"])
    with pytest.raises(ValueError, match="space-separated"):
        tk.main()


def test_cli_requires_a_subcommand(monkeypatch):
    import tokenizer as tk
    monkeypatch.setattr("sys.argv", ["tokenizer.py"])
    with pytest.raises(SystemExit):
        tk.main()

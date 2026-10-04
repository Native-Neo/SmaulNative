import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dataset import IGNORE_INDEX, _preprocess_conversation, iter_texts


class Tok:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text):
        return list(range(2, 2 + len(text.split())))


def test_plain_text_resume_is_one_based(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("one\n\ntwo")
    files = [path.resolve()]
    assert list(iter_texts(files, str(path.resolve()), 1))[0][0] == "two"
    assert list(iter_texts(files, str(path.resolve()), 2)) == []


def test_pretrain_resume_replays_buffered_tokens(tmp_path):
    from dataset import PretrainStream

    path = tmp_path / "x.txt"
    path.write_text("abcdefghij")

    class CharTok:
        eos_token_id = 99

        def encode(self, text):
            return list(range(len(text)))

    stream = PretrainStream(tmp_path, CharTok(), ctx_len=3)
    items = iter(stream)
    next(items)
    saved = next(items)
    resume_path, resume_record = saved[2]
    resumed = PretrainStream(tmp_path, CharTok(), 3, resume_path, resume_record, list(stream.buffer_tokens))
    assert next(iter(resumed))[0].tolist() == next(items)[0].tolist()


def test_resume_file_must_be_in_discovered_files(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("text")
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError, match="resume file not found"):
        list(iter_texts([path.resolve()], str(missing), 0))


def test_negative_resume_record_is_rejected(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("text")
    with pytest.raises(ValueError, match="resume_record"):
        list(iter_texts([path.resolve()], None, -1))


def test_prompt_and_completion_are_combined(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"prompt": "Q", "completion": "A"}) + "\n")
    assert list(iter_texts([path]))[0][0] == "Q\nA"


def test_jsonl_resume_is_one_based(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text("\n".join(json.dumps({"text": f"row{i}"}) for i in range(3)) + "\n")
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row2"]


def test_json_resume_is_one_based(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps([{"text": "row0"}, {"text": "row1"}, {"text": "row2"}]))
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row2"]


def test_csv_resume_is_one_based(tmp_path):
    path = tmp_path / "x.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()
        writer.writerows({"text": f"row{i}"} for i in range(3))
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row2"]


def test_parquet_resume_is_one_based(tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"text": ["row0", "row1", "row2"]}), path)
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row2"]


def test_parquet_prompt_and_completion_are_combined(tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"prompt": ["Q1", "Q2"], "completion": ["A1", "A2"]}), path)
    rows = list(iter_texts([path]))
    assert [row[0] for row in rows] == ["Q1\nA1", "Q2\nA2"]


def test_sft_only_assistant_is_target():
    result = _preprocess_conversation([
        {"from": "system", "value": "rules"},
        {"from": "user", "value": "question"},
        {"from": "assistant", "value": "answer"},
        {"from": "unknown", "value": "noise"},
    ], Tok(), 32, 0)
    labels = result["labels"].tolist()
    assert any(x != IGNORE_INDEX for x in labels)
    first_target = next(i for i, x in enumerate(labels) if x != IGNORE_INDEX)
    assert all(x == IGNORE_INDEX for x in labels[:first_target])


def test_malformed_conversation_entries_are_skipped():
    result = _preprocess_conversation([None, {}, {"from": "assistant"}, {"from": "assistant", "value": "ok"}], Tok(), 32, 0)
    assert result["input_ids"].numel() == 32


# ---------------------------------------------------------------------------
# The file above pinned resume arithmetic for six formats. Nothing tested what
# happens to a record whose shape the loader does not recognise -- which is the
# failure mode that does not announce itself. A dropped record is a smaller
# training set with a healthy-looking loss curve, not a crash.
# ---------------------------------------------------------------------------

def test_extract_text_prefers_prompt_and_completion_over_everything_else():
    from dataset import extract_text
    row = {"id": "a-very-long-identifier-value-here", "prompt": "Q", "completion": "A",
           "text": "this text column is longer than the prompt pair combined"}
    assert extract_text(row) == "Q\nA"


def test_extract_text_falls_back_through_text_keys_in_order():
    from dataset import TEXT_KEYS, extract_text
    # Every TEXT_KEYS name must actually be recognized, not just the first one.
    for key in TEXT_KEYS:
        if key in ("prompt", "completion"):
            continue          # those are only used as a pair
        assert extract_text({key: f"value from {key}"}) == f"value from {key}"


def test_extract_text_ignores_numeric_columns():
    """An id or timestamp column must never be mistaken for the document."""
    from dataset import extract_text
    assert extract_text({"ts": "12:30:45", "n": "42"}) == ""
    assert extract_text({"a": "1/2", "b": "--"}) == ""


def test_extract_text_prefers_a_sentence_over_an_identifier():
    from dataset import extract_text
    row = {"id": "9f8e7d6c5b4a", "note": "this is a real sentence worth training on"}
    assert extract_text(row) == "this is a real sentence worth training on"


def test_extract_text_handles_nesting_and_depth():
    from dataset import extract_text
    assert extract_text([{"text": "a"}, {"text": "b"}]) == "a\nb"
    # The depth cap exists so a self-referential structure cannot spin forever.
    deep = {"text": "x"}
    for _ in range(20):
        deep = [deep]
    assert extract_text(deep) == ""


def test_extract_text_returns_empty_for_useless_objects():
    from dataset import extract_text
    for obj in ({}, [], None, 42, 3.5, True):
        assert extract_text(obj) == ""


def test_extract_text_warns_once_per_file(capsys):
    from dataset import extract_text
    extract_text({"weird": "some text here for the corpus"}, "/tmp/a-ds.jsonl")
    first = capsys.readouterr().out.count("no recognized text column")
    for _ in range(5):
        extract_text({"weird": "some text here for the corpus"}, "/tmp/a-ds.jsonl")
    assert first == 1, "a 2M-record file must not print 2M warnings"
    assert capsys.readouterr().out.count("no recognized text column") == 0


def test_iter_texts_splits_txt_on_blank_lines(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("para one\n\npara two\n\n\npara three\n")
    rows = list(iter_texts([path.resolve()]))
    assert [t for t, _, _ in rows] == ["para one", "para two", "para three"]
    # Record numbers are 1-based and dense, since resume counts them.
    assert [i for _, _, i in rows] == [1, 2, 3]


def test_iter_texts_keeps_inner_blank_lines_out_of_a_paragraph(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("l1\n\nl2\nl3\n\n")
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["l1", "l2\nl3"]


def test_iter_texts_resume_agrees_across_formats(tmp_path):
    """resume_record=N must mean the same thing for .txt and .jsonl.

    These are separate code paths with different index conventions, and an
    off-by-one in either is invisible until a resume silently drops or repeats
    a document.
    """
    txt = tmp_path / "a.txt"
    txt.write_text("one\n\ntwo\n\nthree")
    assert [t for t, _, _ in iter_texts([txt.resolve()], str(txt.resolve()), 1)] == ["two", "three"]

    jl = tmp_path / "a.jsonl"
    jl.write_text("\n".join(json.dumps({"text": t}) for t in ("one", "two", "three")))
    assert [t for t, _, _ in iter_texts([jl.resolve()], str(jl.resolve()), 1)] == ["two", "three"]


def test_iter_texts_json_unwraps_only_a_list_data_field(tmp_path):
    """{"data": "a string"} is a real document, not a wrapper to unwrap."""
    from dataset import extract_text
    wrapped = tmp_path / "w.json"
    wrapped.write_text(json.dumps({"data": [{"text": "x"}, {"text": "y"}]}))
    assert [t for t, _, _ in iter_texts([wrapped.resolve()])] == ["x", "y"]

    literal = tmp_path / "l.json"
    literal.write_text(json.dumps({"data": "a real string field"}))
    assert [t for t, _, _ in iter_texts([literal.resolve()])] == ["a real string field"]


def test_iter_texts_json_accepts_a_bare_object(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"text": "single document"}))
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["single document"]


def test_iter_texts_jsonl_reports_and_skips_malformed_lines(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "good one"}\n{not json}\n{"text": "good two"}\n')
    rows = list(iter_texts([path.resolve()]))
    assert [t for t, _, _ in rows] == ["good one", "good two"]
    assert "skipped 1 malformed JSONL line" in capsys.readouterr().out


def test_iter_texts_keeps_record_numbers_positional_on_jsonl(tmp_path):
    """The yielded index is the line number, so a skipped line still advances it."""
    path = tmp_path / "a.jsonl"
    path.write_text('{"text": "a"}\n{not json}\n{"text": "b"}\n')
    assert [i for _, _, i in iter_texts([path.resolve()])] == [1, 3]


def test_iter_texts_skips_records_that_extract_to_nothing(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text('{"id": "1"}\n{"text": "real"}\n{"ts": "12:00"}\n')
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["real"]


def test_iter_texts_one_bad_file_does_not_abort_the_stream(tmp_path, capsys):
    """The warn-and-skip is the whole point of the outer try; test it."""
    bad = tmp_path / "a.json"
    bad.write_text("{not json at all")
    good = tmp_path / "b.jsonl"
    good.write_text('{"text": "survived"}')
    rows = list(iter_texts([bad.resolve(), good.resolve()]))
    assert [t for t, _, _ in rows] == ["survived"]
    assert "skipping dataset file" in capsys.readouterr().out


def test_iter_texts_csv_without_a_header_is_skipped(tmp_path, capsys):
    path = tmp_path / "a.csv"
    path.write_text("")                 # DictReader has no fieldnames for this
    assert list(iter_texts([path.resolve()])) == []
    assert "missing header" in capsys.readouterr().out


def test_iter_texts_csv_single_row_is_a_header_not_a_record(tmp_path):
    """A one-line CSV is entirely consumed as the header, so it yields nothing.

    Worth pinning because it is a way to get zero records with no warning at
    all -- the file is not skipped, there is simply nothing left to read.
    """
    path = tmp_path / "a.csv"
    path.write_text("col_a,col_b\n")
    assert list(iter_texts([path.resolve()])) == []


def test_iter_texts_cap_oversized_plain_text(tmp_path, capsys, monkeypatch):
    """A runaway .md/.py is skipped with a warning rather than read into RAM."""
    import dataset as ds
    monkeypatch.setattr(ds, "MAX_TEXT_FILE_BYTES", 10)
    path = tmp_path / "a.md"
    path.write_text("x" * 50)
    assert list(iter_texts([path.resolve()])) == []
    assert "oversized text file" in capsys.readouterr().out


def test_iter_texts_cap_oversized_json(tmp_path, capsys, monkeypatch):
    import dataset as ds
    monkeypatch.setattr(ds, "MAX_JSON_FILE_BYTES", 10)
    path = tmp_path / "a.json"
    path.write_text(json.dumps([{"text": "x" * 50}]))
    assert list(iter_texts([path.resolve()])) == []
    assert "oversized JSON file" in capsys.readouterr().out


def test_txt_size_cap_does_not_apply(tmp_path, monkeypatch, capsys):
    """A known asymmetry: the cap is checked for PLAIN_TEXT_SUFFIXES, but
    .txt/.text are matched earlier and stream paragraph by paragraph, so they
    bypass it. Recorded so the behaviour is deliberate rather than surprising
    if MAX_TEXT_FILE_BYTES is ever lowered expecting .txt to be covered.
    """
    import dataset as ds
    monkeypatch.setattr(ds, "MAX_TEXT_FILE_BYTES", 1)
    path = tmp_path / "a.txt"
    path.write_text("still read\n")
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["still read"]
    assert "oversized" not in capsys.readouterr().out


def test_parquet_null_completion_yields_the_prompt(tmp_path):
    """A null must not produce the string "None" or drop the row."""
    path = tmp_path / "a.parquet"
    table = pa.table({"prompt": ["Q1", "Q2"], "completion": ["A1", None]})
    pq.write_table(table, path)
    rows = [t for t, _, _ in iter_texts([path.resolve()])]
    assert rows == ["Q1\nA1", "Q2"]


def test_parquet_fast_path_picks_a_text_column(tmp_path):
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"identifier": ["x", "y"], "text": ["real one", "real two"]}),
                   path)
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["real one", "real two"]


def test_parquet_with_no_recognized_column_uses_the_generic_path(tmp_path):
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"a": ["some prose in an unknown column"],
                             "b": ["more prose here"]}), path)
    assert [t for t, _, _ in iter_texts([path.resolve()])] == ["some prose in an unknown column"]


# --- discovery and tokenizer plumbing ---------------------------------------

def test_discover_files_rejects_a_missing_directory(tmp_path):
    from dataset import discover_files
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_files(tmp_path / "nope")


def test_discover_files_is_sorted_and_recursive(tmp_path):
    from dataset import discover_files
    (tmp_path / "sub").mkdir()
    (tmp_path / "z.txt").write_text("z")
    (tmp_path / "sub" / "a.txt").write_text("a")
    (tmp_path / "sub" / "ignored.bin").write_bytes(b"\x00")
    found = [p.name for p in discover_files(tmp_path)]
    assert found == ["a.txt", "z.txt"]          # sorted, and .bin is not supported


def test_discover_files_skips_an_escaping_symlink(tmp_path, capsys):
    from dataset import discover_files
    outside = tmp_path.parent / "outside-secret"
    outside.mkdir(exist_ok=True)
    (outside / "leak.txt").write_text("should not be read")
    try:
        (tmp_path / "link.txt").symlink_to(outside / "leak.txt")
    except OSError:
        pytest.skip("symlinks unavailable")
    assert discover_files(tmp_path) == []
    assert "escaping dataset root" in capsys.readouterr().out


def test_load_tokenizer_reports_a_missing_file(tmp_path):
    from dataset import load_tokenizer
    with pytest.raises(FileNotFoundError, match="No byte tokenizer found"):
        load_tokenizer(tmp_path / "nope.json")


def test_load_tokenizer_round_trips_a_real_tokenizer(tmp_path):
    import dataset as ds
    from tokenizer import SmaulTokenizer
    p = tmp_path / "tok.json"
    SmaulTokenizer().save(p)
    tok = ds.load_tokenizer(p)
    assert tok.get_vocab_size() == 256
    assert tok.eos_token_id is None and tok.pad_token_id == 0
    assert ds.tokenizer_vocab_size(tok) == tok.get_vocab_size()
    assert ds.tokenizer_vocab_size(p) == tok.get_vocab_size()   # accepts a path


# --- PretrainStream ---------------------------------------------------------

def test_pretrain_stream_rejects_an_empty_or_bad_setup(tmp_path):
    from dataset import PretrainStream
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No supported files"):
        PretrainStream(empty, Tok(), 8)
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("hi")
    with pytest.raises(ValueError, match="ctx_len"):
        PretrainStream(tmp_path / "d", Tok(), 0)


def test_pretrain_stream_yields_shifted_pairs_and_records_position(tmp_path):
    from dataset import PretrainStream

    class CharTok:
        eos_token_id = 99
        def encode(self, text):
            return [ord(c) % 50 for c in text]

    (tmp_path / "a.txt").write_text("abcdefghij")
    stream = PretrainStream(tmp_path, CharTok(), 4)
    items = list(iter(stream))
    assert items
    for x, y, _ in items:
        assert x.numel() == 4 and y.numel() == 4
        assert torch.equal(y[:-1], x[1:])         # it really is a shift
    assert stream.last_pos[0].endswith("a.txt")


def test_pretrain_stream_truncates_an_oversized_document(tmp_path, capsys):
    from dataset import PretrainStream

    class CharTok:
        eos_token_id = 99
        def encode(self, text):
            return [1] * 200_000

    (tmp_path / "a.txt").write_text("x" * 200_000)
    stream = PretrainStream(tmp_path, CharTok(), 8)
    next(iter(stream))
    assert "truncating oversized document" in capsys.readouterr().out


# --- SFT --------------------------------------------------------------------

def test_only_assistant_turns_produce_targets():
    """The real invariant behind "context-only": system, tool and unknown roles
    must all be masked, not just anything before the first assistant turn. The
    existing test only checked the run of IGNORE before the first target, which
    an unmasked system turn at the end would also satisfy.
    """
    result = _preprocess_conversation([
        {"from": "system", "value": "rules"},
        {"from": "user", "value": "question"},
        {"from": "assistant", "value": "answer"},
    ], Tok(), 32, 0)
    labels = result["labels"].tolist()
    targets = [i for i, x in enumerate(labels) if x != IGNORE_INDEX]
    # Two-word turns, prefix "X: " is one token, so each turn is 2 ids and only
    # the assistant's value token is a target.
    assert len(targets) == 1, labels
    assert targets[0] == 4, labels


def test_unknown_roles_are_masked_like_system_turns():
    for role in ("system", "tool", "unknown", "function"):
        result = _preprocess_conversation([
            {"from": role, "value": "context only"},
            {"from": "user", "value": "q"},
            {"from": "assistant", "value": "a"},
        ], Tok(), 32, 0)
        labels = result["labels"].tolist()
        assert len([x for x in labels if x != IGNORE_INDEX]) == 1, role


def test_human_and_gpt_aliases_normalize():
    for user, asst in (("human", "gpt"), ("User", "Assistant"), ("USER", "GPT")):
        result = _preprocess_conversation([
            {"from": user, "value": "q"},
            {"from": asst, "value": "a"},
        ], Tok(), 32, 0)
        assert any(x != IGNORE_INDEX for x in result["labels"].tolist()), (user, asst)


def test_sft_rejects_records_with_no_assistant_targets():
    with pytest.raises(ValueError, match="no assistant targets"):
        _preprocess_conversation([{"from": "user", "value": "q"}], Tok(), 32, 0)


def test_sft_rejects_bad_arguments():
    with pytest.raises(ValueError, match="ctx_len"):
        _preprocess_conversation([{"from": "assistant", "value": "a"}], Tok(), 0, 0)
    with pytest.raises(ValueError, match="must be a list"):
        _preprocess_conversation({"from": "assistant"}, Tok(), 32, 0)


def test_sft_pads_to_ctx_len_with_ignored_labels():
    result = _preprocess_conversation(
        [{"from": "assistant", "value": "a"}], Tok(), 32, 0)
    assert result["input_ids"].numel() == 32 == result["labels"].numel()
    assert all(x == IGNORE_INDEX for x in result["labels"].tolist()[2:])


def test_sft_keeps_the_tail_when_overlong():
    """An overlong record keeps the recent turns rather than the head, so the
    only assistant turn still lands inside the window."""
    convs = [{"from": "user", "value": "filler " * 200}]
    convs.append({"from": "assistant", "value": "the real answer"})
    result = _preprocess_conversation(convs, Tok(), 16, 0)
    assert result["input_ids"].numel() == 16
    assert any(x != IGNORE_INDEX for x in result["labels"].tolist())


def test_discover_sft_records_accepts_both_shapes(tmp_path):
    from dataset import discover_sft_records
    rec = {"conversations": [{"from": "user", "value": "q"},
                             {"from": "assistant", "value": "a"}]}
    (tmp_path / "a.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "b.json").write_text(json.dumps({"data": rec}))   # wrapper object
    (tmp_path / "c.json").write_text(json.dumps([rec, rec]))      # bare list
    assert len(discover_sft_records(tmp_path)) == 4


@pytest.mark.parametrize("name,body,expected", [
    ("jsonl_plain", ("a.jsonl", lambda r: json.dumps(r)), 1),
    ("jsonl_wrapped", ("a.jsonl", lambda r: json.dumps({"data": r})), 1),
    ("json_bare_list", ("a.json", lambda r: json.dumps([r, r])), 2),
    ("json_data_list", ("a.json", lambda r: json.dumps({"data": [r, r]})), 2),
    ("json_data_object", ("a.json", lambda r: json.dumps({"data": r})), 1),
    ("json_elem_wrapped", ("a.json", lambda r: json.dumps({"data": [{"data": r}]})), 1),
    ("json_bare_object", ("a.json", lambda r: json.dumps(r)), 1),
])
def test_every_sft_document_shape_loads(tmp_path, name, body, expected):
    """All seven accepted shapes, so a wrapper is never silently dropped.

    The .json branch used to unwrap "data" only when it was a list, so a
    single wrapped export -- the whole file being
    {"data": {"conversations": ...}}, which is what a one-conversation dump
    looks like -- failed the conversations filter and loaded as an empty
    dataset. The .jsonl branch already handled it. Every row here either
    worked before or was a silent drop; none of them raised.
    """
    from dataset import discover_sft_records
    rec = {"conversations": [{"from": "user", "value": "q"}]}
    filename, render = body
    (tmp_path / filename).write_text(render(rec))
    assert len(discover_sft_records(tmp_path)) == expected, name


def test_a_literal_data_string_is_not_mistaken_for_a_wrapper(tmp_path):
    """The unwrap is type-guarded: a "data" field that is not a dict is data."""
    from dataset import discover_sft_records
    (tmp_path / "a.json").write_text(json.dumps(
        {"conversations": [{"from": "user", "value": "q"}], "data": "a note"}))
    assert len(discover_sft_records(tmp_path)) == 1


def test_discover_sft_records_raises_when_nothing_is_usable(tmp_path):
    from dataset import discover_sft_records
    (tmp_path / "a.jsonl").write_text(json.dumps({"text": "no conversations"}) + "\n")
    with pytest.raises(RuntimeError, match="No valid SFT"):
        discover_sft_records(tmp_path)


def test_discover_sft_records_skips_a_broken_file(tmp_path, capsys):
    from dataset import discover_sft_records
    rec = {"conversations": [{"from": "user", "value": "q"}]}
    (tmp_path / "a.json").write_text("{not json")
    (tmp_path / "b.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "c.jsonl").write_text("{also not json}\n")
    assert len(discover_sft_records(tmp_path)) == 1
    assert "skipping SFT file" in capsys.readouterr().out


def test_sft_dataset_caches_without_leaking_mutations(tmp_path):
    from dataset import SFTDataset
    rec = {"conversations": [{"from": "user", "value": "q"},
                             {"from": "assistant", "value": "a"}]}
    (tmp_path / "a.jsonl").write_text(json.dumps(rec) + "\n")
    ds_obj = SFTDataset(tmp_path, Tok(), 16)
    assert len(ds_obj) == 1
    first = ds_obj[0]
    first[0].fill_(999)              # caller mutates in place...
    second = ds_obj[0]
    assert not bool((second[0] == 999).any()), "the cache returned a shared tensor"


def test_sft_dataset_evicts_beyond_its_cache_limit(tmp_path):
    from dataset import SFTDataset
    rec = {"conversations": [{"from": "user", "value": "q"},
                             {"from": "assistant", "value": "a"}]}
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(rec) for _ in range(4)))
    ds_obj = SFTDataset(tmp_path, Tok(), 16)
    ds_obj._CACHE_MAX = 2
    for i in range(4):
        ds_obj[i]                          # four distinct inserts
    assert len(ds_obj._processed_cache) == 2, "the cache is unbounded, or the cap is dead"
    # It is LRU: the oldest insert went first, so the newest two are resident.
    assert set(ds_obj._processed_cache) == {2, 3}
    # A cache hit must not evict: re-reading 2 and 3 leaves the cache alone.
    ds_obj[2]; ds_obj[3]
    assert set(ds_obj._processed_cache) == {2, 3}


def test_sft_dataset_reports_which_record_is_invalid(tmp_path):
    from dataset import SFTDataset
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"conversations": [{"from": "user", "value": "q"}]}) + "\n")
    ds_obj = SFTDataset(tmp_path, Tok(), 16)
    with pytest.raises(ValueError, match="record 0 invalid"):
        ds_obj[0]

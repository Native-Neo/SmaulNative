import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sys

import pytest

import stream_data


def test_conversation_skips_malformed_rows():
    value = [None, {}, {"from": "user"}, {"from": "user", "value": " hello "}, {"value": "world"}]
    assert stream_data._conversation(value) == "user: hello\nworld"


def test_stream_validation_rejects_bad_bounds():
    with pytest.raises(ValueError):
        list(stream_data.stream_dataset("hindi", min_chars=10, max_chars=5))


def test_stream_validation_rejects_resume_dataset_mismatch():
    with pytest.raises(ValueError, match="start_dataset"):
        list(stream_data.stream_dataset("english", start_dataset="hindi"))


def test_stream_validation_requires_dataset_for_resume_file():
    with pytest.raises(ValueError, match="start_file requires start_dataset"):
        list(stream_data.stream_dataset("english", start_file="data.parquet"))


def test_stream_validation_rejects_missing_resume_file(monkeypatch):
    monkeypatch.setattr(stream_data, "_files", lambda *args, **kwargs: ["other.parquet"])
    with pytest.raises(FileNotFoundError, match="resume file not found"):
        list(stream_data.stream_dataset("english", start_dataset="english", start_file="missing.parquet"))


def test_failed_row_group_is_not_silently_skipped(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network failure")

    monkeypatch.setattr(stream_data, "_read_row_group", fail)
    monkeypatch.setattr(stream_data, "HF_TOKEN", None)

    config = {"repo_id": "repo", "path": "data"}

    class FakeFile:
        num_row_groups = 1
        schema_arrow = type("Schema", (), {"names": ["text"]})()

    class FakeParquetFile:
        def __init__(self, handle):
            self.num_row_groups = FakeFile.num_row_groups
            self.schema_arrow = FakeFile.schema_arrow

    class FakeContext:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(stream_data.pq, "ParquetFile", FakeParquetFile)
    monkeypatch.setattr(stream_data.fs, "open", lambda *args, **kwargs: FakeContext())

    with pytest.raises(RuntimeError, match="failed to read row group 0"):
        list(stream_data._stream_file(config, "test", "file.parquet", 0, 100, 0, False, 1))


# ---------------------------------------------------------------------------
# The traversal and resume logic. Everything above this point tests argument
# validation; nothing tested which files and which datasets actually get
# streamed, which is the part a resume depends on entirely.
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_repo(monkeypatch):
    """Replace the network with a fixed two-file repo per dataset."""
    calls = []

    def files(repo_id, path, retries=3):
        calls.append((repo_id, path))
        return ["a.parquet", "b.parquet"]

    def stream_file(config, dataset_name, rel_path, min_chars, max_chars, skip,
                    with_position, workers):
        # Three raw rows per file, and only `skip` of them skipped.
        for i in range(skip, 3):
            if with_position:
                yield f"{dataset_name}/{rel_path}/row{i}", (dataset_name, rel_path, i + 1)
            else:
                yield f"{dataset_name}/{rel_path}/row{i}"

    monkeypatch.setattr(stream_data, "_files", files)
    monkeypatch.setattr(stream_data, "_stream_file", stream_file)
    return calls


def test_a_resume_does_not_carry_on_into_later_datasets(fake_repo):
    """The bug: resuming inside one dataset used to stream every dataset after it.

    stream_dataset carried a comment saying later datasets do not stream, under
    an `if ...: pass` that could not do anything. With --dataset all a resume at
    english/b.parquet therefore continued through the rest of english and on
    into openthoughts, so the resumed stream no longer matched the run it was
    resuming.
    """
    got = list(stream_data.stream_dataset("all", start_dataset="english",
                                          start_file="b.parquet"))
    assert got == ["english/b.parquet/row0", "english/b.parquet/row1",
                   "english/b.parquet/row2"]
    assert all(not r.startswith("openthoughts") for r in got), got


def test_a_resume_still_streams_later_files_of_the_same_dataset(fake_repo):
    """Only the dataset boundary is a boundary; files are not."""
    got = list(stream_data.stream_dataset("english", start_dataset="english",
                                          start_file="a.parquet"))
    assert got == [f"english/{f}/row{i}" for f in ("a.parquet", "b.parquet")
                   for i in range(3)]


def test_without_a_resume_all_datasets_are_still_streamed(fake_repo):
    """The break above must not narrow the ordinary case."""
    got = list(stream_data.stream_dataset("all"))
    assert {r.split("/")[0] for r in got} == {"hindi", "english", "openthoughts"}
    assert len(got) == 18, "two files x three rows x three datasets"


def test_skip_applies_only_to_the_start_file(fake_repo):
    """start_record offsets the resume point in one file, not in the whole run."""
    got = list(stream_data.stream_dataset("english", start_dataset="english",
                                          start_file="a.parquet", start_record=2))
    assert got == ["english/a.parquet/row2",
                   "english/b.parquet/row0", "english/b.parquet/row1",
                   "english/b.parquet/row2"]


def test_positions_are_raw_row_numbers_with_gaps_by_design(fake_repo):
    """Positions count raw rows, not kept rows, so filtering leaves gaps.

    That is deliberate -- it keeps a resume position meaningful across a change
    in the filter -- so it is pinned rather than left to be "fixed".
    """
    records = list(stream_data.stream_dataset("english", min_chars=0, max_chars=10**9,
                                              with_position=True))
    for text, position in records:
        dataset, rel_path, row = position
        assert text.startswith(f"{dataset}/{rel_path}/")
        assert row == int(text.rsplit("row", 1)[1]) + 1
    assert records[0][1] == ("english", "a.parquet", 1)


def test_resume_before_the_start_dataset_ignores_earlier_ones(fake_repo):
    """hindi comes first in DATASETS, so a resume at english skips it entirely."""
    got = list(stream_data.stream_dataset("all", start_dataset="english",
                                          start_file="a.parquet"))
    assert not any(r.startswith("hindi") for r in got), got


# --- listing ---------------------------------------------------------------

def test_files_keeps_only_parquet_and_sorts(monkeypatch):
    class Item:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(stream_data, "_api", lambda: type(
        "API", (), {"list_repo_tree": lambda self, **kw: [
            Item("z.parquet"), Item("a.parquet"), Item("notes.txt"),
            Item("b.PARQUET"), Item(""),
        ]})())
    monkeypatch.setattr(stream_data.time, "sleep", lambda *_: None)
    got = stream_data._files("repo", "path")
    assert got == ["a.parquet", "b.PARQUET", "z.parquet"], "sorted, parquet only"


def test_files_retries_then_gives_up(monkeypatch):
    attempts = []

    def boom(**kw):
        attempts.append(1)
        raise OSError("network down")

    monkeypatch.setattr(stream_data, "_api", lambda: type(
        "API", (), {"list_repo_tree": lambda self, **kw: boom(**kw)})())
    slept = []
    monkeypatch.setattr(stream_data.time, "sleep", lambda s: slept.append(s))
    with pytest.raises(RuntimeError, match="could not list"):
        stream_data._files("repo", "path", retries=3)
    assert len(attempts) == 3
    assert slept, "it slept between attempts"


def test_files_succeeds_after_a_transient_failure(monkeypatch):
    state = {"n": 0}

    def flaky(**kw):
        state["n"] += 1
        if state["n"] < 3:
            raise OSError("rate limited")
        return [type("I", (), {"path": "only.parquet"})()]

    monkeypatch.setattr(stream_data, "_api", lambda: type(
        "API", (), {"list_repo_tree": lambda self, **kw: flaky(**kw)})())
    monkeypatch.setattr(stream_data.time, "sleep", lambda *_: None)
    assert stream_data._files("repo", "path", retries=3) == ["only.parquet"]
    assert state["n"] == 3


# --- the retry budget ------------------------------------------------------

def test_stream_retries_default_and_clamping(monkeypatch):
    monkeypatch.delenv("SMAUL_STREAM_RETRIES", raising=False)
    assert stream_data._stream_retries() == 6
    monkeypatch.setenv("SMAUL_STREAM_RETRIES", "0")
    assert stream_data._stream_retries() == 1, "0 must not mean no retries"
    monkeypatch.setenv("SMAUL_STREAM_RETRIES", "9999")
    assert stream_data._stream_retries() == 10, "clamped, or a typo hangs a run"
    monkeypatch.setenv("SMAUL_STREAM_RETRIES", "soon")
    with pytest.raises(ValueError, match="integer"):
        stream_data._stream_retries()


# --- row group reads -------------------------------------------------------

class _FakeFS:
    """Counts opens and raises `error` if set, else returns a handle."""
    error = None
    opens = 0
    fail_first_n = 0

    def __init__(self, token=None):
        pass

    def open(self, remote, mode):
        type(self).opens += 1
        if type(self).fail_first_n > 0:
            type(self).fail_first_n -= 1
            raise TimeoutError("read timed out")
        if type(self).error is not None:
            raise type(self).error
        # _read_row_group uses the handle as a context manager.
        import contextlib
        return contextlib.nullcontext(object())


@pytest.fixture
def fake_row_group(monkeypatch):
    """Swap the remote file system and pyarrow for counted local fakes."""
    _FakeFS.error = None
    _FakeFS.opens = 0
    _FakeFS.fail_first_n = 0
    rows = [{"text": "hello"}]

    class FakeParquetFile:
        def __init__(self, handle):
            pass

        def read_row_group(self, index, columns=None):
            return type("T", (), {"to_pylist": lambda self: list(rows)})()

    monkeypatch.setattr(stream_data, "HfFileSystem", _FakeFS)
    monkeypatch.setattr(stream_data, "pq", type("pq", (), {"ParquetFile": FakeParquetFile}))
    monkeypatch.setattr(stream_data.time, "sleep", lambda *_: None)
    monkeypatch.setattr(stream_data.random, "uniform", lambda a, b: 0.0)
    return _FakeFS


def test_row_group_read_returns_rows(fake_row_group):
    assert stream_data._read_row_group("datasets/r/f.parquet", 0, ["text"], None) == \
        [{"text": "hello"}]
    assert fake_row_group.opens == 1


def test_row_group_read_retries_transient_io(fake_row_group, monkeypatch):
    monkeypatch.setenv("SMAUL_STREAM_RETRIES", "4")
    _FakeFS.fail_first_n = 2          # two timeouts, then success
    assert stream_data._read_row_group("datasets/r/f.parquet", 0, None, None) == \
        [{"text": "hello"}]
    assert fake_row_group.opens == 3, "it did not retry through the transient failure"


def test_row_group_read_gives_up_within_its_budget(fake_row_group, monkeypatch):
    monkeypatch.setenv("SMAUL_STREAM_RETRIES", "3")
    _FakeFS.fail_first_n = 99         # never recovers
    with pytest.raises(TimeoutError):
        stream_data._read_row_group("datasets/r/f.parquet", 0, None, None)
    assert fake_row_group.opens == 3, "retried exactly the budget"


def test_a_non_parquet_file_is_not_retried(fake_row_group):
    """Retrying cannot fix a file that is not parquet; the budget is up to 6
    tries of up to 30s of backoff each.

    The guard used to test the exception's class name for the substring
    "parquet", but pyarrow raises ArrowInvalid, so it never fired.
    """
    import pyarrow as pa

    monkeypatch_env = {"SMAUL_STREAM_RETRIES": "6"}
    import os
    old = os.environ.get("SMAUL_STREAM_RETRIES")
    os.environ["SMAUL_STREAM_RETRIES"] = "6"
    try:
        _FakeFS.error = pa.ArrowInvalid(
            "Parquet magic bytes not found in footer. Either the file is "
            "corrupted or this is not a parquet file.")
        with pytest.raises(pa.ArrowInvalid, match="magic bytes"):
            stream_data._read_row_group("datasets/r/f.parquet", 0, None, None)
        assert fake_row_group.opens == 1, "a permanently bad file was retried"
    finally:
        if old is None:
            os.environ.pop("SMAUL_STREAM_RETRIES", None)
        else:
            os.environ["SMAUL_STREAM_RETRIES"] = old
    assert monkeypatch_env


def test_file_not_found_is_never_retried(fake_row_group):
    _FakeFS.error = FileNotFoundError("gone")
    with pytest.raises(FileNotFoundError):
        stream_data._read_row_group("datasets/r/f.parquet", 0, None, None)
    assert fake_row_group.opens == 1, "a missing file cannot be retried into being present"


# --- the CLI ----------------------------------------------------------------

def test_main_writes_jsonl_and_reports_the_count(monkeypatch, capsys):
    monkeypatch.setattr(stream_data, "stream_dataset",
                        lambda *a, **k: iter(["one", "two", "three"]))
    monkeypatch.setattr(sys, "argv", ["stream_data.py", "--dataset", "hindi"])
    stream_data.main()
    out = capsys.readouterr()
    lines = [ln for ln in out.out.splitlines() if ln.strip()]
    assert [json.loads(ln) for ln in lines] == [{"text": "one"}, {"text": "two"},
                                                {"text": "three"}]
    assert "streamed 3 records" in out.err


def test_main_honours_max_records(monkeypatch, capsys):
    monkeypatch.setattr(stream_data, "stream_dataset",
                        lambda *a, **k: iter([f"t{i}" for i in range(50)]))
    monkeypatch.setattr(sys, "argv", ["stream_data.py", "--max_records", "4"])
    stream_data.main()
    out = capsys.readouterr()
    lines = [ln for ln in out.out.splitlines() if ln.strip()]
    assert len(lines) == 4, "it kept going past the cap"
    assert json.loads(lines[-1]) == {"text": "t3"}


def test_main_flushes_a_partial_buffer_at_the_cap(monkeypatch, capsys):
    """The tail must not be lost when the cap lands between flushes."""
    monkeypatch.setattr(stream_data, "stream_dataset",
                        lambda *a, **k: iter(["a", "b", "c"]))
    monkeypatch.setattr(sys, "argv", ["stream_data.py", "--max_records", "2"])
    stream_data.main()
    out = capsys.readouterr()
    assert [json.loads(ln) for ln in out.out.splitlines() if ln.strip()] == \
        [{"text": "a"}, {"text": "b"}]


def test_main_survives_a_broken_pipe(monkeypatch):
    """`| head` must not produce a traceback."""
    def broken(*args, **kwargs):
        raise BrokenPipeError

    monkeypatch.setattr(sys.stdout, "write", broken)
    monkeypatch.setattr(stream_data, "stream_dataset",
                        lambda *a, **k: iter(["a", "b"]))
    monkeypatch.setattr(sys, "argv", ["stream_data.py"])
    stream_data.main()      # must return quietly


def test_main_rejects_a_negative_max_records(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stream_data.py", "--max_records", "-1"])
    with pytest.raises(SystemExit):
        stream_data.main()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def test_failed_row_group_is_not_silently_skipped(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("network failure")

    monkeypatch.setattr(stream_data, "_read_row_group", fail)
    monkeypatch.setattr(stream_data, "HF_TOKEN", None)

    config = {"repo_id": "repo", "path": "data"}

    class FakeFile:
        num_row_groups = 1
        schema_arrow = type("Schema", (), {"names": ["text"]})()

    class FakeContext:
        def __enter__(self):
            return FakeFile()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(stream_data.fs, "open", lambda *args, **kwargs: FakeContext())

    with pytest.raises(RuntimeError, match="failed to read row group 0"):
        list(stream_data._stream_file(config, "test", "file.parquet", 0, 100, 0, False, 1))

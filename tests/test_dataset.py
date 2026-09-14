import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    assert [row[0] for row in rows] == ["row1", "row2"]


def test_csv_resume_is_one_based(tmp_path):
    path = tmp_path / "x.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()
        writer.writerows({"text": f"row{i}"} for i in range(3))
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row1", "row2"]


def test_parquet_resume_is_one_based(tmp_path):
    path = tmp_path / "x.parquet"
    pq.write_table(pa.table({"text": ["row0", "row1", "row2"]}), path)
    rows = list(iter_texts([path], str(path.resolve()), 2))
    assert [row[0] for row in rows] == ["row1", "row2"]


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

########################################################################################################
# universal_dataset.py
#
# Recursively walks --dataset_dir (including subfolders, e.g. ./datasets/some_random_dataset/*)
# for .txt / .jsonl / .json / .csv / .parquet / common source-code extensions, tokenizes with the
# real RWKV World TRIE tokenizer (tokenizer/rwkv_tokenizer.py, same one upstream ships), and yields
# fixed-length ctx_len chunks for pretraining, or masked (input_ids, labels) pairs for SFT.
########################################################################################################

import csv
import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import IterableDataset, Dataset

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER

STOP_TOKEN_INDEX = 261  # matches upstream sft/src/dataset.py
IGNORE_INDEX = -100


def load_tokenizer(vocab_path: Path) -> TRIE_TOKENIZER:
    tok = TRIE_TOKENIZER(str(vocab_path))
    return tok


def tokenizer_vocab_size(tok: TRIE_TOKENIZER) -> int:
    return max(tok.idx2token.keys()) + 1


TEXT_KEYS = ("text", "content", "document", "body", "code", "prompt", "completion")

SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".jsonl", ".json", ".csv", ".parquet",
    ".py", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".rs", ".js", ".ts", ".tsx", ".jsx",
    ".java", ".go", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash",
    ".zsh", ".html", ".css", ".scss", ".sql", ".md", ".rst", ".yaml", ".yml", ".toml", ".xml",
}
PLAIN_TEXT_SUFFIXES = SUPPORTED_SUFFIXES - {".jsonl", ".json", ".csv", ".parquet"}


def discover_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    files = [p.resolve() for p in dataset_dir.rglob("*")
             if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    files.sort()
    return files


def extract_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in TEXT_KEYS:
            v = obj.get(key)
            if isinstance(v, str):
                return v
        return "\n".join(v for v in obj.values() if isinstance(v, str))
    if isinstance(obj, list):
        return "\n".join(extract_text(x) for x in obj)
    return str(obj)


def iter_texts(files: List[Path], resume_file: Optional[str] = None,
               resume_record: int = 0) -> Iterator[Tuple[str, str, int]]:
    """Yields (text, file_path_str, record_index_within_file). Resume is record-index based
    (simple + format-agnostic) rather than byte-offset based, so it works uniformly across
    txt/json/jsonl/csv/parquet without per-format seek logic."""
    started = resume_file is None
    for path in files:
        if not started:
            if str(path) == resume_file:
                started = True
            else:
                continue
        start_idx = resume_record if str(path) == resume_file else 0
        suffix = path.suffix.lower()
        try:
            if suffix in PLAIN_TEXT_SUFFIXES:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i < start_idx:
                            continue
                        line = line.strip()
                        if line:
                            yield line, str(path), i + 1
            elif suffix == ".jsonl":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i < start_idx:
                            continue
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = extract_text(obj).strip()
                        if text:
                            yield text, str(path), i + 1
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                records = data.get("data", data) if isinstance(data, dict) else data
                if not isinstance(records, list):
                    records = [records]
                for i in range(start_idx, len(records)):
                    text = extract_text(records[i]).strip()
                    if text:
                        yield text, str(path), i + 1
            elif suffix == ".csv":
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i < start_idx:
                            continue
                        text = extract_text(row).strip()
                        if text:
                            yield text, str(path), i + 1
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                rows = table.to_pylist()
                for i in range(start_idx, len(rows)):
                    text = extract_text(rows[i]).strip()
                    if text:
                        yield text, str(path), i + 1
        except Exception as e:
            print(f"[WARN] skipping {path}: {e}")
        start_idx = 0  # resume offset only applies to the exact resume file


########################################################################################################
# Pretraining: token-chunk stream
########################################################################################################

class PretrainStream(IterableDataset):
    """Streams (input_ids, labels) fixed-length chunks of size ctx_len for causal LM training."""

    def __init__(self, dataset_dir: Path, tokenizer: TRIE_TOKENIZER, ctx_len: int,
                 resume_file: Optional[str] = None, resume_record: int = 0):
        self.files = discover_files(dataset_dir)
        if not self.files:
            raise RuntimeError(f"No supported files found under {dataset_dir}")
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.resume_file = resume_file
        self.resume_record = resume_record
        self.last_pos: Tuple[Optional[str], int] = (None, 0)  # updated as we go, read by trainer for checkpointing

    def __iter__(self):
        buf: List[int] = []
        for text, path, rec_idx in iter_texts(self.files, self.resume_file, self.resume_record):
            ids = self.tokenizer.encode(text)
            ids.append(STOP_TOKEN_INDEX)
            buf.extend(ids)
            self.last_pos = (path, rec_idx)
            while len(buf) >= self.ctx_len + 1:
                chunk = buf[: self.ctx_len + 1]
                del buf[: self.ctx_len]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y, self.last_pos


########################################################################################################
# SFT: conversation JSON/JSONL with loss masking (mirrors upstream sft/src/dataset.py)
########################################################################################################

DEFAULT_STOP_TOKEN = "\n\n"


def _add_speaker_and_signal(conversations: List[Dict]) -> List[Dict]:
    out = []
    for sentence in conversations:
        frm = sentence["from"]
        frm_str = "User" if frm.lower() == "user" else "Assistant" if frm.lower() == "assistant" else frm
        val = sentence.get("value", "")
        new = dict(sentence)
        new["value"] = (frm_str + ": " + val + DEFAULT_STOP_TOKEN) if val else (frm_str + ":")
        out.append(new)
    return out


def _preprocess_conversation(conversations: List[Dict], tokenizer: TRIE_TOKENIZER, ctx_len: int,
                              pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    conversations = _add_speaker_and_signal(conversations)
    input_ids: List[int] = []
    tokenized_lens: List[int] = []
    speakers: List[str] = []
    for c in conversations:
        ids = tokenizer.encode(c["value"])
        input_ids.extend(ids)
        tokenized_lens.append(len(ids))
        speakers.append(c["from"])

    targets = list(input_ids)
    cur = 0
    for length, speaker in zip(tokenized_lens, speakers):
        if speaker.lower() == "user":
            for j in range(cur, cur + length):
                targets[j] = IGNORE_INDEX
        elif speaker.lower() == "assistant":
            for j in range(cur, min(cur + 3, cur + length)):
                targets[j] = IGNORE_INDEX
        cur += length

    input_ids = input_ids[:ctx_len]
    targets = targets[:ctx_len]
    pad_len = ctx_len - len(input_ids)
    if pad_len > 0:
        input_ids = input_ids + [pad_token_id] * pad_len
        targets = targets + [IGNORE_INDEX] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(targets, dtype=torch.long),
    }


def discover_sft_records(dataset_dir: Path) -> List[Dict]:
    """SFT data must be conversation-style JSON/JSONL: [{"conversations":[{"from":"user","value":..},
    {"from":"assistant","value":..}, ...]}], possibly nested in subfolders, any number of files."""
    records = []
    for path in discover_files(dataset_dir):
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                for line in open(path, "r", encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            else:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                records.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            print(f"[WARN] skipping SFT file {path}: {e}")
    records = [r for r in records if isinstance(r, dict) and "conversations" in r]
    if not records:
        raise RuntimeError(
            f"No SFT conversation records found under {dataset_dir}. "
            'Expected JSON/JSONL objects like {"conversations": [{"from": "user", "value": "..."}, '
            '{"from": "assistant", "value": "..."}]}'
        )
    return records


class SFTDataset(Dataset):
    def __init__(self, dataset_dir: Path, tokenizer: TRIE_TOKENIZER, ctx_len: int):
        self.records = discover_sft_records(dataset_dir)
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        d = _preprocess_conversation(rec["conversations"], self.tokenizer, self.ctx_len)
        return d["input_ids"], d["labels"]

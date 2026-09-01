# dataset.py -- recursive multi-format (txt/jsonl/json/csv/parquet/source) dataset loader, pretrain + SFT.

import csv
import json
import os
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, Dict, Any

import torch
from torch.utils.data import IterableDataset, Dataset

from tokenizers import Tokenizer as HFTokenizer

IGNORE_INDEX = -100


class TokenizerWrapper:
    """Thin wrapper around a trained tokenizers.Tokenizer (see tokenizer.py) so the rest of
    this file, train.py, and merge_moe.py only ever deal with plain encode()/decode() + two special
    token ids -- no dependency on the old TRIE_TOKENIZER / fixed-vocab-file setup."""

    def __init__(self, hf_tokenizer: HFTokenizer):
        self._tok = hf_tokenizer
        self.pad_token_id = hf_tokenizer.token_to_id("<pad>")
        self.eos_token_id = hf_tokenizer.token_to_id("<eos>")
        if self.pad_token_id is None or self.eos_token_id is None:
            raise ValueError(
                "tokenizer.json is missing <pad>/<eos> special tokens. "
                "Train it with tokenizer.py (which adds them automatically), "
                "don't point --tokenizer_path at an unrelated tokenizer.json."
            )

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()


def load_tokenizer(path: Path) -> TokenizerWrapper:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No tokenizer found at {path}. Run tokenizer.py first "
            f"(train.py will also auto-train one if --tokenizer_path is missing)."
        )
    return TokenizerWrapper(HFTokenizer.from_file(str(path)))


def tokenizer_vocab_size(tok: TokenizerWrapper) -> int:
    return tok.get_vocab_size()


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


def _looks_numeric(s: str) -> bool:
    """True for ids/scores/dates-as-strings ('42', '3.14', '2024-01-01', '-7') that shouldn't
    be mistaken for prose when no recognized text column is present."""
    s = s.strip()
    if not s:
        return True
    core = s.replace(".", "", 1).replace("-", "", 1).replace(":", "", 1).replace("/", "", 1)
    return core.isdigit()


_WARNED_FILES: set = set()


def extract_text(obj: Any, source_path: Optional[str] = None) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        for key in TEXT_KEYS:
            v = lower_map.get(key)
            if isinstance(v, str) and v.strip():
                return v
        candidates = [v for v in obj.values() if isinstance(v, str) and not _looks_numeric(v)]
        if candidates:
            if source_path and source_path not in _WARNED_FILES:
                _WARNED_FILES.add(source_path)
                print(f"[WARN] {source_path}: no column named {TEXT_KEYS} found; "
                      f"guessing text column from available keys {list(obj.keys())}. "
                      f"Rename your text column to one of {TEXT_KEYS} to silence this.")
            return max(candidates, key=len)
        return ""
    if isinstance(obj, list):
        return "\n".join(extract_text(x, source_path) for x in obj)
    return ""


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
            if suffix in (".txt", ".text"):
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    doc: List[str] = []
                    record = -1
                    for line in f:
                        line = line.rstrip()
                        if line.strip():
                            doc.append(line)
                            continue
                        if not doc:
                            continue
                        record += 1
                        if record >= start_idx:
                            yield "\n".join(doc), str(path), record + 1
                        doc = []
                    if doc:
                        record += 1
                        if record >= start_idx:
                            yield "\n".join(doc), str(path), record + 1
            elif suffix in PLAIN_TEXT_SUFFIXES:
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
                        text = extract_text(obj, str(path)).strip()
                        if text:
                            yield text, str(path), i + 1
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                records = data.get("data", data) if isinstance(data, dict) else data
                if not isinstance(records, list):
                    records = [records]
                for i in range(start_idx, len(records)):
                    text = extract_text(records[i], str(path)).strip()
                    if text:
                        yield text, str(path), i + 1
            elif suffix == ".csv":
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i < start_idx:
                            continue
                        text = extract_text(row, str(path)).strip()
                        if text:
                            yield text, str(path), i + 1
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                i = -1
                for batch in pf.iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        i += 1
                        if i < start_idx:
                            continue
                        text = extract_text(row, str(path)).strip()
                        if text:
                            yield text, str(path), i + 1
        except Exception as e:
            print(f"[WARN] skipping {path}: {e}")


class PretrainStream(IterableDataset):
    def __init__(self, dataset_dir: Path, tokenizer: TokenizerWrapper, ctx_len: int,
                 resume_file: Optional[str] = None, resume_record: int = 0,
                 buffer_tokens: Optional[List[int]] = None):
        self.files = discover_files(dataset_dir)
        if not self.files:
            raise RuntimeError(f"No supported files found under {dataset_dir}")
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.resume_file = resume_file
        self.resume_record = resume_record
        self.buffer_tokens = list(buffer_tokens) if buffer_tokens is not None else []
        self.last_pos: Tuple[Optional[str], int] = (None, 0)

    def __iter__(self):
        buf: List[int] = list(self.buffer_tokens)
        for text, path, rec_idx in iter_texts(self.files, self.resume_file, self.resume_record):
            ids = self.tokenizer.encode(text)
            ids.append(self.tokenizer.eos_token_id)
            buf.extend(ids)
            self.last_pos = (path, rec_idx)
            while len(buf) >= self.ctx_len + 1:
                chunk = buf[: self.ctx_len + 1]
                del buf[: self.ctx_len]
                self.buffer_tokens = list(buf)
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y, self.last_pos


DEFAULT_STOP_TOKEN = "\n\n"


def _add_speaker_and_signal(conversations: List[Dict]) -> List[Dict]:
    out = []
    for sentence in conversations:
        frm = sentence["from"]
        frm_str = "User" if frm.lower() in ("user", "human") else "Assistant" if frm.lower() in ("assistant", "gpt") else frm
        val = sentence.get("value", "")
        new = dict(sentence)
        new["from"] = frm_str
        new["value"] = frm_str + ": " + val + DEFAULT_STOP_TOKEN
        out.append(new)
    return out


def _preprocess_conversation(conversations: List[Dict], tokenizer: TokenizerWrapper, ctx_len: int,
                              pad_token_id: int) -> Dict[str, torch.Tensor]:
    conversations = _add_speaker_and_signal(conversations)
    input_ids: List[int] = []
    tokenized_lens: List[int] = []
    speakers: List[str] = []
    prefix_lens: List[int] = []
    for c in conversations:
        ids = tokenizer.encode(c["value"])
        input_ids.extend(ids)
        tokenized_lens.append(len(ids))
        speakers.append(c["from"])
        prefix_lens.append(len(tokenizer.encode(c["from"] + ": ")))

    targets = list(input_ids)
    cur = 0
    for length, speaker, prefix_len in zip(tokenized_lens, speakers, prefix_lens):
        if speaker.lower() == "user":
            for j in range(cur, cur + length):
                targets[j] = IGNORE_INDEX
        elif speaker.lower() == "assistant":
            for j in range(cur, min(cur + prefix_len, cur + length)):
                targets[j] = IGNORE_INDEX
        cur += length

    input_ids = input_ids[:ctx_len]
    targets = targets[:ctx_len]
    pad_len = ctx_len - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_token_id] * pad_len
        targets += [IGNORE_INDEX] * pad_len

    return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(targets, dtype=torch.long)}


def discover_sft_records(dataset_dir: Path) -> List[Dict]:
    records = []
    for path in discover_files(dataset_dir):
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
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
    def __init__(self, dataset_dir: Path, tokenizer: TokenizerWrapper, ctx_len: int):
        records = discover_sft_records(dataset_dir)
        self.examples = [
            _preprocess_conversation(r["conversations"], tokenizer, ctx_len, tokenizer.pad_token_id)
            for r in records
        ]
        del records

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        d = self.examples[idx]
        return d["input_ids"], d["labels"]

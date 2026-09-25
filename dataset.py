# dataset.py -- multi-format dataset loader, pretrain + SFT.

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, IterableDataset

IGNORE_INDEX = -100


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self._tok = tokenizer
        self.pad_token_id = tokenizer.token_to_id("<pad>")
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        if self.pad_token_id is None or self.eos_token_id is None:
            raise ValueError("tokenizer.json is missing <pad>/<eos> special tokens")

    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)

    def get_vocab_size(self) -> int:
        return self._tok.get_vocab_size()


def load_tokenizer(path: Path) -> TokenizerWrapper:
    if not Path(path).exists():
        raise FileNotFoundError(f"No tokenizer found at {path}. Run tokenizer.py first")
    try:
        from tokenizer import SmaulTokenizer
    except ImportError:
        # Support `python -m package` / relative layouts.
        from .tokenizer import SmaulTokenizer  # type: ignore
    return TokenizerWrapper(SmaulTokenizer.from_file(path))


def tokenizer_vocab_size(tok) -> int:
    if isinstance(tok, (str, Path)):
        tok = load_tokenizer(Path(tok))
    return tok.get_vocab_size()


TEXT_KEYS = (
    "text", "content", "document", "body", "code", "prompt", "completion",
)
SUPPORTED_SUFFIXES = {
    ".txt", ".text", ".jsonl", ".json", ".csv", ".parquet", ".py", ".cpp", ".c", ".h", ".hpp",
    ".cc", ".cxx", ".rs", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb",
    ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".html", ".css", ".scss", ".sql",
    ".md", ".rst", ".yaml", ".yml", ".toml", ".xml",
}
PLAIN_TEXT_SUFFIXES = SUPPORTED_SUFFIXES - {".jsonl", ".json", ".csv", ".parquet"}


def discover_files(dataset_dir: Path) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    root = dataset_dir.resolve()
    files: List[Path] = []
    for p in root.rglob("*"):
        # Skip symlinks that escape the dataset root and cap scan size.
        try:
            if p.is_symlink() and p.resolve() != p and root not in p.resolve().parents:
                print(f"[WARN] skipping symlink escaping dataset root: {p}")
                continue
            if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
                files.append(p.resolve())
        except OSError:
            continue
        if len(files) > 100_000:
            print("[WARN] file scan capped at 100k files")
            break
    files.sort()
    return files


def _looks_numeric(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    # Robust: timestamps (12:30:45), fractions (1/2), ranges (--) are numeric.
    allowed = set("0123456789.,-: /+%_")
    return all(c in allowed for c in t)


_WARNED_FILES: set = set()
_WARNED_COUNT: int = 0


def extract_text(obj: Any, source_path: Optional[str] = None, _depth: int = 0) -> str:
    if _depth > 8:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        lower_map = {str(k).lower(): v for k, v in obj.items()}
        prompt = lower_map.get("prompt")
        completion = lower_map.get("completion")
        if isinstance(prompt, str) and isinstance(completion, str):
            return prompt + "\n" + completion
        for key in TEXT_KEYS:
            v = lower_map.get(key)
            if isinstance(v, str) and v.strip():
                return v
        candidates = [v for v in obj.values() if isinstance(v, str) and not _looks_numeric(v)]
        if candidates:
            global _WARNED_COUNT
            if source_path and source_path not in _WARNED_FILES:
                _WARNED_FILES.add(source_path)
                _WARNED_COUNT += 1
                if _WARNED_COUNT <= 5:
                    print(f"[WARN] {source_path}: no recognized text column; guessing from {list(obj.keys())}")
            # Prefer sentence-like candidates (contain spaces) over IDs/hashes.
            spaced = [c for c in candidates if " " in c.strip() and len(c.strip()) > 20]
            pool = spaced or candidates
            return max(pool, key=len)
        return ""
    if isinstance(obj, list):
        return "\n".join(extract_text(x, source_path, _depth + 1) for x in obj)
    return ""


MAX_TEXT_FILE_BYTES = 10_000_000
MAX_JSON_FILE_BYTES = 50_000_000


def _open_text(path: Path):
    # utf-8-sig strips BOM; errors=replace keeps one bad file from killing training.
    return open(path, "r", encoding="utf-8-sig", errors="replace")


def iter_texts(files: List[Path], resume_file: Optional[str] = None, resume_record: int = 0) -> Iterator[Tuple[str, str, int]]:
    if resume_record < 0:
        raise ValueError("resume_record must be non-negative")
    if resume_file is not None:
        resume_file = str(Path(resume_file).resolve())
        resolved_files = {str(path.resolve()) for path in files}
        if resume_file not in resolved_files:
            raise FileNotFoundError(f"resume file not found in discovered dataset files: {resume_file}")
    started = resume_file is None
    skipped_jsonl = 0
    for path in files:
        if not started:
            if str(path.resolve()) == resume_file:
                started = True
            else:
                continue
        start_idx = resume_record if str(path.resolve()) == resume_file else 0
        suffix = path.suffix.lower()
        try:
            if suffix in (".txt", ".text"):
                with _open_text(path) as f:
                    doc = []
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
                try:
                    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
                        print(f"[WARN] skipping oversized text file {path} "
                              f"({path.stat().st_size} bytes > {MAX_TEXT_FILE_BYTES})")
                        continue
                except OSError:
                    pass
                with _open_text(path) as f:
                    content = f.read().strip()
                # Preserve code indentation: only strip trailing/leading blank lines,
                # not inner leading spaces (already handled by read().strip() on ends).
                if content and start_idx == 0:
                    yield content, str(path), 1
            elif suffix == ".jsonl":
                with _open_text(path) as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        record = i + 1
                        if record <= start_idx:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            skipped_jsonl += 1
                            continue
                        text = extract_text(obj, str(path)).strip()
                        if text:
                            yield text, str(path), record
            elif suffix == ".json":
                try:
                    if path.stat().st_size > MAX_JSON_FILE_BYTES:
                        print(f"[WARN] skipping oversized JSON file {path}")
                        continue
                except OSError:
                    pass
                data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
                # Only unwrap {"data": [...]} when data is a list; a legit
                # string field named "data" must not misfire.
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    records = data["data"]
                else:
                    records = data
                if not isinstance(records, list):
                    records = [records]
                for i, record_obj in enumerate(records, 1):
                    if i <= start_idx:
                        continue
                    text = extract_text(record_obj, str(path)).strip()
                    if text:
                        yield text, str(path), i
            elif suffix == ".csv":
                import csv as _csv
                _csv.field_size_limit(min(10_000_000, max(131072, _csv.field_size_limit())))
                with _open_text(path) as f:
                    reader = _csv.DictReader(f)
                    if reader.fieldnames is None:
                        print(f"[WARN] skipping CSV with missing header: {path}")
                        continue
                    for i, row in enumerate(reader, 1):
                        if i <= start_idx:
                            continue
                        text = extract_text(row, str(path)).strip()
                        if text:
                            yield text, str(path), i
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                try:
                    pf = pq.ParquetFile(path)
                except ImportError as exc:
                    raise RuntimeError("parquet support requires pyarrow") from exc
                schema_names = pf.schema_arrow.names
                schema_lower = [c.lower() for c in schema_names]
                fast_col = None
                for cand in TEXT_KEYS:
                    if cand in schema_lower:
                        fast_col = schema_names[schema_lower.index(cand)]
                        break
                prompt_col = schema_names[schema_lower.index("prompt")] if "prompt" in schema_lower else None
                completion_col = schema_names[schema_lower.index("completion")] if "completion" in schema_lower else None
                record = 0
                columns = [fast_col] if fast_col is not None else None
                if prompt_col and completion_col and prompt_col != completion_col:
                    columns = [prompt_col, completion_col]
                for batch in pf.iter_batches(batch_size=1024, columns=columns):
                    if prompt_col and completion_col and prompt_col != completion_col:
                        prompt_data = batch.column(prompt_col).to_pylist()
                        completion_data = batch.column(completion_col).to_pylist()
                        for prompt, completion in zip(prompt_data, completion_data):
                            record += 1
                            if record <= start_idx:
                                continue
                            text = extract_text({"prompt": prompt, "completion": completion}, str(path)).strip()
                            if text:
                                yield text, str(path), record
                    elif fast_col is not None:
                        col = batch.column(fast_col)
                        for row_idx in range(batch.num_rows):
                            record += 1
                            if record <= start_idx:
                                continue
                            val = col[row_idx].as_py()
                            if isinstance(val, str) and val.strip():
                                yield val.strip(), str(path), record
                    else:
                        for row in batch.to_pylist():
                            record += 1
                            if record <= start_idx:
                                continue
                            text = extract_text(row, str(path)).strip()
                            if text:
                                yield text, str(path), record
        except RuntimeError:
            raise
        except Exception as e:
            # Warn-and-skip: one bad file must not abort the whole stream.
            print(f"[WARN] skipping dataset file {path}: {type(e).__name__}: {e}")
            continue
    if skipped_jsonl:
        print(f"[WARN] skipped {skipped_jsonl} malformed JSONL line(s)")


class PretrainStream(IterableDataset):
    def __init__(self, dataset_dir: Path, tokenizer: TokenizerWrapper, ctx_len: int, resume_file: Optional[str] = None,
                 resume_record: int = 0, buffer_tokens: Optional[List[int]] = None):
        self.files = discover_files(dataset_dir)
        if not self.files:
            raise RuntimeError(f"No supported files found under {dataset_dir}")
        if ctx_len < 1:
            raise ValueError("ctx_len must be positive")
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.resume_file = resume_file
        self.resume_record = resume_record
        self.buffer_tokens = list(buffer_tokens) if buffer_tokens is not None else []
        self.last_pos: Tuple[Optional[str], int] = (None, 0)

    def __iter__(self):
        buf = list(self.buffer_tokens)
        subchunk = 4096
        # Cap single-document encode to avoid RAM spikes on huge docs.
        max_doc_chars = 100_000
        if buf and self.resume_file is not None:
            self.last_pos = (str(Path(self.resume_file).resolve()), self.resume_record)
            while len(buf) >= self.ctx_len + 1:
                chunk = buf[:self.ctx_len + 1]
                del buf[:self.ctx_len]
                self.buffer_tokens = list(buf)
                yield (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long), self.last_pos)
        for text, path, rec_idx in iter_texts(self.files, self.resume_file, self.resume_record):
            if len(text) > max_doc_chars:
                print(f"[WARN] truncating oversized document ({len(text)} chars) from {path}")
                text = text[:max_doc_chars]
            ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            # The buffer contains the remainder of this record after each
            # yielded chunk, so resume must start at the following record.
            self.last_pos = (path, rec_idx + 1)
            for i in range(0, len(ids), subchunk):
                buf.extend(ids[i:i + subchunk])
                while len(buf) >= self.ctx_len + 1:
                    chunk = buf[:self.ctx_len + 1]
                    del buf[:self.ctx_len]
                    self.buffer_tokens = list(buf)
                    yield (torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long), self.last_pos)


DEFAULT_STOP_TOKEN = "\n\n"


def _add_speaker_and_signal(conversations: List[Dict]) -> List[Dict]:
    out = []
    for sentence in conversations:
        if not isinstance(sentence, dict):
            continue
        frm = sentence.get("from")
        value = sentence.get("value")
        if not isinstance(frm, str) or not isinstance(value, str):
            continue
        role = frm.strip().lower()
        if role in ("user", "human"):
            normalized = "User"
        elif role in ("assistant", "gpt"):
            normalized = "Assistant"
        else:
            normalized = "Other"
        out.append({"from": normalized, "value": normalized + ": " + value + DEFAULT_STOP_TOKEN})
    return out


def _preprocess_conversation(conversations: List[Dict], tokenizer: TokenizerWrapper, ctx_len: int, pad_token_id: int) -> Dict[str, torch.Tensor]:
    if ctx_len < 1:
        raise ValueError("ctx_len must be positive")
    if not isinstance(conversations, list):
        raise ValueError("SFT record 'conversations' must be a list")
    input_ids, tokenized_lens, speakers, prefix_lens = [], [], [], []
    for c in _add_speaker_and_signal(conversations):
        turn_ids = tokenizer.encode(c["value"])
        if not turn_ids:
            continue
        prefix_ids = tokenizer.encode(c["from"] + ": ")
        # Prefix encoded in isolation may not match in-context tokenization.
        # Only use it when it matches the turn start; otherwise include the
        # whole turn (slight prefix leak beats masking response tokens).
        if prefix_ids and turn_ids[:len(prefix_ids)] == prefix_ids:
            prefix_len = len(prefix_ids)
        else:
            prefix_len = 0
        input_ids.extend(turn_ids)
        tokenized_lens.append(len(turn_ids))
        speakers.append(c["from"])
        prefix_lens.append(prefix_len)
    if not input_ids:
        raise ValueError("SFT record contains no valid conversation turns")
    targets = [IGNORE_INDEX] * len(input_ids)
    cur = 0
    for length, speaker, prefix_len in zip(tokenized_lens, speakers, prefix_lens):
        if speaker.lower() == "assistant":
            start = cur + min(prefix_len, length)
            targets[start:cur + length] = input_ids[start:cur + length]
        cur += length
    # Overlong: keep the TAIL (recent assistant turns) instead of crashing
    # when no targets fall in the head window.
    if len(input_ids) > ctx_len + 1:
        input_ids = input_ids[-(ctx_len + 1):]
        targets = targets[-(ctx_len + 1):]
    if not any(x != IGNORE_INDEX for x in targets[1:]):
        raise ValueError("SFT record contains no assistant targets within ctx_len")
    input_ids = input_ids[:-1]
    targets = targets[1:]
    pad_len = ctx_len - len(input_ids)
    if pad_len:
        input_ids.extend([pad_token_id] * pad_len)
        targets.extend([IGNORE_INDEX] * pad_len)
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long), "labels": torch.tensor(targets, dtype=torch.long)}


def discover_sft_records(dataset_dir: Path) -> List[Dict]:
    records = []
    for path in discover_files(dataset_dir):
        if path.suffix.lower() not in (".json", ".jsonl"):
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            print(f"[WARN] skipping malformed SFT record in {path}")
                            continue
                        # Accept both {"conversations": [...]} and {"data": {"conversations": ...}}.
                        if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
                            obj = obj["data"]
                        records.append(obj)
            else:
                data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
                if isinstance(data, dict) and isinstance(data.get("data"), list):
                    records.extend(data["data"])
                else:
                    records.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            print(f"[WARN] skipping SFT file {path}: {type(e).__name__}: {e}")
            continue
    records = [r for r in records if isinstance(r, dict) and isinstance(r.get("conversations"), list)
               and r["conversations"]]
    if not records:
        raise RuntimeError(f"No valid SFT conversation records found under {dataset_dir}")
    if len(records) > 200_000:
        print(f"[WARN] {len(records):,} SFT records materialized in RAM; consider sharding")
    return records


class SFTDataset(Dataset):
    _CACHE_MAX = 2048
    def __init__(self, dataset_dir: Path, tokenizer: TokenizerWrapper, ctx_len: int):
        if ctx_len < 1:
            raise ValueError("ctx_len must be positive")
        self.records = discover_sft_records(dataset_dir)
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.pad_token_id = tokenizer.pad_token_id
        self._processed_cache: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        if idx in self._processed_cache:
            self._processed_cache.move_to_end(idx)
            ids, labels = self._processed_cache[idx]
            # Return clones: callers/collators may mutate in place.
            return ids.clone(), labels.clone()
        try:
            d = _preprocess_conversation(self.records[idx]["conversations"], self.tokenizer, self.ctx_len, self.pad_token_id)
        except ValueError as exc:
            raise ValueError(f"SFT record {idx} invalid: {exc}") from exc
        item = (d["input_ids"], d["labels"])
        self._processed_cache[idx] = (item[0].clone(), item[1].clone())
        if len(self._processed_cache) > self._CACHE_MAX:
            self._processed_cache.popitem(last=False)
        return item[0].clone(), item[1].clone()

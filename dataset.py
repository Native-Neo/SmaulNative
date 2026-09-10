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
    from tokenizer import SmaulTokenizer
    return TokenizerWrapper(SmaulTokenizer.from_file(path))


def tokenizer_vocab_size(tok: TokenizerWrapper) -> int:
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
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    files = [p.resolve() for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    files.sort()
    return files


def _looks_numeric(s: str) -> bool:
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
            if source_path and source_path not in _WARNED_FILES:
                _WARNED_FILES.add(source_path)
                print(f"[WARN] {source_path}: no recognized text column; guessing from {list(obj.keys())}")
            return max(candidates, key=len)
        return ""
    if isinstance(obj, list):
        return "\n".join(extract_text(x, source_path) for x in obj)
    return ""


def iter_texts(files: List[Path], resume_file: Optional[str] = None, resume_record: int = 0) -> Iterator[Tuple[str, str, int]]:
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
                with open(path, "r", encoding="utf-8") as f:
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
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content and start_idx == 0:
                    yield content, str(path), 1
            elif suffix == ".jsonl":
                with open(path, "r", encoding="utf-8") as f:
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
                data = json.loads(path.read_text(encoding="utf-8"))
                records = data.get("data", data) if isinstance(data, dict) else data
                if not isinstance(records, list):
                    records = [records]
                for i in range(start_idx, len(records)):
                    text = extract_text(records[i], str(path)).strip()
                    if text:
                        yield text, str(path), i + 1
            elif suffix == ".csv":
                with open(path, "r", encoding="utf-8", newline="") as f:
                    for i, row in enumerate(csv.DictReader(f)):
                        if i < start_idx:
                            continue
                        text = extract_text(row, str(path)).strip()
                        if text:
                            yield text, str(path), i + 1
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(path)
                schema_names = pf.schema_arrow.names
                schema_lower = [c.lower() for c in schema_names]
                fast_col = None
                for cand in TEXT_KEYS:
                    if cand in schema_lower:
                        fast_col = schema_names[schema_lower.index(cand)]
                        break
                prompt_col = schema_names[schema_lower.index("prompt")] if "prompt" in schema_lower else None
                completion_col = schema_names[schema_lower.index("completion")] if "completion" in schema_lower else None
                i = -1
                columns = [fast_col] if fast_col is not None else None
                if prompt_col and completion_col and prompt_col != completion_col:
                    columns = [prompt_col, completion_col]
                for batch in pf.iter_batches(batch_size=1024, columns=columns):
                    if prompt_col and completion_col and prompt_col != completion_col:
                        prompt_data = batch.column(prompt_col).to_pylist()
                        completion_data = batch.column(completion_col).to_pylist()
                        for prompt, completion in zip(prompt_data, completion_data):
                            i += 1
                            if i < start_idx:
                                continue
                            text = extract_text({"prompt": prompt, "completion": completion}, str(path)).strip()
                            if text:
                                yield text, str(path), i + 1
                    elif fast_col is not None:
                        col = batch.column(fast_col)
                        for row_idx in range(batch.num_rows):
                            i += 1
                            if i < start_idx:
                                continue
                            val = col[row_idx].as_py()
                            if isinstance(val, str) and val.strip():
                                yield val.strip(), str(path), i + 1
                    else:
                        for row in batch.to_pylist():
                            i += 1
                            if i < start_idx:
                                continue
                            text = extract_text(row, str(path)).strip()
                            if text:
                                yield text, str(path), i + 1
        except Exception as e:
            raise RuntimeError(f"failed to read dataset file {path}") from e


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
        for text, path, rec_idx in iter_texts(self.files, self.resume_file, self.resume_record):
            ids = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
            self.last_pos = (path, rec_idx)
            for i in range(0, len(ids), subchunk):
                buf.extend(ids[i:i + subchunk])
                while len(buf) >= self.ctx_len + 1:
                    chunk = buf[:self.ctx_len + 1]
                    del buf[:self.ctx_len]
                    self.buffer_tokens = buf
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
    if not isinstance(conversations, list):
        raise ValueError("SFT record 'conversations' must be a list")
    input_ids, tokenized_lens, speakers, prefix_lens = [], [], [], []
    for c in _add_speaker_and_signal(conversations):
        ids = tokenizer.encode(c["value"])
        input_ids.extend(ids)
        tokenized_lens.append(len(ids))
        speakers.append(c["from"])
        prefix_lens.append(len(tokenizer.encode(c["from"] + ": ")))
    if not input_ids:
        raise ValueError("SFT record contains no valid conversation turns")
    targets = [IGNORE_INDEX] * len(input_ids)
    cur = 0
    for length, speaker, prefix_len in zip(tokenized_lens, speakers, prefix_lens):
        if speaker.lower() == "assistant":
            targets[cur + min(prefix_len, length):cur + length] = input_ids[cur + min(prefix_len, length):cur + length]
        cur += length
    input_ids = input_ids[:ctx_len]
    targets = targets[:ctx_len]
    if not any(x != IGNORE_INDEX for x in targets):
        raise ValueError("SFT record contains no assistant targets within ctx_len")
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
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f"[WARN] skipping malformed SFT record in {path}")
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
                records.extend(data if isinstance(data, list) else [data])
        except Exception as e:
            raise RuntimeError(f"failed to read SFT file {path}") from e
    records = [r for r in records if isinstance(r, dict) and isinstance(r.get("conversations"), list)]
    if not records:
        raise RuntimeError(f"No valid SFT conversation records found under {dataset_dir}")
    return records


class SFTDataset(Dataset):
    _CACHE_MAX = 2048
    def __init__(self, dataset_dir: Path, tokenizer: TokenizerWrapper, ctx_len: int):
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
            return self._processed_cache[idx]
        d = _preprocess_conversation(self.records[idx]["conversations"], self.tokenizer, self.ctx_len, self.pad_token_id)
        item = (d["input_ids"], d["labels"])
        self._processed_cache[idx] = item
        if len(self._processed_cache) > self._CACHE_MAX:
            self._processed_cache.popitem(last=False)
        return item

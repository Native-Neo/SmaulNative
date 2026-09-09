#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
CASE = ["<cap>", "<upper>"]
TOKEN_RE = re.compile(
    r"\s+|[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+|\d+(?:\.\d+)?|==|!=|<=|>=|=>|->|::|//|\*\*|&&|\|\||[^\w\s]",
    re.UNICODE,
)
DEV_BASE = re.compile(r"[\u0900-\u097F]")


class TokenIds(list):

    @property
    def ids(self):
        return self


def _json_texts(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        prompt = data.get("prompt")
        completion = data.get("completion")
        if isinstance(prompt, str):
            yield prompt
        if isinstance(completion, str):
            yield completion
        if not isinstance(prompt, str) and not isinstance(completion, str):
            text = next((data[k] for k in ("text", "content", "document", "body", "code")
                         if isinstance(data.get(k), str)), None)
            if text is not None:
                yield text
            elif "data" in data:
                yield from _json_texts(data["data"])
    elif isinstance(data, list):
        for x in data:
            yield from _json_texts(x)


def read_texts(path, max_records=0):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    seen = 0
    plain = {
        ".txt", ".text", ".py", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".rs", ".js", ".ts",
        ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala",
        ".sh", ".bash", ".zsh", ".html", ".css", ".scss", ".sql", ".md", ".rst", ".yaml", ".yml",
        ".toml", ".xml",
    }

    for f in files:
        ext = f.suffix.lower()
        if ext in plain:
            with f.open("r", encoding="utf-8") as h:
                text = h.read()
            if text:
                yield text
                seen += 1
        elif ext == ".jsonl":
            with f.open("r", encoding="utf-8") as h:
                for line in h:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for text in _json_texts(data):
                        yield text
                        seen += 1
                        if max_records and seen >= max_records:
                            return
        elif ext == ".json":
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            for text in _json_texts(data):
                yield text
                seen += 1
                if max_records and seen >= max_records:
                    return
        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise SystemExit("Parquet support: pip install pyarrow")
            pf = pq.ParquetFile(f)
            names = pf.schema_arrow.names
            lower = {name.lower(): name for name in names}
            col = next((lower[name] for name in ("text", "content", "document", "body", "code") if name in lower), None)
            if col:
                for batch in pf.iter_batches(batch_size=1024, columns=[col]):
                    for x in batch.column(0).to_pylist():
                        if isinstance(x, str):
                            yield x
                            seen += 1
                            if max_records and seen >= max_records:
                                return
        if max_records and seen >= max_records:
            return


def tokenize_text(text):
    return TOKEN_RE.findall(text)


def devanagari_units(text):
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if not DEV_BASE.fullmatch(c):
            out.append(c)
            i += 1
            continue
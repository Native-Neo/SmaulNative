#!/usr/bin/env python3
"""Byte-level codec for SmaulNative.

The model path is true byte-level: vocabulary is exactly the 256 UTF-8 byte
values, text is encoded once via UTF-8, and the model consumes byte IDs
directly. There is no BPE/WordPiece/SentencePiece, no word tokenizer, and no
giant vocabulary in this path.

The corpus text-extraction helpers (``read_texts`` and the record coercers)
are kept: reading heterogeneous dataset files into plain strings is still
needed before the single UTF-8 encoding step. Everything word/vocab-building
(``TOKEN_RE``, ``_build``, case markers, Devanagari units, guaranteed sets)
was specific to the old subword/word tokenizer and is gone.
"""

import argparse
import json
from pathlib import Path

BYTE_VOCAB_SIZE = 256
VERSION = 9
KIND = "byte"

MAX_PLAIN_BYTES = 10_000_000


class TokenIds(list):
    @property
    def ids(self):
        return list(self)


def encode_bytes(text: str) -> list:
    """Encode text to raw UTF-8 byte IDs (each 0-255)."""
    if not isinstance(text, str):
        raise ValueError(f"encode requires str, got {type(text).__name__}")
    return list(text.encode("utf-8"))


def decode_bytes(ids, errors: str = "replace") -> str:
    """Decode byte IDs to text, safely.

    Out-of-range or non-int entries raise; incomplete/invalid UTF-8 is
    handled per ``errors`` (default ``replace`` so arbitrary byte sequences
    never crash and never silently drop bytes without a marker).
    """
    buf = bytearray()
    for i in ids:
        if isinstance(i, bool) or not isinstance(i, int):
            raise ValueError(f"invalid byte id {i!r}: expected int in [0, 255]")
        if not 0 <= i <= 255:
            raise ValueError(f"invalid byte id {i!r}: expected int in [0, 255]")
        buf.append(i)
    return bytes(buf).decode("utf-8", errors=errors)


def _incomplete_tail_len(buf: bytes) -> int:
    """Length of a trailing incomplete UTF-8 sequence (0 if none)."""
    n = len(buf)
    if n == 0:
        return 0
    # Count trailing continuation bytes (10xxxxxx).
    cont = 0
    i = n - 1
    while i >= 0 and (buf[i] & 0xC0) == 0x80:
        cont += 1
        i -= 1
    if i < 0:
        return n  # only continuation bytes; wait for a lead byte
    lead = buf[i]
    if lead < 0x80:
        return 0
    if 0xC2 <= lead <= 0xDF:
        need = 1
    elif 0xE0 <= lead <= 0xEF:
        need = 2
    elif 0xF0 <= lead <= 0xF4:
        need = 3
    else:
        return 0  # invalid lead; let errors=replace surface it
    have = cont
    return (cont + 1) if have < need else 0


class IncrementalByteDecoder:
    """Streaming byte->text decoder that waits for split multi-byte chars.

    Complete prefixes decode immediately; a trailing incomplete UTF-8
    sequence is buffered until more bytes arrive (or ``flush()`` replaces
    it, so generation never hangs and never silently drops bytes).
    """

    def __init__(self):
        self._buf = bytearray()

    def push(self, byte_id: int) -> str:
        if isinstance(byte_id, bool) or not isinstance(byte_id, int) \
                or not 0 <= byte_id <= 255:
            raise ValueError(f"invalid byte id {byte_id!r}")
        self._buf.append(byte_id)
        return self._emit(flush=False)

    def push_ids(self, ids) -> str:
        out = []
        for i in ids:
            out.append(self.push(int(i)))
        return "".join(out)

    def _emit(self, flush: bool) -> str:
        raw = bytes(self._buf)
        if not raw:
            return ""
        if not flush:
            tail = _incomplete_tail_len(raw)
            if tail:
                head = raw[:-tail] if tail < len(raw) else b""
                text = head.decode("utf-8", errors="replace")
                self._buf = bytearray(raw[len(head):])
                return text
        text = raw.decode("utf-8", errors="replace")
        self._buf = bytearray()
        return text

    def flush(self) -> str:
        return self._emit(flush=True)


class SmaulTokenizer:
    """Byte-level tokenizer: fixed 256-entry vocabulary, no training.

    Kept under the historical name so ``dataset.load_tokenizer``,
    ``rawr_graph.build_graph`` and inference keep importing the same symbol.
    ``ByteTokenizer`` is the same class under its accurate name.
    """

    kind = KIND
    unk_token_id = None
    pad_token_id = None
    bos_token_id = None
    eos_token_id = None

    def __init__(self, data=None):
        data = dict(data) if data else {}
        kind = data.get("kind", KIND)
        if kind != KIND:
            raise ValueError(
                f"legacy word-level tokenizer (kind={kind!r}, version={data.get('version')!r}); "
                "byte-level checkpoints need a byte tokenizer "
                f"(kind={KIND!r}). Delete the file and re-run: a byte "
                "tokenizer is written automatically, no training needed.")
        self.data = {"version": VERSION, "kind": KIND,
                     "vocab_size": BYTE_VOCAB_SIZE}
        self.vocab_size = BYTE_VOCAB_SIZE
        # Byte id -> single-byte latin-1 char, for display/debug only.
        self.id_to_token = {i: chr(i) for i in range(256)}
        self.vocab = {chr(i): i for i in range(256)}

    @classmethod
    def from_file(cls, path):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise RuntimeError(f"could not load tokenizer {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"tokenizer {path} must contain a JSON object")
        if data.get("kind", KIND) != KIND or data.get("version") != VERSION:
            # Old word-level files carry kind != byte or version 8.
            if data.get("kind") != KIND:
                raise ValueError(
                    f"tokenizer {path} is a legacy word-level file "
                    f"(version={data.get('version')!r}); byte-level runs need "
                    f"kind={KIND!r}. Delete it and re-run.")
            raise ValueError(
                f"tokenizer {path} version {data.get('version')!r} != {VERSION}; "
                "delete it and re-run")
        return cls(data)

    def save(self, path):
        import os
        import tempfile
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, separators=(",", ":"))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get_vocab_size(self): return BYTE_VOCAB_SIZE
    def token_to_id(self, token): return None
    def encode(self, text): return TokenIds(encode_bytes(text))
    def decode(self, ids): return decode_bytes(ids)


ByteTokenizer = SmaulTokenizer


def load(path): return SmaulTokenizer.from_file(path)


def encode(text, tok):
    if isinstance(tok, SmaulTokenizer):
        return tok.encode(text)
    return TokenIds(encode_bytes(text))


def decode(ids, tok):
    return decode_bytes(ids)


def train_tokenizer(dataset_dir, output_path, vocab_size=BYTE_VOCAB_SIZE,
                    stream_name="none", max_records=0, texts=None):
    """Write a byte tokenizer file (no training; vocabulary is fixed)."""
    if vocab_size != BYTE_VOCAB_SIZE:
        raise ValueError(
            f"byte vocabulary is exactly {BYTE_VOCAB_SIZE}; got {vocab_size}")
    tok = SmaulTokenizer()
    tok.save(output_path)
    return tok


def ensure_tokenizer(output_path, texts_or_dir=None, vocab_size=BYTE_VOCAB_SIZE,
                     max_records=0, stream_name="none"):
    if vocab_size != BYTE_VOCAB_SIZE:
        raise ValueError(
            f"byte vocabulary is exactly {BYTE_VOCAB_SIZE}; got {vocab_size}")
    path = Path(output_path)
    if path.exists():
        try:
            tok = load(path)
            if tok.get_vocab_size() == BYTE_VOCAB_SIZE:
                return tok
        except (RuntimeError, ValueError):
            print("[TOKENIZER] existing file is legacy/unreadable; writing byte tokenizer")
    else:
        print(f"[TOKENIZER] writing byte vocabulary={BYTE_VOCAB_SIZE}")
    return train_tokenizer(None, path, vocab_size)


# ---------------------------------------------------------------------------
# Corpus text extraction (kept): dataset files -> plain strings, encoded to
# UTF-8 bytes exactly once downstream by SmaulTokenizer.encode.
# ---------------------------------------------------------------------------

TEXT_KEYS = ("text", "content", "document", "body", "code", "prompt", "completion",
             "input", "output", "question", "answer")


def _coerce_list_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                for k in ("value", "text", "content", "completion", "output", "answer"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                        break
        return "\n".join(parts) if parts else ""
    return ""


def _string_values(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        lower = {str(k).lower(): v for k, v in data.items()}
        prompt, completion = lower.get("prompt"), lower.get("completion")
        prompt_s, completion_s = _coerce_list_text(prompt), _coerce_list_text(completion)
        if prompt_s:
            yield prompt_s + ("\n" + completion_s if completion_s else "")
            return
        if completion_s:
            yield completion_s
            return
        found = False
        for key in TEXT_KEYS:
            value = lower.get(key)
            coerced = _coerce_list_text(value)
            if coerced and coerced.strip():
                found = True
                yield coerced
        if not found:
            for value in data.values():
                yield from _string_values(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _string_values(value)


def _record_text(record):
    lower = {str(k).lower(): v for k, v in record.items()}
    prompt, completion = lower.get("prompt"), lower.get("completion")
    prompt_s, completion_s = _coerce_list_text(prompt), _coerce_list_text(completion)
    if prompt_s:
        return prompt_s + ("\n" + completion_s if completion_s else "")
    if completion_s:
        return completion_s
    for key in TEXT_KEYS:
        value = _coerce_list_text(lower.get(key))
        if value and value.strip():
            return value
    return "\n".join(x for value in record.values() for x in _string_values(value))


def read_texts(path, max_records=0):
    import csv
    path = Path(path)
    if path.is_file():
        files = [path]
    else:
        root = path.resolve()
        files = []
        for p in sorted(root.rglob("*")):
            try:
                if not p.is_file():
                    continue
                if p.is_symlink() and root not in p.resolve().parents:
                    continue
            except OSError:
                continue
            files.append(p)
    seen = 0
    plain = {".txt", ".text", ".py", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".rs",
             ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb", ".swift",
             ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".html", ".css", ".scss",
             ".sql", ".md", ".rst", ".yaml", ".yml", ".toml", ".xml"}
    for f in files:
        ext = f.suffix.lower()
        if ext in plain:
            try:
                if f.stat().st_size > MAX_PLAIN_BYTES:
                    print(f"[WARN] skipping oversized text file {f}")
                    continue
            except OSError:
                pass
            try:
                text = f.read_text(encoding="utf-8-sig", errors="replace")
            except (OSError, UnicodeError) as exc:
                print(f"[WARN] skipping unreadable file {f}: {exc}")
                continue
            if text and text.strip():
                for para in text.split("\n\n"):
                    para = para.strip()
                    if not para:
                        continue
                    yield para
                    seen += 1
                    if max_records and seen >= max_records:
                        return
        elif ext == ".csv":
            try:
                import csv as _csv
                _csv.field_size_limit(min(10_000_000, max(131072, _csv.field_size_limit())))
                with f.open("r", encoding="utf-8-sig", errors="replace", newline="") as h:
                    reader = _csv.DictReader(h)
                    if reader.fieldnames is None:
                        continue
                    for row in reader:
                        text = _record_text(row)
                        if text and text.strip():
                            yield text
                            seen += 1
                        if max_records and seen >= max_records:
                            return
            except OSError as exc:
                print(f"[WARN] skipping CSV {f}: {exc}")
                continue
        elif ext in {".json", ".jsonl"}:
            if ext == ".jsonl":
                try:
                    source = f.open("r", encoding="utf-8-sig", errors="replace")
                except OSError:
                    continue
                close = True
            else:
                try:
                    if f.stat().st_size > 50_000_000:
                        print(f"[WARN] skipping oversized JSON {f}")
                        continue
                    source = [f.read_text(encoding="utf-8-sig", errors="replace")]
                except OSError:
                    continue
                close = False
            try:
                for raw in source:
                    raw = raw.strip() if ext == ".jsonl" else raw
                    if not raw.strip():
                        continue
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    for text in _string_values(data):
                        if text and text.strip():
                            yield text
                            seen += 1
                        if max_records and seen >= max_records:
                            return
            finally:
                if close:
                    source.close()
        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise ImportError("Parquet support: pip install pyarrow") from exc
            try:
                pf = pq.ParquetFile(f)
            except Exception as exc:
                print(f"[WARN] skipping parquet {f}: {exc}")
                continue
            names = pf.schema_arrow.names
            lower = {str(name).lower(): name for name in names}
            prompt_col, completion_col = lower.get("prompt"), lower.get("completion")
            if prompt_col and completion_col and prompt_col != completion_col:
                columns = [prompt_col, completion_col]
                for batch in pf.iter_batches(batch_size=4096, columns=columns):
                    prompts, completions = batch.column(0).to_pylist(), batch.column(1).to_pylist()
                    for prompt, completion in zip(prompts, completions):
                        prompt_s, completion_s = _coerce_list_text(prompt), _coerce_list_text(completion)
                        text = prompt_s + ("\n" + completion_s if completion_s else "") if prompt_s else completion_s
                        if text and text.strip():
                            yield text
                            seen += 1
                        if max_records and seen >= max_records:
                            return
            else:
                preferred = next((lower[k] for k in TEXT_KEYS if k in lower), None)
                if preferred:
                    columns = [preferred]
                else:
                    try:
                        string_cols = [f.name for f in pf.schema_arrow
                                       if str(f.type).startswith("string")]
                    except Exception:
                        string_cols = names
                    columns = string_cols or names
                for batch in pf.iter_batches(batch_size=4096, columns=columns):
                    if preferred:
                        rows = ((value,) for value in batch.column(0).to_pylist())
                    else:
                        rows = zip(*(batch.column(i).to_pylist() for i in range(len(columns))))
                    for row in rows:
                        text = "\n".join(x for value in row for x in _string_values(value) if x)
                        if text and text.strip():
                            yield text
                            seen += 1
                        if max_records and seen >= max_records:
                            return
        if max_records and seen >= max_records:
            return


def main():
    p = argparse.ArgumentParser(description="Byte-level codec (vocab is fixed at 256)")
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("encode")
    x.add_argument("--tokenizer", default=None,
                   help="Optional byte tokenizer file (validated, not trained)")
    x.add_argument("--text", default=None)
    x.add_argument("--text-file", default=None, help="Read text from file")
    x.set_defaults(f=encode_cmd)
    x = s.add_parser("decode")
    x.add_argument("--tokenizer", default=None)
    x.add_argument("--ids", required=True)
    x.set_defaults(f=decode_cmd)
    a = p.parse_args()
    a.f(a)


def encode_cmd(a):
    if a.tokenizer:
        load(a.tokenizer)  # validate only; encoding needs no table
    if a.text_file:
        if Path(a.text_file).stat().st_size > MAX_PLAIN_BYTES:
            raise ValueError(f"--text-file exceeds {MAX_PLAIN_BYTES} bytes")
        text = Path(a.text_file).read_text(encoding="utf-8-sig", errors="replace")
    elif a.text is not None:
        text = a.text
    else:
        raise ValueError("encode requires --text or --text-file")
    print(*encode_bytes(text))


def decode_cmd(a):
    if a.tokenizer:
        load(a.tokenizer)
    try:
        ids = [int(v) for v in a.ids.split()]
    except ValueError as exc:
        raise ValueError(f"invalid --ids {a.ids!r}: expected space-separated ints") from exc
    print(decode_bytes(ids))


if __name__ == "__main__": main()

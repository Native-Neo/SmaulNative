#!/usr/bin/env python3
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
CASE = ["<cap>", "<upper>"]
CHATML_TAG = re.compile(r"<\|im_start\|>|<\|im_end\|>|</?think>")
# NOTE: Latin words split on hyphens/camelCase by design ("well-known" -> well,-,known;
# "eBay" lowercases lossily, see case_type). Multi-char operators are single tokens.
# Numbers cover ASCII + Devanagari digits (U+0966-096F).
_DIGIT = r"(?:\d|[\u0966-\u096F])"
TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|</?think>|\s+|[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F\u200C\u200D]+|" + _DIGIT + r"+(?:\." + _DIGIT + r"+)?|==|!=|<=|>=|=>|->|::|//|\*\*|&&|\|\||[^\w\s]", re.UNICODE)
DEV_BASE = re.compile(r"[\u0900-\u097F]")
TEXT_KEYS = ("text", "content", "document", "body", "code", "prompt", "completion", "input", "output", "question", "answer")
VERSION = 7

class TokenIds(list):
    @property
    def ids(self):
        return list(self)

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

MAX_PLAIN_BYTES = 10_000_000

def read_texts(path, max_records=0):
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
                # Skip symlinks escaping the corpus root.
                if p.is_symlink() and root not in p.resolve().parents:
                    continue
            except OSError:
                continue
            files.append(p)
    seen = 0
    plain = {".txt", ".text", ".py", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".rs", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".html", ".css", ".scss", ".sql", ".md", ".rst", ".yaml", ".yml", ".toml", ".xml"}
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
                # Split into paragraphs so plain files count like JSONL lines.
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
                    # Fallback: string columns only (skip binary/image).
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

def _clean(text):
    # Preserve ChatML tags as tokens (matched by TOKEN_RE); only normalize
    # whitespace control chars here.
    return text


def tokenize_text(text):
    return TOKEN_RE.findall(text)

def devanagari_units(text):
    out, i = [], 0
    ZWJ = "\u200d"
    ZWNJ = "\u200c"
    cat = unicodedata.category
    while i < len(text):
        c = text[i]
        if not DEV_BASE.fullmatch(c):
            out.append(c)
            i += 1
            continue
        u = c
        i += 1
        while i < len(text):
            c = text[i]
            if c == "्":
                u += c
                i += 1
                while i < len(text) and (cat(text[i]).startswith("M") or text[i] in (ZWJ, ZWNJ)):
                    u += text[i]
                    i += 1
                if i < len(text) and DEV_BASE.fullmatch(text[i]):
                    u += text[i]
                    i += 1
                continue
            if cat(c).startswith("M") or c in (ZWJ, ZWNJ):
                u += c
                i += 1
                continue
            break
        out.append(u)
    return out

def canonical(x):
    return x.lower()

def case_type(x):
    # Lossy by design for mixed case (hELLO/eBay -> lowercase, no marker):
    # vocab stays small at the cost of exact round-trip for odd casing.
    if x.isupper():
        return "upper"
    if x[:1].isupper() and x[1:].islower():
        return "cap"
    return None

def _guaranteed():
    chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    chars.update([" ", "\n", "\t", "  ", "   ", "\n\n", ".", ",", "!", "?", ";", ":", "-", "(", ")", "[", "]", "{", "}", "'", '"', "/", "\\", "|", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "~", "`"])
    for cp in range(0x0900, 0x0980):
        try:
            chars.add(chr(cp))
        except ValueError:
            pass
    return sorted(chars)

def _build(texts, vocab_size, word_budget=40000, max_records=0):
    if vocab_size < len(SPECIAL) + len(CASE) + 1:
        raise ValueError(f"vocab_size must be at least {len(SPECIAL) + len(CASE) + 1}")
    if word_budget < 0 or max_records < 0:
        raise ValueError("word_budget and max_records must be non-negative")
    words, graphemes, chars, symbols = Counter(), Counter(), Counter(), Counter()
    cases = defaultdict(Counter)
    total_words = total_tokens = seen = 0
    for text in texts:
        seen += 1
        text = _clean(text)
        chars.update(text)
        for token in TOKEN_RE.findall(text):
            total_tokens += 1
            if token.isspace(): continue
            if token.isalpha() or token.isdigit():
                base = canonical(token); words[base] += 1; total_words += 1
                case = case_type(token)
                if case: cases[base][case] += 1
                if DEV_BASE.search(token): graphemes.update(devanagari_units(token))
            else:
                symbols[token] += 1
        if max_records and seen >= max_records: break
    if seen == 0 or total_tokens == 0:
        raise RuntimeError("no usable text records found for tokenizer training")
    tokens, seen_tokens = SPECIAL + CASE, set(SPECIAL + CASE)
    for g in _guaranteed():
        if g not in seen_tokens:
            tokens.append(g); seen_tokens.add(g)
    whitespace = [(x, n) for x, n in chars.most_common() if x.isspace() and x not in seen_tokens]
    for source in (whitespace, words.most_common(word_budget), graphemes.most_common(), symbols.most_common(), chars.most_common()):
        for x, _ in source:
            if x not in seen_tokens:
                tokens.append(x); seen_tokens.add(x)
            if len(tokens) >= vocab_size: break
        if len(tokens) >= vocab_size: break
    while len(tokens) < vocab_size:
        token = f"<unused_{len(tokens)}>"
        tokens.append(token)
        seen_tokens.add(token)
    vocab = {x: i for i, x in enumerate(tokens)}
    # case_stats intentionally NOT saved: it bloated tokenizer.json and was
    # never used by encode/decode.
    return {"version": VERSION, "vocab": vocab, "special_tokens": SPECIAL, "case_tokens": CASE, "unk_id": vocab["<unk>"], "stats": {"vocab_size": len(vocab), "whole_words": min(word_budget, len(words)), "unique_words": len(words), "total_words": total_words, "total_tokens": total_tokens, "devanagari_units": len(graphemes), "characters": len(chars), "symbols": len(symbols)}}

def train(dataset, vocab_size=32000, word_budget=20000, max_records=0):
    return _build(read_texts(Path(dataset), max_records), vocab_size, word_budget, max_records)

class SmaulTokenizer:
    def __init__(self, data):
        if not isinstance(data, dict) or "vocab" not in data:
            raise ValueError("invalid tokenizer data: missing 'vocab'")
        self.data = data
        self.vocab = data["vocab"]
        self.id_to_token = {}
        for x, i in self.vocab.items():
            try:
                self.id_to_token[int(i)] = x
            except (TypeError, ValueError):
                raise ValueError(f"invalid vocab id for {x!r}: {i!r}")
        try:
            self.unk_token_id = data["unk_id"]
            self.pad_token_id = self.vocab["<pad>"]
            self.bos_token_id = self.vocab["<bos>"]
            self.eos_token_id = self.vocab["<eos>"]
        except KeyError as exc:
            raise ValueError(f"tokenizer missing required special token {exc}") from exc
    @classmethod
    def from_file(cls, path):
        try:
            return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise RuntimeError(f"could not load tokenizer {path}: {exc}") from exc
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
    def get_vocab_size(self): return len(self.vocab)
    def token_to_id(self, token): return self.vocab.get(token)
    def encode(self, text): return TokenIds(encode(text, self))
    def decode(self, ids): return decode(ids, self)

def load(path): return SmaulTokenizer.from_file(path)

def encode(text, tok):
    v, u = tok.vocab, tok.unk_token_id; cap, upper = v.get("<cap>"), v.get("<upper>"); out = []
    for t in tokenize_text(text):
        if t.isspace():
            out.extend(v.get(c, u) for c in t); continue
        b = canonical(t)
        if b in v:
            c = case_type(t)
            if c == "cap" and cap is not None: out.append(cap)
            elif c == "upper" and upper is not None: out.append(upper)
            out.append(v[b]); continue
        if DEV_BASE.search(t):
            for g in devanagari_units(t): out.extend([v[g]] if g in v else [v.get(c, u) for c in g])
        else: out.extend(v.get(c, u) for c in t)
    return out

def decode(ids, tok):
    tab, out, case = tok.id_to_token, [], None
    for i in ids:
        try:
            t = tab.get(int(i), "<unk>")
        except (TypeError, ValueError):
            t = "<unk>"
        if t == "<cap>":
            case = "cap"
            continue
        if t == "<upper>":
            case = "upper"
            continue
        if t in {"<pad>", "<bos>", "<eos>"}:
            continue
        if t.startswith("<unused_"):
            # Surface invalid IDs instead of silently dropping them.
            out.append("<unk>")
            case = None
            continue
        if case == "cap":
            t = t[:1].upper() + t[1:]
        elif case == "upper":
            t = t.upper()
        out.append(t)
        case = None
    return "".join(out)

def train_tokenizer(dataset_dir, output_path, vocab_size=32000, stream_name="none", max_records=0, texts=None):
    if texts is not None:
        data = _build(texts, vocab_size, min(20000, vocab_size // 2), max_records)
    elif stream_name != "none":
        if not max_records or max_records <= 0:
            raise ValueError("--max-records must be positive when streaming (refusing unbounded FineWeb download)")
        from stream_data import stream_dataset
        data = _build(stream_dataset(stream_name), vocab_size, min(20000, vocab_size // 2), max_records)
    else:
        data = train(dataset_dir, vocab_size=vocab_size, max_records=max_records)
    tok = SmaulTokenizer(data)
    tok.save(output_path)
    return tok

def ensure_tokenizer(output_path, texts_or_dir=None, vocab_size=32000, max_records=0, stream_name="none"):
    if texts_or_dir is not None and not isinstance(texts_or_dir, (str, Path)):
        path = Path(output_path)
        if path.exists():
            try:
                tok = load(path)
                if tok.get_vocab_size() == vocab_size and tok.data.get("version") == VERSION:
                    return tok
                print(f"[TOKENIZER] rebuild: existing={tok.get_vocab_size()} v={tok.data.get('version')} requested={vocab_size} v={VERSION}")
            except Exception:
                print("[TOKENIZER] existing file unreadable; rebuilding")
        else:
            print(f"[TOKENIZER] creating vocabulary={vocab_size}")
        return train_tokenizer(path.parent, path, vocab_size, max_records=max_records, texts=texts_or_dir)
    dataset_dir = texts_or_dir if texts_or_dir is not None else output_path
    path = Path(output_path)
    if isinstance(dataset_dir, (str, Path)) and Path(dataset_dir).suffix == ".json":
        dataset_dir = Path(dataset_dir).parent
    if path.exists():
        try:
            tok = load(path)
            if tok.get_vocab_size() == vocab_size and tok.data.get("version") == VERSION:
                return tok
            print(f"[TOKENIZER] rebuild: existing={tok.get_vocab_size()} v={tok.data.get('version')} requested={vocab_size} v={VERSION}")
        except Exception:
            print("[TOKENIZER] existing file unreadable; rebuilding")
    else:
        print(f"[TOKENIZER] creating vocabulary={vocab_size}")
    if stream_name != "none":
        return train_tokenizer(dataset_dir, path, vocab_size, stream_name, max_records)
    return train_tokenizer(dataset_dir if isinstance(dataset_dir, (str, Path)) else "./datasets", path, vocab_size, max_records=max_records)

def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("train")
    x.add_argument("--fromdataset", required=True)
    x.add_argument("--vocab-size", type=int, default=32000)
    x.add_argument("--word-budget", type=int, default=20000)
    x.add_argument("--max-records", type=int, default=0)
    x.add_argument("--output", default="tokenizer.json")
    x.set_defaults(f=train_cmd)
    x = s.add_parser("encode")
    x.add_argument("--tokenizer", required=True)
    x.add_argument("--text", default=None)
    x.add_argument("--text-file", default=None, help="Read text from file (for Hindi/newlines)")
    x.set_defaults(f=encode_cmd)
    x = s.add_parser("decode")
    x.add_argument("--tokenizer", required=True)
    x.add_argument("--ids", required=True)
    x.set_defaults(f=decode_cmd)
    a = p.parse_args()
    a.f(a)


def encode_cmd(a):
    if a.text_file:
        text = Path(a.text_file).read_text(encoding="utf-8-sig", errors="replace")
    elif a.text is not None:
        text = a.text
    else:
        raise ValueError("encode requires --text or --text-file")
    print(*load(a.tokenizer).encode(text))


def decode_cmd(a):
    try:
        ids = [int(v) for v in a.ids.split()]
    except ValueError as exc:
        raise ValueError(f"invalid --ids {a.ids!r}: expected space-separated ints") from exc
    print(load(a.tokenizer).decode(ids))

def train_cmd(a):
    if a.vocab_size <= 0 or a.word_budget < 0 or a.max_records < 0:
        raise ValueError("--vocab-size/--word-budget/--max-records invalid")
    d = _build((t for t in read_texts(Path(a.fromdataset), a.max_records)), a.vocab_size, a.word_budget, a.max_records)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    import os
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=out.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, out)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    s = d["stats"]
    print(f"Vocabulary: {s['vocab_size']:,}\nWhole words: {s['whole_words']:,}\nUnique words: {s['unique_words']:,}\nCorpus words: {s['total_words']:,}\nDevanagari units: {s['devanagari_units']:,}\nCharacters: {s['characters']:,}\nSymbols/operators: {s['symbols']:,}\nSaved: {a.output}")

if __name__ == "__main__": main()

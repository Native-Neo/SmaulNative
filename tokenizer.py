#!/usr/bin/env python3
import argparse
import csv
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
TEXT_KEYS = ("text", "content", "document", "body", "code", "prompt", "completion", "input", "output", "question", "answer")


class TokenIds(list):
    @property
    def ids(self):
        return self


def _string_values(data):
    if isinstance(data, str):
        yield data
    elif isinstance(data, dict):
        lower = {str(k).lower(): v for k, v in data.items()}
        used = False
        prompt = lower.get("prompt")
        completion = lower.get("completion")
        if isinstance(prompt, str):
            used = True
            if isinstance(completion, str):
                yield prompt + "\n" + completion
            else:
                yield prompt
        elif isinstance(completion, str):
            used = True
            yield completion
        if not used:
            for key in TEXT_KEYS:
                value = lower.get(key)
                if isinstance(value, str):
                    used = True
                    yield value
            if used:
                return
            for value in data.values():
                yield from _string_values(value)
    elif isinstance(data, (list, tuple)):
        for value in data:
            yield from _string_values(value)


def _json_texts(data):
    yield from _string_values(data)


def _record_text(record):
    lower = {str(k).lower(): v for k, v in record.items()}
    prompt = lower.get("prompt")
    completion = lower.get("completion")
    if isinstance(prompt, str) and isinstance(completion, str):
        return prompt + "\n" + completion
    if isinstance(prompt, str):
        return prompt
    if isinstance(completion, str):
        return completion
    for key in TEXT_KEYS:
        value = lower.get(key)
        if isinstance(value, str):
            return value
    values = [x for value in record.values() for x in _string_values(value)]
    return "\n".join(values)


def read_texts(path, max_records=0):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    seen = 0
    plain = {".txt", ".text", ".py", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx", ".rs", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".cs", ".php", ".rb", ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".html", ".css", ".scss", ".sql", ".md", ".rst", ".yaml", ".yml", ".toml", ".xml"}
    for f in files:
        ext = f.suffix.lower()
        if ext in plain:
            with f.open("r", encoding="utf-8") as h:
                text = h.read()
            if text:
                yield text
                seen += 1
        elif ext == ".csv":
            with f.open("r", encoding="utf-8", newline="") as h:
                for row in csv.DictReader(h):
                    text = _record_text(row)
                    if not text:
                        continue
                    yield text
                    seen += 1
                    if max_records and seen >= max_records:
                        return
        elif ext == ".jsonl":
            with f.open("r", encoding="utf-8") as h:
                for line in h:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for text in _json_texts(data):
                        if not text:
                            continue
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
                if not text:
                    continue
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
            lower = {str(name).lower(): name for name in names}
            prompt_col = lower.get("prompt")
            completion_col = lower.get("completion")
            if prompt_col and completion_col and prompt_col != completion_col:
                columns = [prompt_col, completion_col]
                for batch in pf.iter_batches(batch_size=1024, columns=columns):
                    prompts = batch.column(0).to_pylist()
                    completions = batch.column(1).to_pylist()
                    for prompt, completion in zip(prompts, completions):
                        if isinstance(prompt, str) and isinstance(completion, str):
                            text = prompt + "\n" + completion
                        elif isinstance(prompt, str):
                            text = prompt
                        elif isinstance(completion, str):
                            text = completion
                        else:
                            text = ""
                        if text:
                            yield text
                            seen += 1
                        if max_records and seen >= max_records:
                            return
            else:
                preferred = next((lower[name] for name in TEXT_KEYS if name in lower), None)
                columns = [preferred] if preferred else names
                for batch in pf.iter_batches(batch_size=1024, columns=columns):
                    rows = zip(*(batch.column(i).to_pylist() for i in range(len(columns))))
                    for row in rows:
                        if preferred:
                            values = _string_values(row[0])
                        else:
                            values = (text for value in row for text in _string_values(value))
                        text = "\n".join(x for x in values if x)
                        if not text:
                            continue
                        yield text
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
        u = c
        i += 1
        while i < len(text):
            c = text[i]
            if unicodedata.combining(c) or c in "\u200c\u200d":
                u += c
                i += 1
                continue
            if c == "्":
                u += c
                i += 1
                if i < len(text) and DEV_BASE.fullmatch(text[i]):
                    u += text[i]
                    i += 1
                continue
            break
        out.append(u)
    return out


def canonical(x):
    return x.lower()


def case_type(x):
    letters = "".join(c for c in x if c.isalpha())
    if not letters:
        return None
    if letters.isupper():
        return "upper"
    if x[:1].isupper() and x[1:].lower() == x[1:]:
        return "cap"
    return None


def _build(texts, vocab_size, word_budget, max_records=0):
    if vocab_size < len(SPECIAL) + len(CASE):
        raise ValueError(f"vocab_size must be at least {len(SPECIAL) + len(CASE)}")
    if word_budget < 0 or max_records < 0:
        raise ValueError("word_budget and max_records must be non-negative")
    words, graphemes, chars, symbols = Counter(), Counter(), Counter(), Counter()
    cases = defaultdict(Counter)
    total_words = total_tokens = seen = 0
    for text in texts:
        seen += 1
        for token in tokenize_text(text):
            total_tokens += 1
            if token.isspace():
                chars.update(token)
                continue
            if token.isalpha() or token.isdigit():
                base = canonical(token)
                words[base] += 1
                total_words += 1
                case = case_type(token)
                if case:
                    cases[base][case] += 1
                if DEV_BASE.search(token):
                    graphemes.update(devanagari_units(token))
                chars.update(token)
            else:
                symbols[token] += 1
                chars.update(token)
        if max_records and seen >= max_records:
            break
    if seen == 0 or total_tokens == 0:
        raise RuntimeError("no usable text records found for tokenizer training")
    tokens = SPECIAL + CASE
    seen_tokens = set(tokens)
    for x, _ in words.most_common(word_budget):
        if x not in seen_tokens:
            tokens.append(x)
            seen_tokens.add(x)
        if len(tokens) >= vocab_size:
            break
    for x, _ in graphemes.most_common():
        if x not in seen_tokens:
            tokens.append(x)
            seen_tokens.add(x)
        if len(tokens) >= vocab_size:
            break
    for x, _ in symbols.most_common():
        if x not in seen_tokens:
            tokens.append(x)
            seen_tokens.add(x)
        if len(tokens) >= vocab_size:
            break
    for x, _ in chars.most_common():
        if x not in seen_tokens:
            tokens.append(x)
            seen_tokens.add(x)
        if len(tokens) >= vocab_size:
            break
    vocab = {x: i for i, x in enumerate(tokens)}
    return {"version": 5, "vocab": vocab, "special_tokens": SPECIAL, "case_tokens": CASE, "case_stats": {w: dict(c) for w, c in cases.items()}, "unk_id": vocab["<unk>"], "stats": {"vocab_size": len(vocab), "whole_words": min(word_budget, len(words)), "unique_words": len(words), "total_words": total_words, "total_tokens": total_tokens, "devanagari_units": len(graphemes), "characters": len(chars), "symbols": len(symbols)}}


def train(dataset, vocab_size=64000, word_budget=40000, max_records=0):
    return _build(read_texts(Path(dataset), max_records), vocab_size, word_budget, max_records)


class SmaulTokenizer:
    def __init__(self, data):
        self.data = data
        self.vocab = data["vocab"]
        self.id_to_token = {int(i): x for x, i in self.vocab.items()}
        self.unk_token_id = data["unk_id"]
        self.pad_token_id = self.vocab["<pad>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    def get_vocab_size(self):
        return len(self.vocab)

    def token_to_id(self, token):
        return self.vocab.get(token)

    def encode(self, text):
        return TokenIds(encode(text, self))

    def decode(self, ids):
        return decode(ids, self)



def load(path):
    return SmaulTokenizer.from_file(path)


def encode(text, tok):
    v, u = tok.vocab, tok.unk_token_id
    cap, upper = v.get("<cap>"), v.get("<upper>")
    out = []
    for t in tokenize_text(text):
        if t.isspace():
            out.extend(v.get(c, u) for c in t)
            continue
        b = canonical(t)
        if b in v:
            c = case_type(t)
            if c == "cap" and cap is not None:
                out.append(cap)
            elif c == "upper" and upper is not None:
                out.append(upper)
            out.append(v[b])
            continue
        if DEV_BASE.search(t):
            for g in devanagari_units(t):
                out.extend([v[g]] if g in v else [v.get(c, u) for c in g])
        else:
            out.extend(v.get(c, u) for c in t)
    return out


def decode(ids, tok):
    tab = tok.id_to_token
    out, case = [], None
    for i in ids:
        t = tab.get(int(i), "<unk>")
        if t == "<cap>":
            case = "cap"
            continue
        if t == "<upper>":
            case = "upper"
            continue
        if t in {"<pad>", "<bos>", "<eos>"}:
            continue
        if case == "cap":
            t = t[:1].upper() + t[1:]
        elif case == "upper":
            t = t.upper()
        out.append(t)
        case = None
    return "".join(out)


def train_tokenizer(dataset_dir, output_path, vocab_size=64000, stream_name="none", max_records=0):
    if stream_name != "none":
        from stream_data import stream_dataset
        texts = stream_dataset(stream_name)
        data = _build(texts, vocab_size, 40000, max_records)
    else:
        data = train(dataset_dir, vocab_size=vocab_size, max_records=max_records)
    tok = SmaulTokenizer(data)
    tok.save(output_path)
    return tok


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="cmd", required=True)
    x = s.add_parser("train")
    x.add_argument("--fromdataset", required=True)
    x.add_argument("--vocab-size", type=int, default=64000)
    x.add_argument("--word-budget", type=int, default=40000)
    x.add_argument("--max-records", type=int, default=0)
    x.add_argument("--output", default="tokenizer.json")
    x.set_defaults(f=lambda a: train_cmd(a))
    x = s.add_parser("encode")
    x.add_argument("--tokenizer", required=True)
    x.add_argument("--text", required=True)
    x.set_defaults(f=lambda a: print(*load(a.tokenizer).encode(a.text)))
    x = s.add_parser("decode")
    x.add_argument("--tokenizer", required=True)
    x.add_argument("--ids", required=True)
    x.set_defaults(f=lambda a: print(load(a.tokenizer).decode(a.ids.split())))
    a = p.parse_args()
    a.f(a)


def train_cmd(a):
    d = _build(read_texts(Path(a.fromdataset), a.max_records), a.vocab_size, a.word_budget, a.max_records)
    Path(a.output).write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    s = d["stats"]
    print(f"Vocabulary: {s['vocab_size']:,}\nWhole words: {s['whole_words']:,}\nUnique words: {s['unique_words']:,}\nCorpus words: {s['total_words']:,}\nDevanagari units: {s['devanagari_units']:,}\nCharacters: {s['characters']:,}\nSymbols/operators: {s['symbols']:,}\nSaved: {a.output}")


if __name__ == "__main__":
    main()

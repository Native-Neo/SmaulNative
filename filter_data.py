#!/usr/bin/env python3
"""Lightweight filters for streamed training records."""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from typing import Any, Optional

_URL_ONLY = re.compile(r"^(?:https?://|www\.)\S+$", re.I)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
_LATIN = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    # Preserve code indentation: only strip blank lines at the ends, not spaces.
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    return text.strip("\n").strip("\u200b\u200c\u200d\ufeff").strip()


def _ratio(pattern: re.Pattern[str], text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    return len(pattern.findall(text)) / max(letters, 1)


# Keys recognized across dataset.py / tokenizer.py / download.py.
TEXT_KEYS = ("text", "content", "document", "body", "code", "prompt", "completion")


def _coerce_str(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Chat history: [{from,value}...] or [str...].
        parts = []
        for item in value:
            if isinstance(item, str) and item.strip():
                parts.append(item.strip())
            elif isinstance(item, dict):
                v = item.get("value", item.get("text", item.get("content")))
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
        return "\n".join(parts) if parts else None
    return None


def filter_text(text: Any, dataset: str = "auto", min_chars: int = 20, max_chars: int = 1_000_000,
                min_unique: int = 8, min_dev: int = 8, min_latin: int = 12,
                dev_ratio: float = 0.20, latin_ratio: float = 0.50) -> Optional[str]:
    if min_chars < 0 or max_chars < 0 or min_chars > max_chars:
        raise ValueError(f"require 0 <= min_chars <= max_chars, got {min_chars}/{max_chars}")
    if not isinstance(text, str):
        return None
    # Cheap length guard BEFORE expensive NFKC + set() on huge inputs (DoS).
    if len(text) < min_chars or len(text) > max_chars * 4 or len(text) > 2_000_000:
        return None
    text = normalize_text(text)
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if "\ufffd" in text or _URL_ONLY.fullmatch(text):
        return None
    # set() on 1M-char strings is slow; sample first 50k chars for diversity.
    if len(set(text.replace(" ", "")[:50_000])) < min_unique:
        return None
    if dataset == "hindi":
        if len(_DEVANAGARI.findall(text)) < min_dev or _ratio(_DEVANAGARI, text) < dev_ratio:
            return None
    elif dataset == "english":
        if len(_LATIN.findall(text)) < min_latin or _ratio(_LATIN, text) < latin_ratio:
            return None
    # "auto"/"openthoughts": no script gate (lenient); mixed bilingual text
    # should use --dataset auto or tune thresholds explicitly.
    return text


def filter_record(record: dict[str, Any], dataset: str = "auto", min_chars: int = 20,
                  max_chars: int = 1_000_000, **kwargs) -> Optional[str]:
    if not isinstance(record, dict):
        return None
    lower = {str(key).lower(): value for key, value in record.items()}
    prompt = _coerce_str(lower.get("prompt"))
    completion = _coerce_str(lower.get("completion"))
    if prompt and completion:
        value: Any = prompt + "\n" + completion
    elif prompt:
        value = prompt
    elif completion:
        value = completion
    else:
        value = None
        for key in TEXT_KEYS:
            coerced = _coerce_str(lower.get(key))
            if coerced and coerced.strip():
                value = coerced
                break
    return filter_text(value, dataset, min_chars, max_chars, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Filter JSONL text from stdin without storing the dataset")
    p.add_argument("--dataset", choices=["auto", "hindi", "english", "openthoughts"], default="auto")
    p.add_argument("--min_chars", type=int, default=20)
    p.add_argument("--max_chars", type=int, default=1_000_000)
    p.add_argument("--min_unique", type=int, default=8)
    p.add_argument("--min_dev", type=int, default=8)
    p.add_argument("--min_latin", type=int, default=12)
    p.add_argument("--dev_ratio", type=float, default=0.20)
    p.add_argument("--latin_ratio", type=float, default=0.50)
    args = p.parse_args()
    kept = 0
    for line in sys.stdin:
        try:
            record = json.loads(line)
            text = filter_record(record, args.dataset, args.min_chars, args.max_chars,
                                 min_unique=args.min_unique, min_dev=args.min_dev,
                                 min_latin=args.min_latin, dev_ratio=args.dev_ratio,
                                 latin_ratio=args.latin_ratio)
            if text is None:
                continue
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
            kept += 1
        except (json.JSONDecodeError, TypeError):
            continue
    print(f"[FILTER] kept {kept:,} records", file=sys.stderr)


if __name__ == "__main__":
    main()

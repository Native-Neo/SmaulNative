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
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    return text.strip()


def _ratio(pattern: re.Pattern[str], text: str) -> float:
    letters = sum(ch.isalpha() for ch in text)
    return len(pattern.findall(text)) / max(letters, 1)


def filter_text(text: Any, dataset: str = "auto", min_chars: int = 20, max_chars: int = 1_000_000) -> Optional[str]:
    if not isinstance(text, str):
        return None
    text = normalize_text(text)
    if len(text) < min_chars or len(text) > max_chars:
        return None
    if "\ufffd" in text or _URL_ONLY.fullmatch(text):
        return None
    if len(set(text.replace(" ", ""))) < 8:
        return None
    if dataset == "hindi":
        if len(_DEVANAGARI.findall(text)) < 8 or _ratio(_DEVANAGARI, text) < 0.20:
            return None
    elif dataset == "english":
        if len(_LATIN.findall(text)) < 12 or _ratio(_LATIN, text) < 0.50:
            return None
    return text


def filter_record(record: dict[str, Any], dataset: str = "auto", min_chars: int = 20, max_chars: int = 1_000_000) -> Optional[str]:
    value = record.get("text") if isinstance(record, dict) else None
    return filter_text(value, dataset, min_chars, max_chars)


def main() -> None:
    p = argparse.ArgumentParser(description="Filter JSONL text from stdin without storing the dataset")
    p.add_argument("--dataset", choices=["auto", "hindi", "english", "openthoughts"], default="auto")
    p.add_argument("--min_chars", type=int, default=20)
    p.add_argument("--max_chars", type=int, default=1_000_000)
    args = p.parse_args()
    kept = 0
    for line in sys.stdin:
        try:
            record = json.loads(line)
            text = filter_record(record, args.dataset, args.min_chars, args.max_chars)
            if text is None:
                continue
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
            kept += 1
        except (json.JSONDecodeError, TypeError):
            continue
    print(f"[FILTER] kept {kept:,} records", file=sys.stderr)


if __name__ == "__main__":
    main()

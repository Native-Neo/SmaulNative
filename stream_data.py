#!/usr/bin/env python3
"""Stream HF Parquet records directly without downloading datasets to disk."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from filter_data import filter_text

DATASETS: Dict[str, Dict[str, str]] = {
    "hindi": {"repo_id": "HuggingFaceFW/fineweb-2", "path": "data/hin_Deva/train"},
    "english": {"repo_id": "HuggingFaceFW/fineweb", "path": "data/100BT"},
    "openthoughts": {"repo_id": "open-thoughts/OpenThoughts3-1.2M", "path": "data"},
}

api = HfApi()
fs = HfFileSystem()


def _conversation(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        text = turn.get("value")
        if not isinstance(text, str) or not text.strip():
            continue
        role = str(turn.get("from", "")).strip()
        parts.append(f"{role}: {text.strip()}" if role else text.strip())
    return "\n".join(parts)


def _files(repo_id: str, path: str) -> list[str]:
    items = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path, recursive=True)
    return sorted(item.path for item in items if getattr(item, "path", "").endswith(".parquet"))


def _text_column(pf: pq.ParquetFile) -> tuple[str | None, bool]:
    names = pf.schema_arrow.names
    lower = {name.lower(): name for name in names}
    for name in ("text", "content", "document", "body"):
        if name in lower:
            return lower[name], False
    if "conversations" in lower:
        return lower["conversations"], True
    return None, False


def stream_dataset(name: str, min_chars: int = 20, max_chars: int = 1_000_000,
                   start_file: str | None = None, start_record: int = 0,
                   with_position: bool = False) -> Iterator[Any]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    config = DATASETS[name]
    paths = _files(config["repo_id"], config["path"])
    if not paths:
        raise RuntimeError(f"No Parquet files found for {name}")
    active = start_file is None
    for rel_path in paths:
        if not active:
            if rel_path != start_file:
                continue
            active = True
        skip = start_record if rel_path == start_file else 0
        remote = f"datasets/{config['repo_id']}/{rel_path}"
        print(f"[STREAM] {rel_path}" + (f" from row {skip:,}" if skip else ""), file=sys.stderr)
        with fs.open(remote, "rb") as handle:
            pf = pq.ParquetFile(handle)
            column, conversation = _text_column(pf)
            columns = [column] if column else None
            record = 0
            for batch in pf.iter_batches(batch_size=4096, columns=columns):
                values = batch.column(0).to_pylist() if column else [row for row in batch.to_pylist()]
                for value in values:
                    if record < skip:
                        record += 1
                        continue
                    if column:
                        text = _conversation(value) if conversation else value
                    else:
                        if not isinstance(value, dict):
                            record += 1
                            continue
                        text = ""
                        for item in value.values():
                            if isinstance(item, str) and len(item) > len(text):
                                text = item
                    text = filter_text(text, name, min_chars, max_chars)
                    position = (rel_path, record + 1)
                    record += 1
                    if text is not None:
                        yield (text, position) if with_position else text


def main() -> None:
    p = argparse.ArgumentParser(description="Stream and filter HF datasets with no dataset files on disk")
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--min_chars", type=int, default=20)
    p.add_argument("--max_chars", type=int, default=1_000_000)
    p.add_argument("--max_records", type=int, default=0, help="0 means unlimited")
    args = p.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    count = 0
    for name in names:
        for text in stream_dataset(name, args.min_chars, args.max_chars):
            print(json.dumps({"text": text}, ensure_ascii=False), flush=True)
            count += 1
            if args.max_records and count >= args.max_records:
                print(f"[DONE] streamed {count:,} records", file=sys.stderr)
                return
    print(f"[DONE] streamed {count:,} records", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stream HF Parquet records directly without downloading datasets to disk."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from dataset import extract_text
from filter_data import filter_text

DATASETS: Dict[str, Dict[str, str]] = {
    "hindi": {"repo_id": "HuggingFaceFW/fineweb-2", "path": "data/hin_Deva/train"},
    "english": {"repo_id": "HuggingFaceFW/fineweb", "path": "data/100BT"},
    "openthoughts": {"repo_id": "open-thoughts/OpenThoughts3-1.2M", "path": "data"},
}

def _token() -> str | None:
    # Prefer live env, fall back to import-time value for test compat.
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or globals().get("HF_TOKEN")


def _api() -> HfApi:
    return HfApi(token=_token())


def _fs() -> HfFileSystem:
    return HfFileSystem(token=_token())


# Kept for backward compat (tests patch stream_data.fs / HF_TOKEN).
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
api = HfApi(token=HF_TOKEN)
fs = HfFileSystem(token=HF_TOKEN)


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


def _files(repo_id: str, path: str, retries: int = 3) -> list[str]:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            items = _api().list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path,
                                          recursive=True)
            return sorted(item.path for item in items if getattr(item, "path", "").endswith(".parquet"))
        except Exception as exc:
            last = exc
            print(f"[STREAM] list_repo_tree attempt {attempt}/{retries} failed: {type(exc).__name__}",
                  file=sys.stderr)
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"could not list {repo_id}/{path}: {last}")


def _text_column(pf: pq.ParquetFile) -> tuple[str | None, bool]:
    names = pf.schema_arrow.names
    lower = {name.lower(): name for name in names}
    for name in ("text", "content", "document", "body", "code"):
        if name in lower:
            return lower[name], False
    if "conversations" in lower:
        return lower["conversations"], True
    return None, False


def _stream_retries() -> int:
    try:
        r = int(os.environ.get("SMAUL_STREAM_RETRIES", "6"))
    except (TypeError, ValueError):
        raise ValueError("SMAUL_STREAM_RETRIES must be an integer")
    return max(1, min(r, 10))


def _read_row_group(remote: str, row_group: int, columns: list[str] | None, token: str | None) -> list[Any]:
    retries = _stream_retries()
    for attempt in range(retries):
        try:
            with HfFileSystem(token=token).open(remote, "rb") as handle:
                return pq.ParquetFile(handle).read_row_group(row_group, columns=columns).to_pylist()
        except FileNotFoundError:
            raise
        except Exception as exc:
            # Do not retry schema/type errors: only transient IO.
            if "parquet" in type(exc).__name__.lower() and "magic" in str(exc).lower():
                raise
            if attempt + 1 >= retries:
                raise
            delay = min(30.0, 2.0**attempt) + random.uniform(0, 1)
            print(
                f"[STREAM] row group {row_group} read failed: {type(exc).__name__}; retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    return []


def _stream_file(config: Dict[str, str], dataset_name: str, rel_path: str, min_chars: int,
                 max_chars: int, skip: int, with_position: bool, workers: int) -> Iterator[Any]:
    remote = f"datasets/{config['repo_id']}/{rel_path}"
    token = _token()
    # Metadata open with retry; repo may update between files.
    meta_attempts = 3
    pf_info = None
    for attempt in range(1, meta_attempts + 1):
        try:
            with fs.open(remote, "rb") as handle:
                pf = pq.ParquetFile(handle)
                column, conversation = _text_column(pf)
                schema_names = pf.schema_arrow.names
                lower = {name.lower(): name for name in schema_names}
                prompt_col = lower.get("prompt")
                completion_col = lower.get("completion")
                pair_columns = prompt_col is not None and completion_col is not None and prompt_col != completion_col
                columns = [prompt_col, completion_col] if pair_columns else ([column] if column else None)
                row_groups = pf.num_row_groups
                pf_info = (column, conversation, prompt_col, completion_col, pair_columns, columns, row_groups)
            break
        except Exception as exc:
            if attempt >= meta_attempts:
                raise RuntimeError(f"could not open {remote}: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
    assert pf_info is not None
    column, conversation, prompt_col, completion_col, pair_columns, columns, row_groups = pf_info

    record = 0
    # Bound buffering: at most 8 row-groups in flight (each to_pylist() can be
    # hundreds of MB). workers capped by caller.
    batch_size = max(1, min(workers * 2, 8))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for first in range(0, row_groups, batch_size):
            futures = [
                pool.submit(_read_row_group, remote, i, columns, token)
                for i in range(first, min(first + batch_size, row_groups))
            ]
            for group_index, future in enumerate(futures, first):
                try:
                    rows = future.result()
                except Exception as exc:
                    raise RuntimeError(f"failed to read row group {group_index} in {rel_path}") from exc
                for value in rows:
                    if record < skip:
                        record += 1
                        continue
                    if pair_columns and isinstance(value, dict):
                        prompt = value.get(prompt_col)
                        completion = value.get(completion_col)
                        if isinstance(prompt, str) and isinstance(completion, str):
                            text = prompt + "\n" + completion
                        elif isinstance(prompt, str):
                            text = prompt
                        elif isinstance(completion, str):
                            text = completion
                        else:
                            text = ""
                    elif column:
                        text = _conversation(value) if conversation else value
                    else:
                        # Same heuristic as dataset.extract_text (numeric filter).
                        text = extract_text(value, rel_path) if isinstance(value, dict) else ""
                    # record counts RAW rows; kept-row counting happens in main
                    # via max_records. Positions thus have gaps by design.
                    position = (dataset_name, rel_path, record + 1)
                    record += 1
                    if not isinstance(text, str):
                        continue
                    # Length pre-check before expensive normalization in filter.
                    if len(text) < min_chars or len(text) > max_chars * 4:
                        # filter_text re-checks precisely; this is a cheap guard.
                        pass
                    text = filter_text(text, dataset_name, min_chars, max_chars)
                    if text is not None:
                        yield (text, position) if with_position else text


def stream_dataset(name: str, min_chars: int = 20, max_chars: int = 1_000_000,
                   start_dataset: str | None = None, start_file: str | None = None,
                   start_record: int = 0, with_position: bool = False,
                   workers: int | None = None) -> Iterator[Any]:
    if min_chars < 0 or max_chars < min_chars:
        raise ValueError("require 0 <= min_chars <= max_chars")
    if start_record < 0:
        raise ValueError("start_record must be non-negative")
    names = list(DATASETS) if name == "all" else [name]
    if name != "all" and name not in DATASETS:
        raise ValueError(f"unknown dataset: {name}")
    if start_dataset is not None and start_dataset not in names:
        raise ValueError(f"start_dataset {start_dataset!r} is not part of selected dataset {name!r}")
    if start_file is not None and start_dataset is None:
        raise ValueError("start_file requires start_dataset")
    if start_record > 0 and (start_file is None or start_dataset is None):
        raise ValueError("start_record requires start_dataset and start_file")

    workers = workers if workers is not None else int(os.environ.get("SMAUL_STREAM_WORKERS", "0") or 0)
    if workers <= 0:
        workers = min(4, max(1, os.cpu_count() or 1))
    workers = max(1, min(workers, 16))

    active_dataset = start_dataset is None
    found_start_file = start_file is None
    for dataset_name in names:
        if not active_dataset:
            if dataset_name != start_dataset:
                continue
            active_dataset = True
        config = DATASETS[dataset_name]
        paths = _files(config["repo_id"], config["path"])
        if not paths:
            raise RuntimeError(f"No Parquet files found for {dataset_name}")
        if start_dataset == dataset_name and start_file is not None and start_file not in paths:
            raise FileNotFoundError(f"resume file not found: {start_dataset}/{start_file}")

        active_file = start_file is None or dataset_name != start_dataset
        for rel_path in paths:
            if not active_file:
                if rel_path != start_file:
                    continue
                active_file = True
                found_start_file = True
            skip = start_record if dataset_name == start_dataset and rel_path == start_file else 0
            print(
                f"[STREAM] {dataset_name}/{rel_path}" + (f" from row {skip:,}" if skip else ""),
                file=sys.stderr,
            )
            yield from _stream_file(config, dataset_name, rel_path, min_chars, max_chars, skip, with_position, workers)
        if dataset_name == start_dataset and start_file is not None and not found_start_file:
            raise FileNotFoundError(f"resume file not found: {start_dataset}/{start_file}")
        # Do not stream later datasets when resuming: a typo'd start_file used
        # to stream everything after it before failing at the end.
        if start_dataset is not None and dataset_name == start_dataset and start_file is not None:
            # Resume is within one file; later files of the SAME dataset still
            # stream (loop above), but later DATASETS do not.
            pass
    if not found_start_file:
        raise FileNotFoundError(f"resume file not found: {start_dataset}/{start_file}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stream and filter HF datasets with no dataset files on disk")
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--min_chars", type=int, default=20)
    p.add_argument("--max_chars", type=int, default=1_000_000)
    p.add_argument("--max_records", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    args = p.parse_args()
    if args.max_records < 0:
        p.error("--max_records must be non-negative")
    count = 0
    buf: list = []
    try:
        for text in stream_dataset(args.dataset, args.min_chars, args.max_chars, workers=args.workers):
            # Buffer 500 lines per syscall: per-record print() kills throughput
            # for millions of rows.
            buf.append(json.dumps({"text": text}, ensure_ascii=False))
            count += 1
            if len(buf) >= 500:
                try:
                    sys.stdout.write("\n".join(buf) + "\n")
                except BrokenPipeError:
                    return
                buf.clear()
            if args.max_records and count >= args.max_records:
                break
        if buf:
            try:
                sys.stdout.write("\n".join(buf) + "\n")
            except BrokenPipeError:
                return
    except BrokenPipeError:
        return
    print(f"[DONE] streamed {count:,} records", file=sys.stderr)


if __name__ == "__main__":
    main()

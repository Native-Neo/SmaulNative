#!/usr/bin/env python3
"""Streaming Parquet downloader with no tokenizer dependency."""
from __future__ import annotations
import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from dataset import extract_text

DATASET_CONFIGS = {
    "hindi": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "path": "data/hin_Deva/train",
        "desc": "FineWeb-2 Hindi (Devanagari)"
    },
    "english": {
        "repo_id": "HuggingFaceFW/fineweb",
        "path": "data/100BT",
        "desc": "FineWeb English (100BT sample)"
    },
    "openthoughts": {
        "repo_id": "open-thoughts/OpenThoughts3-1.2M",
        "path": "data",
        "desc": "OpenThoughts3-1.2M reasoning (math, code, science)"
    },
}
DEFAULT_OUTPUT_ROOT = Path("./datasets")
DEFAULT_SHARD_ROWS = 100_000
DEFAULT_COMPRESSION = "zstd"


def _get_api() -> HfApi:
    # Lazily construct so import has no network/token side effect and HF_TOKEN
    # picked up at call time (not frozen at import).
    return HfApi()


def get_repo_files(repo_id: str, path: str, retries: int = 3) -> list[dict]:
    print(f"[HF] Inspecting {repo_id}/{path}...")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            info = _get_api().list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path,
                                             recursive=True)
            files = []
            for item in info:
                item_path = getattr(item, "path", None)
                if item_path and item_path.endswith(".parquet"):
                    files.append({"path": item_path, "size": int(getattr(item, "size", 0) or 0)})
            files.sort(key=lambda x: x["path"])
            if not files:
                raise RuntimeError(f"No Parquet files found in {repo_id}/{path}")
            print(f"[HF] Found {len(files):,} remote parquet files.")
            return files
        except Exception as exc:  # transient HF/network errors
            last = exc
            print(f"[WARN] list_repo_tree attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"could not list {repo_id}/{path} after {retries} attempts: {last}")


def format_conversation(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    parts = []
    for turn in value:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from", "")).strip()
        text = turn.get("value")
        if isinstance(text, str) and text.strip():
            parts.append(f"{role}: {text.strip()}" if role else text.strip())
    return "\n".join(parts)


def stream_raw_parquet(parquet_path: Path, start_row: int = 0) -> Iterator[tuple[int, str]]:
    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    lower = [n.lower() for n in names]
    text_col = next((names[lower.index(c)] for c in ("text", "content", "document", "body", "code")
                     if c in lower), None)
    conversation_col = names[lower.index("conversations")] if "conversations" in lower else None
    prompt_col = names[lower.index("prompt")] if "prompt" in lower else None
    completion_col = names[lower.index("completion")] if "completion" in lower else None
    if text_col:
        columns: Optional[List[str]] = [text_col]
    elif conversation_col:
        columns = [conversation_col]
    elif prompt_col and completion_col and prompt_col != completion_col:
        columns = [prompt_col, completion_col]
    else:
        columns = None
    current = 0
    for batch in pf.iter_batches(batch_size=4096, columns=columns):
        if current + batch.num_rows <= start_row:
            current += batch.num_rows
            continue
        if prompt_col and completion_col and prompt_col != completion_col and columns == [prompt_col, completion_col]:
            prompts = batch.column(0).to_pylist()
            completions = batch.column(1).to_pylist()
            for i, (pr, co) in enumerate(zip(prompts, completions)):
                row = current + i
                if row < start_row:
                    continue
                text = extract_text({"prompt": pr, "completion": co}, str(parquet_path)).strip()
                if text:
                    yield row, text
        elif text_col or conversation_col:
            col = batch.column(0)
            for i in range(batch.num_rows):
                row = current + i
                if row < start_row:
                    continue
                value = col[i].as_py()
                text = format_conversation(value) if conversation_col else value
                if isinstance(text, str) and text.strip():
                    yield row, text.strip()
        else:
            for i, row_data in enumerate(batch.to_pylist()):
                row = current + i
                if row < start_row:
                    continue
                text = extract_text(row_data, str(parquet_path)).strip()
                if text:
                    yield row, text
        current += batch.num_rows


class ShardWriter:

    def __init__(self, output_dir: Path, shard_rows: int, compression: Optional[str], dataset: str, repo_id: str,
                 start_idx: int):
        if shard_rows <= 0:
            raise ValueError(f"shard_rows must be positive, got {shard_rows}")
        if compression not in (None, "zstd", "snappy", "gzip"):
            raise ValueError(f"unsupported compression {compression!r}")
        if start_idx < 0:
            raise ValueError(f"start_idx must be non-negative, got {start_idx}")
        if start_idx > 9999:
            raise ValueError("shard index >9999 would break shard_XXXX lexicographic order; prune/archive first")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_rows = shard_rows
        self.compression = compression
        self.dataset = dataset
        self.repo_id = repo_id
        self.index = start_idx
        self.rows: List[str] = []
        self.schema = pa.schema([("text", pa.string())])

    def add(self, text: str) -> bool:
        self.rows.append(text)
        return len(self.rows) >= self.shard_rows

    def close(self) -> Optional[Dict[str, Any]]:
        if not self.rows:
            return None
        path = self.output_dir / f"shard_{self.index:04d}.parquet"
        # Atomic: write tmp then replace so a crash never leaves a torn shard
        # that the next run would delete as orphan (wasting re-download).
        tmp = path.with_suffix(".parquet.tmp")
        table = pa.Table.from_arrays([pa.array(self.rows, type=pa.string())], schema=self.schema)
        pq.write_table(table, tmp, compression=self.compression, use_dictionary=True)
        os.replace(tmp, path)
        meta = {
            "shard_file": path.name,
            "shard_idx": self.index,
            "row_count": len(self.rows),
            "dataset": self.dataset,
            "source_repo": self.repo_id,
            "compression": self.compression or "none"
        }
        print(f"[SHARD COMPLETE] {path.name}: {len(self.rows):,} rows")
        self.index += 1
        self.rows.clear()
        return meta


def default_manifest() -> Dict[str, Any]:
    return {"total_rows": 0, "shards": [], "completed_raw_files": [], "last_raw_file": None, "last_row_index": 0}


def read_manifest(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        return default_manifest()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"[WARN] corrupt manifest {path}: {exc}; starting from empty manifest (existing shards kept).")
        return default_manifest()
    if not isinstance(manifest, dict):
        print(f"[WARN] corrupt manifest {path}: expected object; starting fresh (existing shards kept).")
        return default_manifest()
    base = default_manifest()
    base.update(manifest)
    shards = base.get("shards") or []
    # Drop malformed shard entries instead of crashing later with AttributeError.
    clean_shards = [s for s in shards if isinstance(s, dict) and isinstance(s.get("shard_file"), str)]
    if len(clean_shards) != len(shards):
        print(f"[WARN] dropped {len(shards) - len(clean_shards)} malformed shard entries from manifest.")
    base["shards"] = clean_shards
    completed = base.get("completed_raw_files") or []
    base["completed_raw_files"] = sorted({c for c in completed if isinstance(c, str)})
    return base


def save_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    # Unique tmp name so concurrent runs do not race on manifest.json.tmp.
    fd, tmp_name = tempfile.mkstemp(dir=str(output_dir), prefix="manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def reconcile_output(output_dir: Path, manifest: Dict[str, Any], prune: bool = False) -> Dict[str, Any]:
    """Validate manifest against on-disk shards without destroying data by default.

    Historically this deleted any ``shard_*.parquet`` not listed in the manifest,
    so a corrupt/truncated manifest caused mass deletion of valid shards.
    Now orphans are kept and reported unless ``prune=True`` is passed explicitly.
    """
    shards = manifest.get("shards") or []
    if not isinstance(shards, list):
        raise RuntimeError("Manifest 'shards' must be a list")
    committed = {str(s.get("shard_file")) for s in shards if isinstance(s, dict) and s.get("shard_file")}
    orphans = [p for p in output_dir.glob("shard_*.parquet") if p.name not in committed]
    if orphans:
        if prune:
            for path in orphans:
                try:
                    path.unlink()
                    print(f"[PRUNE] removed orphan shard {path.name}")
                except OSError as exc:
                    print(f"[WARN] could not remove orphan shard {path.name}: {exc}")
        else:
            names = ", ".join(sorted(p.name for p in orphans))
            print(f"[WARN] found {len(orphans)} orphan shard(s) not in manifest (kept): {names}. "
                  f"Pass prune=True/--prune to delete them.")
    rows = sum(int(s.get("row_count", 0)) for s in shards if isinstance(s, dict))
    total = int(manifest.get("total_rows", 0))
    if rows != total:
        print(f"[WARN] manifest total_rows={total} != sum(shard row_counts)={rows}; repairing total_rows.")
        manifest["total_rows"] = rows
    return manifest


def _download_with_retry(repo_id: str, rel_path: str, temp_dir: Path, retries: int = 4) -> Path:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=rel_path,
                                        local_dir=str(temp_dir)))
        except Exception as exc:
            last = exc
            print(f"[WARN] download {rel_path} attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            time.sleep(min(2 ** attempt, 15))
    raise RuntimeError(f"download failed for {rel_path} after {retries} attempts: {last}")


def process_dataset(name: str, config: dict, output_dir: Path, temp_dir: Path, max_rows: int, shard_rows: int,
                    compression: Optional[str], clean_temp: bool, prune: bool = False) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"unsafe dataset name {name!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = reconcile_output(output_dir, read_manifest(output_dir), prune=prune)
    total_rows = int(manifest["total_rows"])
    if max_rows and total_rows >= max_rows:
        print(f"[COMPLETE] {name}: row limit already reached.")
        return
    repo_files = get_repo_files(config["repo_id"], config["path"])
    completed = set(manifest["completed_raw_files"])
    # Orphan shards are kept by default (reconcile_output without prune), so
    # len(shards) can collide with an on-disk shard_XXXX and os.replace would
    # silently overwrite it. Start after the max on-disk index.
    disk_idx = -1
    for _p in output_dir.glob("shard_*.parquet"):
        try:
            disk_idx = max(disk_idx, int(_p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    start_idx = max(len(manifest["shards"]), disk_idx + 1)
    writer = ShardWriter(output_dir, shard_rows, compression, name, config["repo_id"], start_idx)
    last_file = manifest.get("last_raw_file")
    last_row = int(manifest.get("last_row_index", 0))
    try:
        for file_info in repo_files:
            if max_rows and total_rows >= max_rows:
                break
            rel_path = file_info["path"]
            if rel_path in completed:
                continue
            start_row = last_row if last_file == rel_path else 0
            print(f"[DOWNLOAD] {rel_path} ({file_info['size'] / 1024**2:.1f} MB)...")
            downloaded = _download_with_retry(config["repo_id"], rel_path, temp_dir)
            file_completed = True
            try:
                for row_idx, text in stream_raw_parquet(downloaded, start_row):
                    if max_rows and total_rows >= max_rows:
                        file_completed = False
                        break
                    full = writer.add(text)
                    total_rows += 1
                    manifest["total_rows"] = total_rows
                    manifest["last_raw_file"] = rel_path
                    manifest["last_row_index"] = row_idx + 1
                    if full:
                        meta = writer.close()
                        if meta:
                            manifest["shards"].append(meta)
                        save_manifest(output_dir, manifest)
            except Exception:
                # Flush partial progress so a transient failure does not discard
                # up to shard_rows rows; resume position already in manifest.
                if writer.rows:
                    print(f"[WARN] flushing {len(writer.rows)} buffered rows before retry/abort")
                    meta = writer.close()
                    if meta:
                        manifest["shards"].append(meta)
                    save_manifest(output_dir, manifest)
                raise
            if file_completed:
                completed.add(rel_path)
                manifest["completed_raw_files"] = sorted(completed)
                manifest["last_raw_file"] = None
                manifest["last_row_index"] = 0
                save_manifest(output_dir, manifest)
                if clean_temp and downloaded.exists():
                    downloaded.unlink()
        meta = writer.close()
        if meta:
            manifest["shards"].append(meta)
        manifest["total_rows"] = total_rows
        save_manifest(output_dir, manifest)
    except BaseException:
        # Flush (not discard) buffered rows on abort so a KeyboardInterrupt
        # does not throw away up to shard_rows rows of progress.
        if writer.rows:
            try:
                meta = writer.close()
                if meta:
                    manifest["shards"].append(meta)
                manifest["total_rows"] = total_rows
                save_manifest(output_dir, manifest)
            except Exception as exc:
                print(f"[WARN] could not flush partial shard: {exc}")
        raise
    print(f"[DONE] {name.upper()}: {total_rows:,} rows in {len(manifest['shards']):,} shards.")


def parse_args():
    p = argparse.ArgumentParser(description="Download and shard Parquet datasets without tokenization")
    p.add_argument("--max_rows", type=int, default=0, help="Maximum rows per dataset; 0 means unlimited")
    p.add_argument("--shard_rows", type=int, default=100_000)
    p.add_argument("--compression", choices=["zstd", "snappy", "gzip", "none"], default="zstd")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--languages", nargs="+", choices=["hindi", "english", "openthoughts", "all"], default=["all"])
    p.add_argument("--temp_dir", default="./datasets/.temp_raw")
    p.add_argument("--no_clean_temp", action="store_true")
    p.add_argument("--prune", action="store_true",
                   help="Delete orphan shard_*.parquet files not listed in manifest.json (default: keep them)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.shard_rows <= 0 or args.max_rows < 0:
        raise ValueError("shard_rows must be positive and max_rows cannot be negative")
    if "all" in args.languages:
        langs = list(DATASET_CONFIGS)
    else:
        langs = args.languages
    compression = None if args.compression == "none" else args.compression
    root = Path(args.output_dir)
    temp = Path(args.temp_dir)
    for name in langs:
        process_dataset(name, DATASET_CONFIGS[name], root / name, temp / name, args.max_rows, args.shard_rows,
                        compression, not args.no_clean_temp, prune=args.prune)


if __name__ == "__main__":
    main()

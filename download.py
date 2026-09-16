#!/usr/bin/env python3
"""Streaming Parquet downloader with no tokenizer dependency."""
from __future__ import annotations
import argparse
import json
import os
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
api = HfApi()


def get_repo_files(repo_id: str, path: str) -> list[dict]:
    print(f"[HF] Inspecting {repo_id}/{path}...")
    info = api.list_repo_tree(repo_id=repo_id, repo_type="dataset", path_in_repo=path, recursive=True)
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
    text_col = next((names[lower.index(c)] for c in ("text", "content", "document", "body") if c in lower), None)
    conversation_col = names[lower.index("conversations")] if "conversations" in lower else None
    columns = [text_col or conversation_col] if text_col or conversation_col else None
    current = 0
    for batch in pf.iter_batches(batch_size=4096, columns=columns):
        if current + batch.num_rows <= start_row:
            current += batch.num_rows
            continue
        if text_col or conversation_col:
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
        table = pa.Table.from_arrays([pa.array(self.rows, type=pa.string())], schema=self.schema)
        pq.write_table(table, path, compression=self.compression, use_dictionary=True)
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
    manifest = json.loads(path.read_text(encoding="utf-8"))
    base = default_manifest()
    base.update(manifest)
    base["shards"] = list(base.get("shards") or [])
    base["completed_raw_files"] = list(base.get("completed_raw_files") or [])
    return base


def save_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    tmp = output_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def reconcile_output(output_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    committed = {str(s.get("shard_file")) for s in manifest["shards"]}
    for path in output_dir.glob("shard_*.parquet"):
        if path.name not in committed:
            path.unlink()
    rows = sum(int(s.get("row_count", 0)) for s in manifest["shards"])
    if rows != int(manifest.get("total_rows", 0)):
        raise RuntimeError("Manifest row count does not match committed shards")
    return manifest


def process_dataset(name: str, config: dict, output_dir: Path, temp_dir: Path, max_rows: int, shard_rows: int,
                    compression: Optional[str], clean_temp: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = reconcile_output(output_dir, read_manifest(output_dir))
    total_rows = int(manifest["total_rows"])
    if max_rows and total_rows >= max_rows:
        print(f"[COMPLETE] {name}: row limit already reached.")
        return
    repo_files = get_repo_files(config["repo_id"], config["path"])
    completed = set(manifest["completed_raw_files"])
    writer = ShardWriter(output_dir, shard_rows, compression, name, config["repo_id"], len(manifest["shards"]))
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
            downloaded = Path(
                hf_hub_download(repo_id=config["repo_id"], repo_type="dataset", filename=rel_path,
                                local_dir=str(temp_dir)))
            file_completed = True
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
        writer.rows.clear()
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
    return p.parse_args()


def main():
    args = parse_args()
    if args.shard_rows <= 0 or args.max_rows < 0:
        raise ValueError("shard_rows must be positive and max_rows cannot be negative")
    langs = ["hindi", "english"] if "all" in args.languages else args.languages
    compression = None if args.compression == "none" else args.compression
    root = Path(args.output_dir)
    temp = Path(args.temp_dir)
    for name in langs:
        process_dataset(name, DATASET_CONFIGS[name], root / name, temp / name, args.max_rows, args.shard_rows,
                        compression, not args.no_clean_temp)


if __name__ == "__main__":
    main()

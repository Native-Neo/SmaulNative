#!/usr/bin/env python3
"""download.py -- Token-limited, streaming Parquet dataset downloader with 1B-token sharding.

Features:
- Hard token-count limit per dataset (default: 10 billion tokens).
- Shards output into compressed Parquet files (default: ~1 billion tokens per shard, zstd compressed).
- Exact token accounting using the repository's native TokenizerWrapper.
- Streaming architecture: batches records incrementally through pq.ParquetWriter without buffering in RAM.
- Fully resumable: checks existing shards and manifest.json to resume without duplication.
- Preserves Hindi (FineWeb-2) and English (FineWeb) sources and text extraction/cleaning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from dataset import load_tokenizer, TokenizerWrapper, extract_text

# Repositories and paths
DATASET_CONFIGS = {
    "hindi": {
        "repo_id": "HuggingFaceFW/fineweb-2",
        "path": "data/hin_Deva/train",
        "desc": "FineWeb-2 Hindi (Devanagari)",
    },
    "english": {
        "repo_id": "HuggingFaceFW/fineweb",
        "path": "data/100BT",
        "desc": "FineWeb English (100BT sample)",
    },
}

DEFAULT_MAX_TOKENS = 10_000_000_000      # 10 Billion tokens hard limit per dataset
DEFAULT_SHARD_TOKENS = 1_000_000_000     # 1 Billion tokens per Parquet shard
DEFAULT_OUTPUT_ROOT = Path("./datasets")
DEFAULT_TOKENIZER_PATH = Path("./SmaulNative/tokenizer.json")
DEFAULT_COMPRESSION = "zstd"

api = HfApi()


def get_repo_files(repo_id: str, path: str) -> list[dict]:
    print(f"
[HF] Inspecting {repo_id}/{path}...")
    info = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path,
        recursive=True,
    )
    files = []
    for item in info:
        if not hasattr(item, "path") or not item.path.endswith(".parquet"):
            continue
        size = getattr(item, "size", None)
        if size is not None:
            files.append({"path": item.path, "size": int(size)})

    files.sort(key=lambda x: x["path"])
    if not files:
        raise RuntimeError(f"No Parquet files found in {repo_id}/{path}")
    print(f"[HF] Found {len(files):,} remote parquet files.")
    return files


class ShardWriter:
    """Incrementally writes documents to compressed Parquet shards capped by token count."""

    def __init__(
        self,
        output_dir: Path,
        shard_tokens: int,
        compression: str,
        language: str,
        repo_id: str,
        start_shard_idx: int = 0,
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = shard_tokens
        self.compression = compression
        self.language = language
        self.repo_id = repo_id
        self.current_shard_idx = start_shard_idx

        self.writer: Optional[pq.ParquetWriter] = None
        self.current_shard_path: Optional[Path] = None
        self.current_shard_tokens = 0
        self.current_shard_rows = 0

        self._buf_texts: List[str] = []
        self._buf_tokens: int = 0
        self._batch_size = 5000  # Flush to Parquet every 5000 rows

        self.schema = pa.schema([("text", pa.string())])

    def _open_shard(self):
        shard_name = f"shard_{self.current_shard_idx:03d}.parquet"
        self.current_shard_path = self.output_dir / shard_name
        self.writer = pq.ParquetWriter(
            str(self.current_shard_path),
            self.schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.current_shard_tokens = 0
        self.current_shard_rows = 0
        print(f"
[SHARD] Opened new shard: {self.current_shard_path.name} (target: {self.shard_tokens:,} tokens)")

    def _flush_buffer(self):
        if not self._buf_texts:
            return
        if self.writer is None:
            self._open_shard()

        table = pa.Table.from_arrays([pa.array(self._buf_texts, type=pa.string())], schema=self.schema)
        self.writer.write_table(table)
        self.current_shard_tokens += self._buf_tokens
        self.current_shard_rows += len(self._buf_texts)
        self._buf_texts.clear()
        self._buf_tokens = 0

    def add_document(self, text: str, token_count: int) -> bool:
        """Add a document. Returns True if a shard was closed."""
        self._buf_texts.append(text)
        self._buf_tokens += token_count

        if len(self._buf_texts) >= self._batch_size:
            self._flush_buffer()

        shard_closed = False
        if (self.current_shard_tokens + self._buf_tokens) >= self.shard_tokens:
            self._flush_buffer()
            self.close_shard()
            shard_closed = True
        return shard_closed

    def close_shard(self) -> Optional[Dict[str, Any]]:
        self._flush_buffer()
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            meta = {
                "shard_file": self.current_shard_path.name,
                "shard_idx": self.current_shard_idx,
                "token_count": self.current_shard_tokens,
                "row_count": self.current_shard_rows,
                "language": self.language,
                "source_repo": self.repo_id,
                "compression": self.compression,
            }
            print(f"[SHARD COMPLETE] {self.current_shard_path.name}: {self.current_shard_tokens:,} tokens, {self.current_shard_rows:,} rows")
            self.current_shard_idx += 1
            return meta
        return None


def read_existing_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read existing manifest {manifest_path}: {e}")
    return {
        "total_tokens": 0,
        "shards": [],
        "completed_raw_files": [],
        "last_raw_file": None,
        "last_row_index": 0,
    }


def save_manifest(output_dir: Path, manifest: Dict[str, Any]):
    manifest_path = output_dir / "manifest.json"
    tmp_path = manifest_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def stream_raw_parquet(parquet_path: Path, start_row: int = 0) -> Iterator[tuple[int, str]]:
    """Yields (row_idx, cleaned_text) from a Parquet file without loading all into RAM."""
    pf = pq.ParquetFile(parquet_path)
    current_row = 0
    col_names = [c.lower() for c in pf.schema_arrow.names]
    text_col = None
    for candidate in ("text", "content", "document", "body"):
        if candidate in col_names:
            text_col = pf.schema_arrow.names[col_names.index(candidate)]
            break

    columns = [text_col] if text_col else None
    for batch in pf.iter_batches(batch_size=2048, columns=columns):
        num_rows = batch.num_rows
        if current_row + num_rows <= start_row:
            current_row += num_rows
            continue

        if text_col:
            col_data = batch.column(0)
            for idx in range(num_rows):
                global_idx = current_row + idx
                if global_idx < start_row:
                    continue
                val = col_data[idx].as_py()
                if isinstance(val, str) and val.strip():
                    yield global_idx, val.strip()
        else:
            for idx, row in enumerate(batch.to_pylist()):
                global_idx = current_row + idx
                if global_idx < start_row:
                    continue
                text = extract_text(row, str(parquet_path)).strip()
                if text:
                    yield global_idx, text
        current_row += num_rows


def process_dataset(
    language: str,
    config: dict,
    tokenizer: TokenizerWrapper,
    output_dir: Path,
    temp_dir: Path,
    max_tokens: int,
    shard_tokens: int,
    compression: str,
    clean_temp: bool = True,
):
    print("=" * 70)
    print(f"PROCESSING DATASET: {language.upper()} ({config['desc']})")
    print(f"Target: {max_tokens:,} tokens | Shard size: {shard_tokens:,} tokens | Compression: {compression}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_existing_manifest(output_dir)
    total_tokens = manifest.get("total_tokens", 0)

    if total_tokens >= max_tokens:
        print(f"[COMPLETE] {language}: Hard token limit already reached ({total_tokens:,} / {max_tokens:,} tokens).")
        return

    repo_files = get_repo_files(config["repo_id"], config["path"])
    completed_raw = set(manifest.get("completed_raw_files", []))
    start_shard_idx = len(manifest.get("shards", []))

    writer = ShardWriter(
        output_dir=output_dir,
        shard_tokens=shard_tokens,
        compression=compression,
        language=language,
        repo_id=config["repo_id"],
        start_shard_idx=start_shard_idx,
    )

    last_raw_file = manifest.get("last_raw_file")
    last_row_index = manifest.get("last_row_index", 0)

    print(f"[RESUME] Resuming with {total_tokens:,} tokens written across {len(manifest.get('shards', []))} existing shards.")

    for file_info in repo_files:
        if total_tokens >= max_tokens:
            print(f"
[LIMIT REACHED] Reached maximum token budget: {total_tokens:,} >= {max_tokens:,}")
            break

        rel_path = file_info["path"]
        if rel_path in completed_raw:
            continue

        start_row = 0
        if last_raw_file == rel_path:
            start_row = last_row_index

        print(f"
[DOWNLOAD] Downloading {rel_path} ({file_info['size'] / 1024**2:.1f} MB)...")
        downloaded_path = Path(
            hf_hub_download(
                repo_id=config["repo_id"],
                repo_type="dataset",
                filename=rel_path,
                local_dir=str(temp_dir),
            )
        )

        file_completed = True
        for row_idx, text in stream_raw_parquet(downloaded_path, start_row=start_row):
            tok_ids = tokenizer.encode(text)
            tok_count = len(tok_ids) + 1  # include EOS token

            if total_tokens + tok_count > max_tokens:
                print(f"
[BUDGET] Next document ({tok_count:,} tokens) reaches hard limit {max_tokens:,}.")
                file_completed = False
                break

            shard_closed = writer.add_document(text, tok_count)
            total_tokens += tok_count
            manifest["total_tokens"] = total_tokens
            manifest["last_raw_file"] = rel_path
            manifest["last_row_index"] = row_idx + 1

            if shard_closed:
                closed_meta = writer.close_shard()
                if closed_meta:
                    manifest["shards"].append(closed_meta)
                save_manifest(output_dir, manifest)

            if total_tokens >= max_tokens:
                file_completed = False
                break

        if file_completed:
            completed_raw.add(rel_path)
            manifest["completed_raw_files"] = list(completed_raw)
            manifest["last_raw_file"] = None
            manifest["last_row_index"] = 0
            save_manifest(output_dir, manifest)

            if clean_temp and downloaded_path.exists():
                try:
                    downloaded_path.unlink()
                except Exception:
                    pass

    final_meta = writer.close_shard()
    if final_meta:
        manifest["shards"].append(final_meta)
        manifest["total_tokens"] = total_tokens
        save_manifest(output_dir, manifest)

    print(f"
[DONE] {language.upper()} finished: {total_tokens:,} tokens written into {output_dir}.")


def parse_args():
    p = argparse.ArgumentParser(description="Token-limited FineWeb dataset downloader and Parquet sharder")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="Maximum tokens per language dataset (default: 10,000,000,000 = 10B)")
    p.add_argument("--shard_tokens", type=int, default=DEFAULT_SHARD_TOKENS,
                   help="Target tokens per compressed Parquet shard (default: 1,000,000,000 = 1B)")
    p.add_argument("--compression", type=str, default=DEFAULT_COMPRESSION, choices=["zstd", "snappy", "gzip", "none"],
                   help="Columnar Parquet compression codec (default: zstd)")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                   help="Output directory for processed shards (default: ./datasets)")
    p.add_argument("--tokenizer_path", type=str, default=str(DEFAULT_TOKENIZER_PATH),
                   help="Path to tokenizer.json (default: ./SmaulNative/tokenizer.json)")
    p.add_argument("--languages", nargs="+", choices=["hindi", "english", "all"], default=["all"],
                   help="Languages to process: hindi, english, or all")
    p.add_argument("--temp_dir", type=str, default="./datasets/.temp_raw",
                   help="Temporary directory for downloading raw HF files before sharding")
    p.add_argument("--no_clean_temp", action="store_true",
                   help="Do not delete raw downloaded files after sharding")
    return p.parse_args()


def main():
    args = parse_args()
    output_base = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    tok_path = Path(args.tokenizer_path)

    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. A valid tokenizer is required for exact token counting."
        )

    print(f"[INIT] Loading tokenizer from {tok_path}...")
    tokenizer = load_tokenizer(tok_path)
    print(f"[INIT] Tokenizer loaded (vocab size: {tokenizer.get_vocab_size():,})")

    langs = ["hindi", "english"] if "all" in args.languages else args.languages

    for lang in langs:
        config = DATASET_CONFIGS[lang]
        lang_out = output_base / lang
        lang_temp = temp_dir / lang
        process_dataset(
            language=lang,
            config=config,
            tokenizer=tokenizer,
            output_dir=lang_out,
            temp_dir=lang_temp,
            max_tokens=args.max_tokens,
            shard_tokens=args.shard_tokens,
            compression=None if args.compression == "none" else args.compression,
            clean_temp=not args.no_clean_temp,
        )

    if temp_dir.exists() and not os.listdir(temp_dir):
        try:
            temp_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()

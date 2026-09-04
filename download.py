#!/usr/bin/env python3
"""download.py -- Token-limited, streaming Parquet dataset downloader."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from dataset import TokenizerWrapper, extract_text, load_tokenizer

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

DEFAULT_MAX_TOKENS = 10_000_000_000
DEFAULT_SHARD_TOKENS = 1_000_000_000
DEFAULT_OUTPUT_ROOT = Path("./datasets")
DEFAULT_TOKENIZER_PATH = Path("./SmaulNative/tokenizer.json")
DEFAULT_COMPRESSION = "zstd"

api = HfApi()


def get_repo_files(repo_id: str, path: str) -> list[dict]:
    print(f"[HF] Inspecting {repo_id}/{path}...")
    info = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=path,
        recursive=True,
    )
    files = []
    for item in info:
        item_path = getattr(item, "path", None)
        if not item_path or not item_path.endswith(".parquet"):
            continue
        size = getattr(item, "size", None)
        if size is not None:
            files.append({"path": item_path, "size": int(size)})
    files.sort(key=lambda x: x["path"])
    if not files:
        raise RuntimeError(f"No Parquet files found in {repo_id}/{path}")
    print(f"[HF] Found {len(files):,} remote parquet files.")
    return files


class ShardWriter:
    """Incrementally writes documents to token-sized compressed Parquet shards."""

    def __init__(self, output_dir: Path, shard_tokens: int, compression: Optional[str],
                 language: str, repo_id: str, start_shard_idx: int = 0):
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
        self._buf_tokens = 0
        self._batch_size = 5000
        self.schema = pa.schema([("text", pa.string())])

    def _open_shard(self) -> None:
        self.current_shard_path = self.output_dir / f"shard_{self.current_shard_idx:03d}.parquet"
        self.writer = pq.ParquetWriter(
            str(self.current_shard_path),
            self.schema,
            compression=self.compression,
            use_dictionary=True,
        )
        self.current_shard_tokens = 0
        self.current_shard_rows = 0
        print(f"[SHARD] Opened {self.current_shard_path.name} (target: {self.shard_tokens:,} tokens)")

    def _flush_buffer(self) -> None:
        if not self._buf_texts:
            return
        if self.writer is None:
            self._open_shard()
        table = pa.Table.from_arrays(
            [pa.array(self._buf_texts, type=pa.string())],
            schema=self.schema,
        )
        self.writer.write_table(table)
        self.current_shard_tokens += self._buf_tokens
        self.current_shard_rows += len(self._buf_texts)
        self._buf_texts.clear()
        self._buf_tokens = 0

    def add_document(self, text: str, token_count: int) -> bool:
        """Add one document and report whether the current shard reached its target."""
        self._buf_texts.append(text)
        self._buf_tokens += token_count

        if len(self._buf_texts) >= self._batch_size:
            self._flush_buffer()

        if self.current_shard_tokens + self._buf_tokens >= self.shard_tokens:
            self._flush_buffer()
            return True
        return False

    def close_shard(self) -> Optional[Dict[str, Any]]:
        self._flush_buffer()
        if self.writer is None:
            return None
        self.writer.close()
        self.writer = None
        meta = {
            "shard_file": self.current_shard_path.name,
            "shard_idx": self.current_shard_idx,
            "token_count": self.current_shard_tokens,
            "row_count": self.current_shard_rows,
            "language": self.language,
            "source_repo": self.repo_id,
            "compression": self.compression or "none",
        }
        print(f"[SHARD COMPLETE] {self.current_shard_path.name}: {self.current_shard_tokens:,} tokens, {self.current_shard_rows:,} rows")
        self.current_shard_idx += 1
        self.current_shard_path = None
        return meta

    def abort_open_shard(self) -> None:
        """Close and remove a shard that was not committed to the manifest."""
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None
        if self.current_shard_path and self.current_shard_path.exists():
            try:
                self.current_shard_path.unlink()
            except OSError:
                pass
        self.current_shard_path = None
        self._buf_texts.clear()
        self._buf_tokens = 0


def default_manifest() -> Dict[str, Any]:
    return {
        "total_tokens": 0,
        "shards": [],
        "completed_raw_files": [],
        "last_raw_file": None,
        "last_row_index": 0,
    }


def read_existing_manifest(output_dir: Path) -> Dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return default_manifest()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot safely resume: invalid manifest {manifest_path}: {exc}") from exc
    base = default_manifest()
    base.update(manifest)
    base["shards"] = list(base.get("shards") or [])
    base["completed_raw_files"] = list(base.get("completed_raw_files") or [])
    return base


def save_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.json"
    tmp_path = output_dir / "manifest.json.tmp"
    tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)


def reconcile_output(output_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Remove uncommitted shard files left by a crash and validate committed metadata."""
    committed = {str(s.get("shard_file")) for s in manifest["shards"] if s.get("shard_file")}
    for path in output_dir.glob("shard_*.parquet"):
        if path.name not in committed:
            print(f"[RECOVER] Removing uncommitted shard: {path.name}")
            try:
                path.unlink()
            except OSError as exc:
                raise RuntimeError(f"Cannot remove uncommitted shard {path}: {exc}") from exc

    manifest_tokens = int(manifest.get("total_tokens", 0))
    shard_tokens = sum(int(s.get("token_count", 0)) for s in manifest["shards"])
    if manifest_tokens != shard_tokens:
        raise RuntimeError(
            f"Manifest token mismatch: total_tokens={manifest_tokens:,}, "
            f"sum(shards)={shard_tokens:,}. Refusing to resume automatically."
        )

    expected_indices = list(range(len(manifest["shards"])))
    actual_indices = [int(s.get("shard_idx", -1)) for s in manifest["shards"]]
    if actual_indices != expected_indices:
        raise RuntimeError("Manifest shard indices are inconsistent; refusing unsafe resume.")
    return manifest


def stream_raw_parquet(parquet_path: Path, start_row: int = 0) -> Iterator[tuple[int, str]]:
    pf = pq.ParquetFile(parquet_path)
    current_row = 0
    names = pf.schema_arrow.names
    lower_names = [name.lower() for name in names]
    text_col = None
    for candidate in ("text", "content", "document", "body"):
        if candidate in lower_names:
            text_col = names[lower_names.index(candidate)]
            break

    columns = [text_col] if text_col else None
    for batch in pf.iter_batches(batch_size=2048, columns=columns):
        if current_row + batch.num_rows <= start_row:
            current_row += batch.num_rows
            continue
        if text_col:
            col = batch.column(0)
            for idx in range(batch.num_rows):
                global_idx = current_row + idx
                if global_idx < start_row:
                    continue
                value = col[idx].as_py()
                if isinstance(value, str) and value.strip():
                    yield global_idx, value.strip()
        else:
            for idx, row in enumerate(batch.to_pylist()):
                global_idx = current_row + idx
                if global_idx < start_row:
                    continue
                text = extract_text(row, str(parquet_path)).strip()
                if text:
                    yield global_idx, text
        current_row += batch.num_rows


def process_dataset(language: str, config: dict, tokenizer: TokenizerWrapper,
                    output_dir: Path, temp_dir: Path, max_tokens: int,
                    shard_tokens: int, compression: Optional[str], clean_temp: bool = True) -> None:
    print("=" * 70)
    print(f"PROCESSING DATASET: {language.upper()} ({config['desc']})")
    print(f"Target: {max_tokens:,} tokens | Shard size: {shard_tokens:,} tokens | Compression: {compression or 'none'}")
    print("=" * 70)

    if max_tokens <= 0 or shard_tokens <= 0:
        raise ValueError("max_tokens and shard_tokens must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    manifest = reconcile_output(output_dir, read_existing_manifest(output_dir))
    total_tokens = int(manifest["total_tokens"])

    if total_tokens >= max_tokens:
        print(f"[COMPLETE] {language}: hard token limit already reached ({total_tokens:,} / {max_tokens:,}).")
        return

    repo_files = get_repo_files(config["repo_id"], config["path"])
    completed_raw = set(manifest["completed_raw_files"])
    writer = ShardWriter(
        output_dir,
        shard_tokens,
        compression,
        language,
        config["repo_id"],
        len(manifest["shards"]),
    )

    last_raw_file = manifest.get("last_raw_file")
    last_row_index = int(manifest.get("last_row_index", 0))
    print(f"[RESUME] {total_tokens:,} tokens across {len(manifest['shards'])} committed shards.")

    try:
        for file_info in repo_files:
            if total_tokens >= max_tokens:
                break
            rel_path = file_info["path"]
            if rel_path in completed_raw:
                continue

            start_row = last_row_index if last_raw_file == rel_path else 0
            print(f"[DOWNLOAD] {rel_path} ({file_info['size'] / 1024**2:.1f} MB)...")
            downloaded_path = Path(hf_hub_download(
                repo_id=config["repo_id"],
                repo_type="dataset",
                filename=rel_path,
                local_dir=str(temp_dir),
            ))

            file_completed = True
            for row_idx, text in stream_raw_parquet(downloaded_path, start_row=start_row):
                token_count = len(tokenizer.encode(text)) + 1
                if total_tokens + token_count > max_tokens:
                    print(f"[BUDGET] Next document ({token_count:,} tokens) would exceed {max_tokens:,}.")
                    file_completed = False
                    break

                shard_closed = writer.add_document(text, token_count)
                total_tokens += token_count
                manifest["total_tokens"] = total_tokens
                manifest["last_raw_file"] = rel_path
                manifest["last_row_index"] = row_idx + 1

                if shard_closed:
                    closed_meta = writer.close_shard()
                    if closed_meta:
                        manifest["shards"].append(closed_meta)
                    manifest["completed_raw_files"] = sorted(completed_raw)
                    save_manifest(output_dir, manifest)

                if total_tokens >= max_tokens:
                    file_completed = False
                    break

            if file_completed:
                completed_raw.add(rel_path)
                manifest["completed_raw_files"] = sorted(completed_raw)
                manifest["last_raw_file"] = None
                manifest["last_row_index"] = 0
                save_manifest(output_dir, manifest)
                if clean_temp and downloaded_path.exists():
                    try:
                        downloaded_path.unlink()
                    except OSError:
                        pass

        final_meta = writer.close_shard()
        if final_meta:
            manifest["shards"].append(final_meta)
            manifest["total_tokens"] = total_tokens
            save_manifest(output_dir, manifest)
        elif total_tokens != int(manifest["total_tokens"]):
            manifest["total_tokens"] = total_tokens
            save_manifest(output_dir, manifest)

    except BaseException:
        writer.abort_open_shard()
        raise

    print(f"[DONE] {language.upper()}: {total_tokens:,} tokens in {len(manifest['shards']):,} shards.")


def parse_args():
    p = argparse.ArgumentParser(description="Token-limited FineWeb downloader and Parquet sharder")
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--shard_tokens", type=int, default=DEFAULT_SHARD_TOKENS)
    p.add_argument("--compression", choices=["zstd", "snappy", "gzip", "none"], default=DEFAULT_COMPRESSION)
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--tokenizer_path", default=str(DEFAULT_TOKENIZER_PATH))
    p.add_argument("--languages", nargs="+", choices=["hindi", "english", "all"], default=["all"])
    p.add_argument("--temp_dir", default="./datasets/.temp_raw")
    p.add_argument("--no_clean_temp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_base = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    tok_path = Path(args.tokenizer_path)

    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run tokenizer.py first")

    print(f"[INIT] Loading tokenizer from {tok_path}...")
    tokenizer = load_tokenizer(tok_path)
    print(f"[INIT] Tokenizer loaded (vocab size: {tokenizer.get_vocab_size():,})")

    langs = ["hindi", "english"] if "all" in args.languages else args.languages
    compression = None if args.compression == "none" else args.compression
    for lang in langs:
        process_dataset(
            language=lang,
            config=DATASET_CONFIGS[lang],
            tokenizer=tokenizer,
            output_dir=output_base / lang,
            temp_dir=temp_dir / lang,
            max_tokens=args.max_tokens,
            shard_tokens=args.shard_tokens,
            compression=compression,
            clean_temp=not args.no_clean_temp,
        )

    if temp_dir.exists() and not any(temp_dir.iterdir()):
        try:
            temp_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()

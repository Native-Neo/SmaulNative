#!/usr/bin/env python3
"""
data.py

High-throughput multiprocessing dataset cleaning pipeline for LLM pre-training.

Supported input formats:
    .txt .md .json .jsonl .csv .parquet

Pipeline:
    discovery
      -> streaming document batches
      -> ProcessPoolExecutor worker cleaning/filtering
      -> exact SHA256 deduplication
      -> approximate MinHash deduplication
      -> language filtering
      -> PII redaction
      -> markup removal
      -> Parquet ZSTD shards
      -> telemetry + summary

Dependencies:
    pip install pyarrow tqdm langdetect datasketch

Optional:
    pip install fasttext

Examples:
    python filter_datasets_parallel.py

    python filter_datasets_parallel.py \
        --input ./datasets \
        --output ./cleaned_datasets \
        --languages en hi \
        --workers 4 \
        --chunk-size 5000 \
        --max-memory-gb 3

Notes:
    - fastText is used automatically if --fasttext-model is supplied.
    - Otherwise langdetect is used.
    - Exact SHA256 deduplication is global and performed by the main process.
    - Approximate MinHash LSH deduplication is global and performed by the
      main process to avoid race conditions between workers.
    - Workers only process bounded chunks, preventing entire datasets from
      being loaded into RAM.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import json
import math
import os
import re
import signal
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
    HAVE_DATASKETCH = True
except ImportError:
    HAVE_DATASKETCH = False

try:
    import fasttext
    HAVE_FASTTEXT = True
except ImportError:
    HAVE_FASTTEXT = False

try:
    from langdetect import detect, DetectorFactory
    HAVE_LANGDETECT = True
    DetectorFactory.seed = 42
except ImportError:
    HAVE_LANGDETECT = False


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".parquet",
}

TEXT_KEYS = (
    "text",
    "content",
    "document",
    "body",
    "article",
    "prompt",
    "completion",
    "instruction",
    "response",
    "message",
)

EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

IPV4_RE = re.compile(
    r"\b(?:"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"\b"
)

PHONE_RE = re.compile(
    r"(?<!\w)"
    r"(?:\+?\d{1,3}[\s.\-]?)?"
    r"(?:\(?\d{2,5}\)?[\s.\-]?)?"
    r"\d{3,5}[\s.\-]?\d{3,6}"
    r"(?!\w)"
)

HTML_RE = re.compile(
    r"<(?:(?:script|style)\b[^>]*>.*?</(?:script|style)>|/?[A-Za-z][^>]{0,500})>",
    re.IGNORECASE | re.DOTALL,
)

XML_DECL_RE = re.compile(
    r"<\?.*?\?>|<!\[CDATA\[.*?\]\]>|<!--.*?-->",
    re.DOTALL,
)

CONTROL_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
)

WHITESPACE_RE = re.compile(r"\s+")

WORD_RE = re.compile(
    r"\b[\w\u0900-\u097F][\w'\-]*\b",
    re.UNICODE,
)

ALNUM_RE = re.compile(
    r"[A-Za-z0-9\u0900-\u097F]"
)

TOKEN_ESTIMATE_RE = re.compile(
    r"\w+|[^\w\s]",
    re.UNICODE,
)


@dataclass(frozen=True)
class Config:
    languages: Tuple[str, ...]
    min_chars: int
    max_chars: int
    min_tokens: int
    max_tokens: int
    max_non_alnum_ratio: float
    max_symbol_word_ratio: float
    max_char_repeat: int
    max_ngram_repeat_ratio: float
    redact_pii: bool
    fasttext_model: Optional[str]
    minhash_num_perm: int
    minhash_shingle_size: int
    enable_minhash: bool


@dataclass
class WorkerMetrics:
    scanned: int = 0
    accepted: int = 0

    dropped_empty: int = 0
    dropped_length: int = 0
    dropped_token_length: int = 0
    dropped_non_alnum: int = 0
    dropped_repetition: int = 0
    dropped_symbol_ratio: int = 0
    dropped_language: int = 0
    dropped_decode: int = 0

    bytes_in: int = 0
    chars_out: int = 0
    estimated_tokens: int = 0

    def merge(self, other: "WorkerMetrics") -> None:
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))


_WORKER_CONFIG: Optional[Config] = None
_WORKER_FASTTEXT_MODEL = None


def init_worker(config_dict: Dict) -> None:
    """
    Initialize immutable worker-local state.

    Every ProcessPoolExecutor worker receives its own compiled Python module
    state and optionally loads one fastText language model.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    global _WORKER_CONFIG
    global _WORKER_FASTTEXT_MODEL

    _WORKER_CONFIG = Config(
        languages=tuple(config_dict["languages"]),
        min_chars=config_dict["min_chars"],
        max_chars=config_dict["max_chars"],
        min_tokens=config_dict["min_tokens"],
        max_tokens=config_dict["max_tokens"],
        max_non_alnum_ratio=config_dict["max_non_alnum_ratio"],
        max_symbol_word_ratio=config_dict["max_symbol_word_ratio"],
        max_char_repeat=config_dict["max_char_repeat"],
        max_ngram_repeat_ratio=config_dict["max_ngram_repeat_ratio"],
        redact_pii=config_dict["redact_pii"],
        fasttext_model=config_dict["fasttext_model"],
        minhash_num_perm=config_dict["minhash_num_perm"],
        minhash_shingle_size=config_dict["minhash_shingle_size"],
        enable_minhash=config_dict["enable_minhash"],
    )

    _WORKER_FASTTEXT_MODEL = None

    if _WORKER_CONFIG.fasttext_model:
        if not HAVE_FASTTEXT:
            raise RuntimeError(
                "A fastText model was specified, but the 'fasttext' package "
                "is not installed."
            )

        _WORKER_FASTTEXT_MODEL = fasttext.load_model(
            _WORKER_CONFIG.fasttext_model
        )


def discover_files(root: Path) -> Iterator[Path]:
    """
    Recursively discover supported dataset files.
    """
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def safe_text(value) -> Optional[str]:
    """
    Convert arbitrary input into text while avoiding accidental serialization
    of nested metadata structures as training documents.
    """
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    if isinstance(value, (int, float, bool)):
        return str(value)

    return None


def extract_text_from_json_object(obj) -> Optional[str]:
    """
    Extract the most likely textual field from a JSON object.
    """
    if isinstance(obj, str):
        return obj

    if isinstance(obj, list):
        parts = []

        for item in obj:
            text = extract_text_from_json_object(item)
            if text:
                parts.append(text)

        return "\n".join(parts) if parts else None

    if not isinstance(obj, dict):
        return safe_text(obj)

    for key in TEXT_KEYS:
        value = obj.get(key)

        text = safe_text(value)

        if text and text.strip():
            return text

    string_values = []

    for value in obj.values():
        text = safe_text(value)

        if text and text.strip():
            string_values.append(text)

    if len(string_values) == 1:
        return string_values[0]

    return None


def iter_txt(path: Path) -> Iterator[str]:
    """
    Treat non-empty lines as independent documents.
    """
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            if line.strip():
                yield line


def iter_jsonl(path: Path) -> Iterator[str]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = extract_text_from_json_object(obj)

            if text:
                yield text


def iter_json(path: Path) -> Iterator[str]:
    """
    Supports JSON arrays, JSON objects, and line-oriented fallback parsing.
    """
    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as handle:
            obj = json.load(handle)

        if isinstance(obj, list):
            for item in obj:
                text = extract_text_from_json_object(item)

                if text:
                    yield text

        else:
            text = extract_text_from_json_object(obj)

            if text:
                yield text

        return

    except (
        json.JSONDecodeError,
        UnicodeError,
        OSError,
    ):
        pass

    yield from iter_jsonl(path)


def iter_csv(path: Path) -> Iterator[str]:
    with path.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames:
            text_columns = [
                column
                for column in reader.fieldnames
                if column.lower() in TEXT_KEYS
            ]

            for row in reader:
                if text_columns:
                    parts = [
                        row[column]
                        for column in text_columns
                        if row.get(column)
                    ]
                else:
                    parts = [
                        value
                        for value in row.values()
                        if value and value.strip()
                    ]

                if parts:
                    yield "\n".join(parts)


def iter_parquet(path: Path) -> Iterator[str]:
    """
    Stream Parquet row groups/batches instead of reading the entire file.
    """
    parquet_file = pq.ParquetFile(path)

    schema_names = parquet_file.schema_arrow.names

    text_columns = [
        name
        for name in schema_names
        if name.lower() in TEXT_KEYS
    ]

    if not text_columns:
        text_columns = schema_names[:1]

    for batch in parquet_file.iter_batches(
        batch_size=4096,
        columns=text_columns,
        use_threads=False,
    ):
        columns = batch.to_pydict()

        if not columns:
            continue

        num_rows = batch.num_rows

        for index in range(num_rows):
            parts = []

            for column_name in text_columns:
                values = columns.get(column_name)

                if values is None:
                    continue

                text = safe_text(values[index])

                if text and text.strip():
                    parts.append(text)

            if parts:
                yield "\n".join(parts)


def iter_documents(path: Path) -> Iterator[str]:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        yield from iter_txt(path)

    elif suffix == ".jsonl":
        yield from iter_jsonl(path)

    elif suffix == ".json":
        yield from iter_json(path)

    elif suffix == ".csv":
        yield from iter_csv(path)

    elif suffix == ".parquet":
        yield from iter_parquet(path)


def chunked(
    iterable: Iterator[str],
    size: int,
) -> Iterator[List[str]]:
    """
    Build bounded IPC payloads.
    """
    batch: List[str] = []

    for item in iterable:
        batch.append(item)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def clean_text(
    text: str,
    config: Config,
) -> str:
    """
    Normalize Unicode, remove markup/control bytes, decode HTML entities,
    normalize whitespace, and optionally redact common PII.
    """
    text = unicodedata.normalize("NFKC", text)

    text = text.encode(
        "utf-8",
        errors="replace",
    ).decode(
        "utf-8",
        errors="replace",
    )

    text = XML_DECL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = html.unescape(text)
    text = CONTROL_RE.sub(" ", text)

    if config.redact_pii:
        text = EMAIL_RE.sub("[EMAIL]", text)
        text = IPV4_RE.sub("[IPV4]", text)
        text = PHONE_RE.sub("[PHONE]", text)

    text = WHITESPACE_RE.sub(" ", text)

    return text.strip()


def estimate_tokens(text: str) -> int:
    """
    Lightweight tokenizer-independent estimate.

    This intentionally avoids loading a large tokenizer inside every worker.
    """
    return len(TOKEN_ESTIMATE_RE.findall(text))


def non_alnum_ratio(text: str) -> float:
    if not text:
        return 1.0

    count = len(ALNUM_RE.findall(text))

    return 1.0 - (count / len(text))


def symbol_to_word_ratio(text: str) -> float:
    words = WORD_RE.findall(text)

    word_count = len(words)

    if word_count == 0:
        return float("inf")

    symbol_count = sum(
        1
        for char in text
        if not char.isspace()
        and not char.isalnum()
        and char not in "'-"
        and not ("\u0900" <= char <= "\u097F")
    )

    return symbol_count / word_count


def has_excessive_char_repetition(
    text: str,
    max_repeat: int,
) -> bool:
    """
    Detect long runs such as:
        aaaaaaaaaaaaaaaaa
        !!!!!!!!
        hahahahahahahaha
    """
    if max_repeat <= 1:
        return False

    previous = None
    run_length = 0

    for char in text:
        if char == previous:
            run_length += 1

            if run_length > max_repeat:
                return True
        else:
            previous = char
            run_length = 1

    return False


def has_ngram_repetition(
    text: str,
    threshold: float,
) -> bool:
    """
    Detect repeated token trigrams.

    The ratio is:
        repeated_ngram_occurrences / total_ngram_occurrences
    """
    tokens = WORD_RE.findall(text.lower())

    n = 3

    if len(tokens) < n * 2:
        return False

    ngrams = [
        tuple(tokens[index:index + n])
        for index in range(len(tokens) - n + 1)
    ]

    counts = Counter(ngrams)

    repeated = sum(
        count
        for count in counts.values()
        if count > 1
    )

    return (repeated / len(ngrams)) > threshold


def detect_language(text: str) -> Optional[str]:
    """
    Worker-local language detection with built-in script heuristic fallback.
    """
    global _WORKER_FASTTEXT_MODEL

    if _WORKER_FASTTEXT_MODEL is not None:
        sample = text[:10000].replace("\n", " ")

        labels, probabilities = _WORKER_FASTTEXT_MODEL.predict(
            sample,
            k=1,
        )

        if labels and probabilities:
            return labels[0].replace("__label__", "")

    if HAVE_LANGDETECT:
        try:
            return detect(text[:5000])
        except Exception:
            pass

    # Built-in zero-dependency fallback heuristic for Hindi & English
    hindi_count = sum(1 for char in text if "\u0900" <= char <= "\u097F")
    ascii_count = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    total_len = max(len(text), 1)

    if (hindi_count / total_len) > 0.05:
        return "hi"
    elif (ascii_count / total_len) > 0.2:
        return "en"

    return None


def normalize_language_code(language: str) -> str:
    """
    Normalize common fastText/langdetect output variations.
    """
    aliases = {
        "eng": "en",
        "hin": "hi",
    }

    return aliases.get(language.lower(), language.lower())


def make_minhash_signature(
    text: str,
    num_perm: int,
    shingle_size: int,
) -> Tuple[int, ...]:
    """
    Build a serializable MinHash signature.

    Workers return only hash values rather than full MinHash objects.
    """
    normalized = WHITESPACE_RE.sub(
        " ",
        text.lower(),
    ).strip()

    if len(normalized) <= shingle_size:
        shingles = {normalized}
    else:
        shingles = {
            normalized[index:index + shingle_size]
            for index in range(
                len(normalized) - shingle_size + 1
            )
        }

    if not shingles:
        shingles = {""}

    if HAVE_DATASKETCH:
        minhash = MinHash(
            num_perm=num_perm,
        )

        for shingle in shingles:
            minhash.update(
                shingle.encode(
                    "utf-8",
                    errors="replace",
                )
            )

        return tuple(
            int(value)
            for value in minhash.hashvalues
        )

    # Fallback signature if datasketch is unavailable.
    values = []

    for seed in range(num_perm):
        minimum = (1 << 64) - 1

        seed_bytes = seed.to_bytes(
            4,
            "little",
            signed=False,
        )

        for shingle in shingles:
            digest = hashlib.blake2b(
                seed_bytes + shingle.encode(
                    "utf-8",
                    errors="replace",
                ),
                digest_size=8,
            ).digest()

            value = int.from_bytes(
                digest,
                "little",
                signed=False,
            )

            if value < minimum:
                minimum = value

        values.append(minimum)

    return tuple(values)


def process_chunk(
    documents: Sequence[str],
) -> Tuple[List[Tuple[str, str, Optional[Tuple[int, ...]]]], WorkerMetrics]:
    """
    Process one bounded document chunk.

    Returns:
        [
            (
                cleaned_text,
                exact_sha256,
                optional_minhash_signature
            )
        ],
        worker_metrics
    """
    if _WORKER_CONFIG is None:
        raise RuntimeError(
            "Worker configuration was not initialized."
        )

    config = _WORKER_CONFIG
    metrics = WorkerMetrics()
    output = []

    for raw_text in documents:
        metrics.scanned += 1

        try:
            if not isinstance(raw_text, str):
                raw_text = safe_text(raw_text) or ""

            metrics.bytes_in += len(
                raw_text.encode(
                    "utf-8",
                    errors="replace",
                )
            )

            text = clean_text(
                raw_text,
                config,
            )

        except Exception:
            metrics.dropped_decode += 1
            continue

        if not text:
            metrics.dropped_empty += 1
            continue

        char_length = len(text)

        if (
            char_length < config.min_chars
            or char_length > config.max_chars
        ):
            metrics.dropped_length += 1
            continue

        token_count = estimate_tokens(text)

        if (
            token_count < config.min_tokens
            or token_count > config.max_tokens
        ):
            metrics.dropped_token_length += 1
            continue

        if (
            non_alnum_ratio(text)
            > config.max_non_alnum_ratio
        ):
            metrics.dropped_non_alnum += 1
            continue

        if (
            symbol_to_word_ratio(text)
            > config.max_symbol_word_ratio
        ):
            metrics.dropped_symbol_ratio += 1
            continue

        if has_excessive_char_repetition(
            text,
            config.max_char_repeat,
        ):
            metrics.dropped_repetition += 1
            continue

        if has_ngram_repetition(
            text,
            config.max_ngram_repeat_ratio,
        ):
            metrics.dropped_repetition += 1
            continue

        if config.languages:
            language = detect_language(text)

            if (
                language is None
                or normalize_language_code(language)
                not in config.languages
            ):
                metrics.dropped_language += 1
                continue

        exact_hash = hashlib.sha256(
            text.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

        signature = None

        if config.enable_minhash:
            signature = make_minhash_signature(
                text,
                config.minhash_num_perm,
                config.minhash_shingle_size,
            )

        output.append(
            (
                text,
                exact_hash,
                signature,
            )
        )

        metrics.accepted += 1
        metrics.chars_out += len(text)
        metrics.estimated_tokens += token_count

    return output, metrics


class GlobalDeduplicator:
    """
    Main-process global deduplication coordinator.

    Exact hashes are checked first.

    Approximate duplicate detection is then performed with MinHash LSH.
    Keeping this state in the main process avoids synchronization overhead
    and race conditions between workers.
    """

    def __init__(
        self,
        enable_minhash: bool,
        num_perm: int,
        threshold: float,
    ):
        self.exact_hashes = set()

        self.enable_minhash = (
            enable_minhash
            and HAVE_DATASKETCH
        )

        self.num_perm = num_perm
        self.counter = 0

        self.lsh = None

        if self.enable_minhash:
            self.lsh = MinHashLSH(
                threshold=threshold,
                num_perm=num_perm,
            )

        self.dropped_exact = 0
        self.dropped_minhash = 0

    def is_duplicate(
        self,
        exact_hash: str,
        signature: Optional[Tuple[int, ...]],
    ) -> bool:
        if exact_hash in self.exact_hashes:
            self.dropped_exact += 1
            return True

        if (
            self.enable_minhash
            and signature is not None
        ):
            minhash = MinHash(
                num_perm=self.num_perm,
            )

            minhash.hashvalues[:] = list(signature)

            matches = self.lsh.query(minhash)

            if matches:
                self.dropped_minhash += 1
                return True

            key = str(self.counter)
            self.counter += 1

            self.lsh.insert(
                key,
                minhash,
            )

        self.exact_hashes.add(exact_hash)

        return False


class ShardedParquetWriter:
    """
    Buffered single-process Parquet shard writer.

    Only the main process writes output files, preventing concurrent file
    corruption and keeping output ordering deterministic at the shard level.
    """

    def __init__(
        self,
        output_dir: Path,
        shard_size: int,
        compression_level: int,
    ):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.compression_level = compression_level

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.buffer: List[str] = []
        self.shard_index = 0

        self.raw_bytes = 0
        self.output_bytes = 0
        self.documents_written = 0

    def add(
        self,
        text: str,
    ) -> None:
        self.buffer.append(text)
        self.raw_bytes += len(
            text.encode(
                "utf-8",
                errors="replace",
            )
        )

        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        shard_path = self.output_dir / (
            f"cleaned-{self.shard_index:06d}.parquet"
        )

        table = pa.Table.from_arrays(
            [
                pa.array(
                    self.buffer,
                    type=pa.string(),
                )
            ],
            names=["text"],
        )

        pq.write_table(
            table,
            shard_path,
            compression="zstd",
            compression_level=self.compression_level,
            use_dictionary=True,
            write_statistics=True,
            row_group_size=min(
                len(self.buffer),
                65536,
            ),
        )

        self.output_bytes += shard_path.stat().st_size
        self.documents_written += len(self.buffer)

        self.buffer.clear()
        self.shard_index += 1

    def close(self) -> None:
        self.flush()


def build_config(
    args: argparse.Namespace,
) -> Config:
    languages = tuple(
        sorted(
            {
                normalize_language_code(language)
                for language in args.languages
            }
        )
    )

    return Config(
        languages=languages,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        max_non_alnum_ratio=args.max_non_alnum_ratio,
        max_symbol_word_ratio=args.max_symbol_word_ratio,
        max_char_repeat=args.max_char_repeat,
        max_ngram_repeat_ratio=args.max_ngram_repeat_ratio,
        redact_pii=not args.no_redact_pii,
        fasttext_model=args.fasttext_model,
        minhash_num_perm=args.minhash_num_perm,
        minhash_shingle_size=args.minhash_shingle_size,
        enable_minhash=not args.disable_minhash,
    )


def calculate_worker_count(
    requested_workers: Optional[int],
    max_memory_gb: float,
    chunk_size: int,
) -> int:
    """
    Conservative worker calculation.

    Text-heavy chunks are variable, so memory cannot be perfectly predicted.
    This caps workers based on a conservative 128 MB estimated active working
    budget per worker while never exceeding CPU count.
    """
    cpu_count = os.cpu_count() or 1

    if requested_workers is not None:
        workers = min(
            requested_workers,
            cpu_count,
        )
    else:
        workers = cpu_count

    memory_budget_bytes = int(
        max_memory_gb * 1024 ** 3
    )

    estimated_worker_budget = max(
        128 * 1024 ** 2,
        min(
            256 * 1024 ** 2,
            chunk_size * 32 * 1024,
        ),
    )

    memory_limited_workers = max(
        1,
        memory_budget_bytes // estimated_worker_budget,
    )

    return max(
        1,
        min(
            workers,
            memory_limited_workers,
        ),
    )


def main(args: argparse.Namespace) -> int:
    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        print(
            f"Input directory does not exist: {input_dir}",
            file=sys.stderr,
        )
        return 1

    if input_dir == output_dir:
        print(
            "Input and output directories must be different.",
            file=sys.stderr,
        )
        return 1

    if (
        args.fasttext_model
        and not Path(args.fasttext_model).is_file()
    ):
        print(
            f"fastText model not found: {args.fasttext_model}",
            file=sys.stderr,
        )
        return 1

    if (
        args.fasttext_model
        and not HAVE_FASTTEXT
    ):
        print(
            "Install fastText first: pip install fasttext",
            file=sys.stderr,
        )
        return 1

    if (
        not args.fasttext_model
        and not HAVE_LANGDETECT
        and args.languages
    ):
        print(
            "Language filtering requires either langdetect or fasttext.\n"
            "Install langdetect:\n"
            "    pip install langdetect",
            file=sys.stderr,
        )
        return 1

    config = build_config(args)

    if config.enable_minhash and not HAVE_DATASKETCH:
        print(
            "Warning: datasketch is not installed. "
            "MinHash deduplication will use the fallback signature but "
            "global LSH deduplication will be unavailable.",
            file=sys.stderr,
        )

    files = list(
        discover_files(input_dir)
    )

    if not files:
        print(
            f"No supported dataset files found in {input_dir}",
            file=sys.stderr,
        )
        return 1

    total_input_bytes = sum(
        path.stat().st_size
        for path in files
    )

    workers = calculate_worker_count(
        args.workers,
        args.max_memory_gb,
        args.chunk_size,
    )

    print("=" * 80)
    print("Parallel Dataset Filtering Pipeline")
    print("=" * 80)
    print(f"Input:              {input_dir}")
    print(f"Output:             {output_dir}")
    print(f"Files discovered:   {len(files):,}")
    print(
        f"Input size:         "
        f"{total_input_bytes / (1024 ** 3):.2f} GB"
    )
    print(f"Workers:            {workers}")
    print(f"Chunk size:         {args.chunk_size:,} documents")
    print(f"Output shard size:  {args.shard_size:,} documents")
    print(f"Languages:          {', '.join(config.languages) or 'disabled'}")
    print(
        f"MinHash:            "
        f"{'enabled' if config.enable_minhash and HAVE_DATASKETCH else 'disabled'}"
    )
    print("=" * 80)

    metrics = WorkerMetrics()

    deduplicator = GlobalDeduplicator(
        enable_minhash=config.enable_minhash,
        num_perm=config.minhash_num_perm,
        threshold=args.minhash_threshold,
    )

    writer = ShardedParquetWriter(
        output_dir=output_dir,
        shard_size=args.shard_size,
        compression_level=args.zstd_level,
    )

    start_time = time.perf_counter()

    processed_bytes = 0
    completed_chunks = 0
    stop_requested = False

    def handle_sigint(
        signum,
        frame,
    ):
        nonlocal stop_requested

        stop_requested = True

    previous_handler = signal.signal(
        signal.SIGINT,
        handle_sigint,
    )

    config_dict = asdict(config)

    progress = tqdm(
        total=total_input_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Processing",
        dynamic_ncols=True,
    )

    pending = set()

    max_pending = max(
        workers * 2,
        2,
    )

    def submit_chunk(
        executor,
        chunk: List[str],
    ):
        future = executor.submit(
            process_chunk,
            chunk,
        )

        pending.add(
            (
                future,
                sum(
                    len(
                        item.encode(
                            "utf-8",
                            errors="replace",
                        )
                    )
                    for item in chunk
                ),
            )
        )

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(config_dict,),
        ) as executor:

            for path in files:
                if stop_requested:
                    break

                try:
                    document_iterator = iter_documents(path)

                    for chunk in chunked(
                        document_iterator,
                        args.chunk_size,
                    ):
                        if stop_requested:
                            break

                        submit_chunk(
                            executor,
                            chunk,
                        )

                        while len(pending) >= max_pending:
                            done, _ = concurrent.futures.wait(
                                [
                                    item[0]
                                    for item in pending
                                ],
                                return_when=(
                                    concurrent.futures.FIRST_COMPLETED
                                ),
                            )

                            new_pending = set()

                            for future, chunk_bytes in pending:
                                if future not in done:
                                    new_pending.add(
                                        (
                                            future,
                                            chunk_bytes,
                                        )
                                    )
                                    continue

                                result, worker_metrics = future.result()

                                metrics.merge(
                                    worker_metrics
                                )

                                processed_bytes += chunk_bytes
                                completed_chunks += 1

                                for (
                                    text,
                                    exact_hash,
                                    signature,
                                ) in result:
                                    if deduplicator.is_duplicate(
                                        exact_hash,
                                        signature,
                                    ):
                                        continue

                                    writer.add(text)

                                elapsed = max(
                                    time.perf_counter() - start_time,
                                    1e-9,
                                )

                                mb_per_second = (
                                    processed_bytes
                                    / elapsed
                                    / (1024 ** 2)
                                )

                                items_per_second = (
                                    metrics.scanned
                                    / elapsed
                                )

                                progress.update(
                                    chunk_bytes
                                )

                                progress.set_postfix(
                                    {
                                        "MB/s": f"{mb_per_second:.1f}",
                                        "items/s": f"{items_per_second:.0f}",
                                        "kept": (
                                            f"{writer.documents_written + len(writer.buffer):,}"
                                        ),
                                    }
                                )

                            pending = new_pending

                except Exception as exc:
                    print(
                        f"\nError reading {path}: {exc}",
                        file=sys.stderr,
                    )

            while pending:
                done, _ = concurrent.futures.wait(
                    [
                        item[0]
                        for item in pending
                    ],
                    return_when=(
                        concurrent.futures.FIRST_COMPLETED
                    ),
                )

                new_pending = set()

                for future, chunk_bytes in pending:
                    if future not in done:
                        new_pending.add(
                            (
                                future,
                                chunk_bytes,
                            )
                        )
                        continue

                    result, worker_metrics = future.result()

                    metrics.merge(worker_metrics)

                    processed_bytes += chunk_bytes
                    completed_chunks += 1

                    for (
                        text,
                        exact_hash,
                        signature,
                    ) in result:
                        if deduplicator.is_duplicate(
                            exact_hash,
                            signature,
                        ):
                            continue

                        writer.add(text)

                    elapsed = max(
                        time.perf_counter() - start_time,
                        1e-9,
                    )

                    mb_per_second = (
                        processed_bytes
                        / elapsed
                        / (1024 ** 2)
                    )

                    items_per_second = (
                        metrics.scanned
                        / elapsed
                    )

                    progress.update(
                        chunk_bytes
                    )

                    progress.set_postfix(
                        {
                            "MB/s": f"{mb_per_second:.1f}",
                            "items/s": f"{items_per_second:.0f}",
                            "kept": (
                                f"{writer.documents_written + len(writer.buffer):,}"
                            ),
                        }
                    )

                pending = new_pending

    finally:
        signal.signal(
            signal.SIGINT,
            previous_handler,
        )

        progress.close()

        writer.close()

    elapsed = max(
        time.perf_counter() - start_time,
        1e-9,
    )

    kept_documents = writer.documents_written

    total_dropped = (
        metrics.dropped_empty
        + metrics.dropped_length
        + metrics.dropped_token_length
        + metrics.dropped_non_alnum
        + metrics.dropped_repetition
        + metrics.dropped_symbol_ratio
        + metrics.dropped_language
        + metrics.dropped_decode
        + deduplicator.dropped_exact
        + deduplicator.dropped_minhash
    )

    compression_ratio = (
        writer.raw_bytes / writer.output_bytes
        if writer.output_bytes > 0
        else 0.0
    )

    report = {
        "files_discovered": len(files),
        "chunks_completed": completed_chunks,
        "documents_scanned": metrics.scanned,
        "documents_worker_accepted": metrics.accepted,
        "documents_written": kept_documents,
        "documents_dropped_total": total_dropped,
        "dropped": {
            "empty": metrics.dropped_empty,
            "character_length": metrics.dropped_length,
            "token_length": metrics.dropped_token_length,
            "non_alphanumeric_ratio": metrics.dropped_non_alnum,
            "repetition": metrics.dropped_repetition,
            "symbol_to_word_ratio": metrics.dropped_symbol_ratio,
            "language": metrics.dropped_language,
            "decode": metrics.dropped_decode,
            "exact_duplicate": deduplicator.dropped_exact,
            "minhash_duplicate": deduplicator.dropped_minhash,
        },
        "estimated_cleaned_tokens": metrics.estimated_tokens,
        "input_bytes_discovered": total_input_bytes,
        "bytes_scanned": processed_bytes,
        "clean_text_bytes": writer.raw_bytes,
        "compressed_output_bytes": writer.output_bytes,
        "compression_ratio": compression_ratio,
        "elapsed_seconds": elapsed,
        "documents_per_second": metrics.scanned / elapsed,
        "megabytes_per_second": (
            processed_bytes / elapsed / (1024 ** 2)
        ),
        "workers": workers,
        "interrupted": stop_requested,
    }

    report_path = output_dir / "summary.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(
        f"Files discovered:              {len(files):,}"
    )
    print(
        f"Chunks completed:              {completed_chunks:,}"
    )
    print(
        f"Documents scanned:             {metrics.scanned:,}"
    )
    print(
        f"Worker-level accepted:         {metrics.accepted:,}"
    )
    print(
        f"Documents written:             {kept_documents:,}"
    )
    print()

    print("Dropped by rule:")
    print(
        f"  Empty:                       {metrics.dropped_empty:,}"
    )
    print(
        f"  Character length:            {metrics.dropped_length:,}"
    )
    print(
        f"  Token length:                {metrics.dropped_token_length:,}"
    )
    print(
        f"  Non-alphanumeric ratio:      {metrics.dropped_non_alnum:,}"
    )
    print(
        f"  Repetition:                  {metrics.dropped_repetition:,}"
    )
    print(
        f"  Symbol/word ratio:           {metrics.dropped_symbol_ratio:,}"
    )
    print(
        f"  Language:                    {metrics.dropped_language:,}"
    )
    print(
        f"  Decode/processing:           {metrics.dropped_decode:,}"
    )
    print(
        f"  Exact SHA256 duplicates:     {deduplicator.dropped_exact:,}"
    )
    print(
        f"  MinHash duplicates:          {deduplicator.dropped_minhash:,}"
    )
    print()

    print(
        f"Estimated cleaned tokens:      {metrics.estimated_tokens:,}"
    )
    print(
        f"Clean text size:               "
        f"{writer.raw_bytes / (1024 ** 3):.3f} GB"
    )
    print(
        f"Compressed Parquet size:       "
        f"{writer.output_bytes / (1024 ** 3):.3f} GB"
    )
    print(
        f"Compression ratio:             {compression_ratio:.2f}:1"
    )
    print(
        f"Elapsed:                       {elapsed:.2f} seconds"
    )
    print(
        f"Throughput:                    "
        f"{metrics.scanned / elapsed:.2f} docs/s"
    )
    print(
        f"Read throughput:               "
        f"{processed_bytes / elapsed / (1024 ** 2):.2f} MB/s"
    )
    print(
        f"Output directory:              {output_dir}"
    )
    print(
        f"Summary report:                {report_path}"
    )

    if stop_requested:
        print(
            "\nWARNING: Processing was interrupted. "
            "All completed output shards were flushed safely."
        )

    return 130 if stop_requested else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel streaming dataset cleaner and deduplicator "
            "for LLM pre-training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        default="./datasets",
        help="Input dataset directory.",
    )

    parser.add_argument(
        "--output",
        default="./cleaned_datasets",
        help="Output directory for ZSTD-compressed Parquet shards.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Maximum worker processes. "
            "Default dynamically scales up to CPU count and memory budget."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Documents per multiprocessing task.",
    )

    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Documents per output Parquet shard.",
    )

    parser.add_argument(
        "--max-memory-gb",
        type=float,
        default=3.0,
        help=(
            "Approximate aggregate memory budget used to conservatively "
            "limit worker count."
        ),
    )

    parser.add_argument(
        "--languages",
        nargs="*",
        default=["en", "hi"],
        help=(
            "Allowed ISO language codes. "
            "Use no values to disable language filtering."
        ),
    )

    parser.add_argument(
        "--fasttext-model",
        default=None,
        help=(
            "Optional fastText language ID model path. "
            "If omitted, langdetect is used."
        ),
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--max-non-alnum-ratio",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--max-symbol-word-ratio",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-char-repeat",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--max-ngram-repeat-ratio",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--no-redact-pii",
        action="store_true",
        help="Disable email/IP/phone redaction.",
    )

    parser.add_argument(
        "--disable-minhash",
        action="store_true",
        help="Disable approximate MinHash LSH deduplication.",
    )

    parser.add_argument(
        "--minhash-num-perm",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--minhash-shingle-size",
        type=int,
        default=5,
        help="Character shingle size used for MinHash.",
    )

    parser.add_argument(
        "--minhash-threshold",
        type=float,
        default=0.85,
        help="Approximate Jaccard similarity threshold.",
    )

    parser.add_argument(
        "--zstd-level",
        type=int,
        default=15,
        help="ZSTD compression level.",
    )

    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be > 0")

    if args.shard_size <= 0:
        parser.error("--shard-size must be > 0")

    if args.max_memory_gb <= 0:
        parser.error("--max-memory-gb must be > 0")

    if args.workers is not None and args.workers <= 0:
        parser.error("--workers must be > 0")

    if args.min_chars < 0:
        parser.error("--min-chars must be >= 0")

    if args.max_chars < args.min_chars:
        parser.error(
            "--max-chars must be >= --min-chars"
        )

    if args.min_tokens < 0:
        parser.error("--min-tokens must be >= 0")

    if args.max_tokens < args.min_tokens:
        parser.error(
            "--max-tokens must be >= --min-tokens"
        )

    if not 0.0 <= args.max_non_alnum_ratio <= 1.0:
        parser.error(
            "--max-non-alnum-ratio must be between 0 and 1"
        )

    if args.max_symbol_word_ratio < 0:
        parser.error(
            "--max-symbol-word-ratio must be >= 0"
        )

    if not 0.0 <= args.max_ngram_repeat_ratio <= 1.0:
        parser.error(
            "--max-ngram-repeat-ratio must be between 0 and 1"
        )

    if not 0.0 < args.minhash_threshold <= 1.0:
        parser.error(
            "--minhash-threshold must be > 0 and <= 1"
        )

    if args.minhash_num_perm <= 0:
        parser.error(
            "--minhash-num-perm must be > 0"
        )

    if args.minhash_shingle_size <= 0:
        parser.error(
            "--minhash-shingle-size must be > 0"
        )

    return args


if __name__ == "__main__":
    try:
        exit_code = main(
            parse_args()
        )
    except BrokenPipeError:
        exit_code = 1
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as exc:
        print(
            f"Fatal error: {exc}",
            file=sys.stderr,
        )
        exit_code = 1

    raise SystemExit(exit_code)

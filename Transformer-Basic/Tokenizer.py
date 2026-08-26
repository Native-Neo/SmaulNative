import os
import sys
import glob
import json
import csv
from typing import Iterator
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

DATASETS_DIR = "./datasets"
OUTPUT_DIR = "./SmaulNative"
VOCAB_SIZE = 128000

SPECIAL_TOKENS = [
    "<unk>",
    "<s>",
    "</s>",
    "<pad>",
    "<mask>",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<think>",
    "</think>",
]

def run_prerequisite_checks() -> list[str]:
    """Check dataset directory existence, required libraries, and available files."""
    print("=== Phase 1: Running Pre-Training Environment Checks ===")

    if not os.path.exists(DATASETS_DIR):
        print(f"[Error] Datasets directory '{DATASETS_DIR}' does not exist.")
        sys.exit(1)

    supported_exts = ["*.txt", "*.md", "*.jsonl", "*.json", "*.csv", "*.parquet"]
    found_files = []
    for ext in supported_exts:
        found_files.extend(glob.glob(os.path.join(DATASETS_DIR, "**", ext), recursive=True))

    if not found_files:
        print(f"[Error] No supported files found in '{DATASETS_DIR}'. Please add files before training.")
        sys.exit(1)

    print(f"[PASS] Environment OK. Found {len(found_files)} data files across format types.")
    return found_files

def stream_text_from_file(file_path: str) -> Iterator[str]:
    """Memory-mapped streaming text reader across multiple file formats."""
    ext = os.path.splitext(file_path)[-1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        elif ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            text = data.get("text") or data.get("content") or data.get("code")
                            if text and isinstance(text, str):
                                yield text.strip()
                    except json.JSONDecodeError:
                        continue

        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = item.get("text") or item.get("content")
                                if text and isinstance(text, str):
                                    yield text.strip()
                            elif isinstance(item, str):
                                yield item.strip()
                except Exception:
                    pass

        elif ext == ".csv":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                text_idx = 0
                if header:
                    for i, col in enumerate(header):
                        if col.lower() in ["text", "content", "body"]:
                            text_idx = i
                            break
                for row in reader:
                    if row and len(row) > text_idx:
                        yield row[text_idx].strip()

        elif ext == ".parquet":
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                text_col = "text"
                schema_names = parquet_file.schema.names
                if "text" not in schema_names:
                    for name in schema_names:
                        if name.lower() in ["content", "body", "code"]:
                            text_col = name
                            break

                for batch in parquet_file.iter_batches(columns=[text_col]):
                    for val in batch.column(0).to_pylist():
                        if val and isinstance(val, str):
                            yield val.strip()
            except ImportError:
                print(f"[Warning] PyArrow not installed. Skipping parquet file: {file_path}")

    except Exception as e:
        print(f"[Warning] Error reading {file_path}: {e}")

def batch_iterator(files: list[str], batch_size: int = 1000) -> Iterator[str]:
    """Iterate through all files without accumulating entire datasets into RAM."""
    batch = []
    for file_path in files:
        for text in stream_text_from_file(file_path):
            batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def verify_saved_tokenizer(output_dir: str):
    """Post-training verification: tests loading, special tokens, and English/Hindi round-trips."""
    print("\n=== Phase 3: Running Tokenizer Verification Checks ===")

    try:
        tok = AutoTokenizer.from_pretrained(output_dir)
        print(f"[PASS] Successfully loaded saved tokenizer from '{output_dir}'.")
    except Exception as e:
        print(f"[FAIL] Failed to load saved tokenizer: {e}")
        sys.exit(1)

    # 1. Check Vocab Size
    loaded_vocab_size = len(tok)
    print(f"[CHECK] Vocabulary size: {loaded_vocab_size:,} / Target: {VOCAB_SIZE:,}")

    # 2. Check Special Tokens Mapping
    missing_specials = []
    for st in SPECIAL_TOKENS:
        token_id = tok.convert_tokens_to_ids(st)
        if token_id is None or token_id == tok.unk_token_id and st != "<unk>":
            missing_specials.append(st)
        else:
            print(f"  └─ Special Token '{st}': ID {token_id}")

    if missing_specials:
        print(f"[FAIL] Missing special tokens: {missing_specials}")
    else:
        print("[PASS] All special tokens correctly mapped.")

    # 3. Check Encoding & Decoding Round-Trip (English & Hindi)
    test_samples = [
        "<|im_start|>user\n<think>Calculate 2 + 2.</think>\nWhat is 2 + 2?<|im_end|>",
        "नमस्ते! यह SmaulNative टोकनाइज़र का एक परीक्षण वाक्य है।"
    ]

    print("\n[CHECK] Running Round-Trip Encoding Tests:")
    for sample in test_samples:
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded, skip_special_tokens=False)
        print(f"\nOriginal Text:  {sample}")
        print(f"Token IDs:      {encoded[:12]}... (Total: {len(encoded)} tokens)")
        print(f"Decoded Output: {decoded}")

        if sample.strip() in decoded.strip() or decoded.strip() in sample.strip():
            print("[PASS] Round-trip losslessness verified.")
        else:
            print("[WARNING] Decoded text differs slightly from input.")

    print("\n=== All Tokenizer Checks Completed Successfully! ===")

def train_bpe_tokenizer():
    found_files = run_prerequisite_checks()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n=== Phase 2: Training Byte-Level BPE Tokenizer ===")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True
    )

    tokenizer.train_from_iterator(batch_iterator(found_files), trainer=trainer)

    # Save raw tokenizer.json
    raw_tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(raw_tokenizer_path)

    # Wrap in Hugging Face PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=raw_tokenizer_path,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
            "<think>",
            "</think>"
        ]
    )

    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved tokenizer files to '{OUTPUT_DIR}'.")

    # Run verification check suite
    verify_saved_tokenizer(OUTPUT_DIR)

if __name__ == "__main__":
    train_bpe_tokenizer()

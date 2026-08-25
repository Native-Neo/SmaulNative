import argparse
import json
import os
import sys
from datasets import load_dataset

DEFAULT_DATASETS_DIR = "./datasets"
DEFAULT_TARGET_GB = 20.0

DEFAULT_DATASETS = [
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT"),
    ("enwiki", "HuggingFaceFW/finewiki", "en"),
    ("hiwiki", "HuggingFaceFW/finewiki", "hi"),
    ("finepdfs_hi", "HuggingFaceFW/finepdfs-edu", "hin_Deva")
]

def parse_args():
    parser = argparse.ArgumentParser(description="Download pre-training datasets with streaming limits.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_DATASETS_DIR,
        help="Directory to save output jsonl files."
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=DEFAULT_TARGET_GB,
        help="Target size cap in GB per dataset."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    datasets_dir = args.output_dir
    target_bytes = int(args.target_gb * 1024 * 1024 * 1024)

    os.makedirs(datasets_dir, exist_ok=True)
    print(f"Starting / Resuming dataset streaming into '{datasets_dir}' (Limit: {args.target_gb:.2f} GB per dataset)...")

    for name, path, config in DEFAULT_DATASETS:
        out_path = os.path.join(datasets_dir, f"{name}.jsonl")
        bytes_written = 0
        existing_lines = 0

        # Resume check: Verify file size and existing line count if resuming
        if os.path.exists(out_path):
            bytes_written = os.path.getsize(out_path)
            if bytes_written >= target_bytes:
                print(f"[{name}] Already reached target limit ({bytes_written / (1024**3):.2f} GB). Skipping.")
                continue

            print(f"[{name}] Found existing file ({bytes_written / (1024**3):.2f} GB). Counting records to skip...")
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                existing_lines = sum(1 for _ in f)
            print(f"[{name}] Resuming download from record #{existing_lines:,}...")

        print(f"\n---> Fetching {name} ({path} | config: {config}). Target: {args.target_gb:.2f} GB")

        try:
            ds = load_dataset(path, name=config, split="train", streaming=True)

            # Fast-forward through existing lines if resuming
            if existing_lines > 0:
                ds = ds.skip(existing_lines)

            # Append mode prevents overwriting downloaded data
            with open(out_path, "a", encoding="utf-8") as f_out:
                for item in ds:
                    text = item.get("text", "") or item.get("content", "")
                    if not text:
                        continue

                    line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                    line_bytes = len(line.encode("utf-8"))

                    f_out.write(line)
                    bytes_written += line_bytes

                    # Progress update every 500 MB
                    if bytes_written % (500 * 1024 * 1024) < line_bytes:
                        f_out.flush()
                        gb_written = bytes_written / (1024 ** 3)
                        print(f"[{name}] Total written: {gb_written:.2f} GB / {args.target_gb:.2f} GB")

                    if bytes_written >= target_bytes:
                        f_out.flush()
                        print(f"[{name}] Successfully reached target limit.")
                        break

        except Exception as e:
            print(f"Skipped or hit stream boundary for {name}: {e}")

    print("\nAll dataset downloads finished or reached target limits!")

if __name__ == "__main__":
    main()


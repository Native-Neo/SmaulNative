#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# Configuration
# ============================================================

OUTPUT_ROOT = Path("./datasets/raw")

# Maximum download size PER LANGUAGE.
# 40 GiB = 40 * 1024^3 bytes.
MAX_GIB_PER_LANGUAGE = 40

HINDI_REPO = "HuggingFaceFW/fineweb-2"
HINDI_PATH = "data/hin_Deva/train"

ENGLISH_REPO = "HuggingFaceFW/fineweb"

# Original FineWeb English 100BT sample.
ENGLISH_PATH = "data/100BT"


# ============================================================
# Hugging Face
# ============================================================

api = HfApi()


# ============================================================
# Helpers
# ============================================================

def gib_to_bytes(gib: float) -> int:
    return int(gib * 1024 ** 3)


def get_repo_files(
    repo_id: str,
    path: str,
) -> list[dict]:

    print(f"\nInspecting {repo_id}/{path}")

    info = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path=path,
        recursive=True,
    )

    files = []

    for item in info:
        if not hasattr(item, "path"):
            continue

        if not item.path.endswith(".parquet"):
            continue

        size = getattr(item, "size", None)

        if size is None:
            continue

        files.append(
            {
                "path": item.path,
                "size": int(size),
            }
        )

    files.sort(
        key=lambda x: x["path"]
    )

    if not files:
        raise RuntimeError(
            f"No Parquet files found in:\n"
            f"{repo_id}/{path}"
        )

    return files


def get_existing_size(
    directory: Path,
) -> int:

    total = 0

    if not directory.exists():
        return 0

    for path in directory.rglob("*"):

        if path.is_file():
            total += path.stat().st_size

    return total


# ============================================================
# Downloader
# ============================================================

def download_dataset(
    repo_id: str,
    dataset_path: str,
    output_name: str,
) -> None:

    output_dir = (
        OUTPUT_ROOT / output_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_bytes = gib_to_bytes(
        MAX_GIB_PER_LANGUAGE
    )

    files = get_repo_files(
        repo_id,
        dataset_path,
    )

    existing_bytes = get_existing_size(
        output_dir
    )

    print()
    print("=" * 70)
    print(f"DATASET: {output_name}")
    print("=" * 70)

    print(
        f"Files available: "
        f"{len(files):,}"
    )

    print(
        f"Existing data: "
        f"{existing_bytes / 1024**3:.2f} GiB"
    )

    print(
        f"Maximum: "
        f"{MAX_GIB_PER_LANGUAGE:.2f} GiB"
    )

    if existing_bytes >= max_bytes:

        print(
            "\nMaximum size already reached."
        )

        return

    downloaded_bytes = existing_bytes
    downloaded_files = 0
    skipped_files = 0

    for index, file_info in enumerate(
        files,
        start=1,
    ):

        filename = file_info["path"]
        file_size = file_info["size"]

        destination = (
            output_dir / filename
        )

        # ----------------------------------------------------
        # Already downloaded.
        # ----------------------------------------------------

        if destination.exists():

            actual_size = (
                destination.stat().st_size
            )

            if actual_size == file_size:

                downloaded_files += 1

                print(
                    f"[{index:,}/{len(files):,}] "
                    f"already exists: "
                    f"{filename}"
                )

                continue

            print(
                f"[{index:,}/{len(files):,}] "
                f"partial/corrupt file detected: "
                f"{filename}"
            )

            downloaded_bytes -= actual_size
            destination.unlink()

        # ----------------------------------------------------
        # Check size limit BEFORE downloading.
        # ----------------------------------------------------

        remaining = (
            max_bytes - downloaded_bytes
        )

        if file_size > remaining:

            skipped_files += 1

            print(
                f"\nSIZE LIMIT REACHED"
            )

            print(
                f"Current:   "
                f"{downloaded_bytes / 1024**3:.2f} GiB"
            )

            print(
                f"Next file: "
                f"{file_size / 1024**3:.2f} GiB"
            )

            print(
                f"Remaining: "
                f"{remaining / 1024**3:.2f} GiB"
            )

            print(
                "\nStopping instead of exceeding "
                "the configured limit."
            )

            break

        # ----------------------------------------------------
        # Download.
        # ----------------------------------------------------

        print(
            f"\n[{index:,}/{len(files):,}] "
            f"Downloading:"
        )

        print(
            f"  {filename}"
        )

        print(
            f"  Size: "
            f"{file_size / 1024**3:.2f} GiB"
        )

        print(
            f"  Total after: "
            f"{(downloaded_bytes + file_size) / 1024**3:.2f} GiB"
        )

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(output_dir),
            resume_download=True,
        )

        # ----------------------------------------------------
        # Verify the resulting file.
        # ----------------------------------------------------

        if not destination.exists():

            raise RuntimeError(
                f"Download returned but file "
                f"was not found:\n{destination}"
            )

        actual_size = (
            destination.stat().st_size
        )

        if actual_size != file_size:

            raise RuntimeError(
                f"Size verification failed:\n"
                f"File: {filename}\n"
                f"Expected: {file_size}\n"
                f"Actual: {actual_size}"
            )

        downloaded_bytes += actual_size
        downloaded_files += 1

        print(
            f"  OK — "
            f"{downloaded_bytes / 1024**3:.2f} GiB total"
        )

    print()
    print("=" * 70)
    print(f"{output_name.upper()} COMPLETE")
    print("=" * 70)

    print(
        f"Downloaded: "
        f"{downloaded_bytes / 1024**3:.2f} GiB"
    )

    print(
        f"Files: "
        f"{downloaded_files:,}"
    )

    print(
        f"Location: "
        f"{output_dir.resolve()}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FineWeb English + Hindi Limited Downloader")
    print("=" * 70)

    print(
        f"\nMaximum per language: "
        f"{MAX_GIB_PER_LANGUAGE} GiB"
    )

    # Hindi: FineWeb2
    download_dataset(
        repo_id=HINDI_REPO,
        dataset_path=HINDI_PATH,
        output_name="hindi",
    )

    # English: original FineWeb
    download_dataset(
        repo_id=ENGLISH_REPO,
        dataset_path=ENGLISH_PATH,
        output_name="english",
    )

    print()
    print("=" * 70)
    print("ALL DOWNLOADS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

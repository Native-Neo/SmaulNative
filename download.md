# download.py

Downloads FineWeb (English) and FineWeb-2 (Hindi) parquet shards from Hugging Face into
`./datasets/raw/{english,hindi}/`.

## Configuration

There are no CLI flags -- edit the constants at the top of the file if you want different
repos/paths/limits:

```python
OUTPUT_ROOT           = Path("./datasets/raw")
MAX_GIB_PER_LANGUAGE  = 40                          # per-language download cap
HINDI_REPO, HINDI_PATH     = "HuggingFaceFW/fineweb-2", "data/hin_Deva/train"
ENGLISH_REPO, ENGLISH_PATH = "HuggingFaceFW/fineweb", "data/100BT"
```

- `MAX_GIB_PER_LANGUAGE` is a hard per-language ceiling: it stops downloading once a language
  reaches the limit, so you can cap disk usage without touching the rest of the file.
- `ENGLISH_PATH = "data/100BT"` points at the original FineWeb English 100BT sample; change it to
  another FineWeb split if you want a different English subset.

## Run it

```bash
python download.py
```

## How it behaves

- **Safe to re-run**: it lists the available repo files, compares against what's already on disk,
  and skips files whose size matches. A partial or corrupt file (size mismatch) is deleted and
  re-downloaded, and a broken partial is *counted against* the cap so it can't silently eat your
  limit.
- **Size limit is checked *before* each download** (`download.py:225`), so it never starts a file
  that would push past `MAX_GIB_PER_LANGUAGE` -- it prints the current/remaing/next-file sizes and
  breaks instead of exceeding the cap.
- **Verification**: every download is checked after it lands; a file whose on-disk size doesn't
  match the repo's declared size raises a `RuntimeError`.
- Hugging Face access: you need an account/token for gated or rate-limited pulls
  (`huggingface-cli login`).
- Output goes straight under `./datasets/raw/` -- point `dataset.py`-consuming scripts
  (`tokenizer.py`, `train.py`) at `./datasets` (or wherever you move/symlink the parquet files).

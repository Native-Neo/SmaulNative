# stream_data.py

Stream Hugging Face Parquet records to stdout without downloading datasets to disk.

```bash
python stream_data.py --dataset hindi --max_records 1000 > streamed.jsonl
python stream_data.py --dataset all --min_chars 20 --max_chars 1000000 --workers 4 > out.jsonl
```

| Flag | Default | What it does |
|---|---|---|
| `--dataset` | `all` | `hindi`, `english`, `openthoughts`, or `all` |
| `--min_chars` / `--max_chars` | `20` / `1000000` | passed to `filter_data.filter_text` |
| `--max_records` | `0` | kept-record cap (`0` = unlimited; counts *kept*, not raw rows) |
| `--workers` | auto (min(4, CPUs), cap 16) | row-group fetch threads (`SMAUL_STREAM_WORKERS` overrides) |

Library resume: `stream_dataset(name, min_chars, max_chars, start_dataset, start_file,
start_record, with_position, workers)`.

## Behavior

- File listing retries with backoff; row-group reads retry transient IO with jitter
  (`SMAUL_STREAM_RETRIES`, 1-10, default 6) but not fatal errors; metadata opens retry.
- At most 8 row-groups are in flight (bounded RAM); workers cap at 16.
- Text extraction prefers `text`/`content`/`document`/`body`/`code`, then `conversations`,
  then `prompt`+`completion` pairs, else `dataset.extract_text` heuristics.
- `record` positions count **raw** rows, so kept-stream positions have gaps by design.
- `start_record` requires `start_dataset` + `start_file`; unknown resume files fail fast
  before streaming (no more typo-then-stream-everything-later).
- Stdout is buffered (500 lines/syscall) for million-row throughput; `BrokenPipeError`
  (piped to `head`) exits quietly. Auth uses `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`, read live
  per call.

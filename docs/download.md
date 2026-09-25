# download.py

Downloads Hugging Face Parquet datasets and re-shards them into uniform local Parquet
shards with a resumable manifest -- no tokenization involved.

## Datasets

| Name | Repo | Path |
|---|---|---|
| `hindi` | `HuggingFaceFW/fineweb-2` | `data/hin_Deva/train` |
| `english` | `HuggingFaceFW/fineweb` | `data/100BT` |
| `openthoughts` | `open-thoughts/OpenThoughts3-1.2M` | `data` |

`--languages all` downloads every dataset in `DATASET_CONFIGS` (`hindi` + `english` +
`openthoughts`).

## Run it

```bash
python download.py --languages hindi english --max_rows 100000
```

| Flag | Default | What it does |
|---|---|---|
| `--languages` | `all` | `hindi`, `english`, `openthoughts`, and/or `all` |
| `--max_rows` | `0` | max rows per dataset; `0` means unlimited |
| `--shard_rows` | `100000` | rows per output shard (must be positive; index caps at 9999 shards) |
| `--compression` | `zstd` | `zstd`, `snappy`, `gzip`, or `none` |
| `--output_dir` | `./datasets` | root; each dataset lands in `<output_dir>/<name>/` |
| `--temp_dir` | `./datasets/.temp_raw` | raw download staging area |
| `--no_clean_temp` | off | keep staged raw files after processing |
| `--prune` | off | delete orphan `shard_*.parquet` files not listed in the manifest |

Gated repos use your cached Hugging Face credentials (`huggingface-cli login`). HF API clients
are constructed per call (no import-time token freeze). Listing and downloads retry with
backoff; `ShardWriter` validates its args and writes shards atomically (tmp + rename).

## Output & resume

Each dataset directory holds `shard_*.parquet` (single `text` column) plus a
`manifest.json` tracking total rows, committed shards, completed raw files, and the
in-progress file/row position. Re-running resumes where it stopped: finished raw files
are skipped, orphan shard files not listed in the manifest are **kept** with a warning
(pass `--prune` to delete them), and a manifest whose row count disagrees with its shards
is repaired (not a fatal error). Corrupt manifests and malformed shard entries are handled
gracefully instead of deleting valid data. Transient failures flush the buffered rows to a
partial shard before aborting, so at most one batch -- not a full `shard_rows` -- is redone.
Point `train.py`
`--data` at the output root -- `dataset.py` discovers shards recursively.

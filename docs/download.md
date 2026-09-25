# download.py

Downloads Hugging Face Parquet datasets and re-shards them into uniform local Parquet
shards with a resumable manifest -- no tokenization involved.

## Datasets

| Name | Repo | Path |
|---|---|---|
| `hindi` | `HuggingFaceFW/fineweb-2` | `data/hin_Deva/train` |
| `english` | `HuggingFaceFW/fineweb` | `data/100BT` |
| `openthoughts` | `open-thoughts/OpenThoughts3-1.2M` | `data` |

`--languages all` currently downloads `hindi` + `english` only (`download.py:247`).

## Run it

```bash
python download.py --languages hindi english --max_rows 100000
```

| Flag | Default | What it does |
|---|---|---|
| `--languages` | `all` | `hindi`, `english`, `openthoughts`, and/or `all` |
| `--max_rows` | `0` | max rows per dataset; `0` means unlimited |
| `--shard_rows` | `100000` | rows per output shard |
| `--compression` | `zstd` | `zstd`, `snappy`, `gzip`, or `none` |
| `--output_dir` | `./datasets` | root; each dataset lands in `<output_dir>/<name>/` |
| `--temp_dir` | `./datasets/.temp_raw` | raw download staging area |
| `--no_clean_temp` | off | keep staged raw files after processing |

Gated repos use your cached Hugging Face credentials (`huggingface-cli login`).

## Output & resume

Each dataset directory holds `shard_*.parquet` (single `text` column) plus a
`manifest.json` tracking total rows, committed shards, completed raw files, and the
in-progress file/row position. Re-running resumes where it stopped: finished raw files
are skipped, orphan shard files not listed in the manifest are deleted, and a manifest
whose row count disagrees with its shards raises `RuntimeError`. Point `train.py`
`--data` at the output root -- `dataset.py` discovers shards recursively.

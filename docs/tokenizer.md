# tokenizer.py

Trains a single byte-level BPE tokenizer (vocab + merges + `<pad>`/`<eos>` specials) over
everything `dataset.py` can discover under `--dataset_dir`.

## Run it

```bash
python tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json --vocab_size 32768
```

| Flag | Default | What it does |
|---|---|---|
| `--dataset_dir` | `./datasets` | recursively scanned for text (txt/json/jsonl/csv/parquet/source) |
| `--output` | `./tokenizer.json` | where the trained tokenizer is saved |
| `--vocab_size` | `32768` | target vocab size |
| `--min_frequency` | `2` | minimum pair frequency to merge |

## How it works

- Builds a `Tokenizers` `BPE` model with the **ByteLevel** pre-tokenizer, decoder, and
  post-processor (`tokenizer.py:33`), so the vocabulary is strictly byte-level and never runs out
  of OOV handling -- any byte sequence can be encoded and decoded losslessly.
- `SPECIAL_TOKENS = ["<pad>", "<eos>"]` (`tokenizer.py:18`) are added automatically. These two are
  load-bearing: `dataset.py`'s `TokenizerWrapper` **requires** both to exist, and raises an error
  if you point `--tokenizer_path` at a tokenizer that lacks them (`dataset.py:26`).
- Reads text via `dataset.discover_files()` + `iter_texts()`, so `.json/.jsonl/.csv/.parquet`
  containers are decoded down to raw text first -- it's not just plain `.txt`.
- Saves everything (vocab, merges, specials) into one `tokenizer.json` file (`tokenizer.py:49`).

## When to run it yourself

- You usually don't need to run this by hand: `train.py` auto-trains one at `--tokenizer_path` if
  it doesn't exist yet.
- Run it yourself when you want to pre-train a tokenizer **once** and reuse it across several
  `train.py` runs (pretrain + multiple SFT branches) without retraining each time -- SFT after
  pretrain must reuse the pretrained tokenizer so vocab stays consistent.

# dataset.py

Not a script -- a library imported by `tokenizer.py`, `train.py`, and `merge_moe.py`. Nothing to
run directly. It provides file discovery, text extraction, a tokenizer wrapper, and the two
training datasets (pretrain stream + SFT).

## File discovery & text extraction

- **`discover_files(dir)`** (`dataset.py:67`) recursively walks a directory and returns every
  file whose suffix is supported, sorted. Supported suffixes:
  - text/data: `.txt`, `.jsonl`, `.json`, `.csv`, `.parquet`
  - source: `.py`, `.cpp`, `.c`, `.h`, `.rs`, `.js`, `.ts`, `.go`, `.java`, `.sh`, `.md` and many
    more (the full list is `SUPPORTED_SUFFIXES` at `dataset.py:58`).
- **`iter_texts(files, resume_file, resume_record)`** (`dataset.py:90`) yields
  `(text, file_path, record_index)` tuples, decoding each format to raw text:
  - plain text: one line per record
  - `.jsonl`: each JSON line
  - `.json`: objects from the top-level list (or `data` key)
  - `.csv`: each row
  - `.parquet`: each row (needs `pyarrow`)
  - For structured records, `extract_text()` (`dataset.py:76`) pulls the first of
    `text/content/document/body/code/prompt/completion` values, so a single code path handles
    pretrain data and SFT-style fields alike.
  - **Resume is record-index based** (not byte-offset), so it works uniformly across all formats
    without per-format seek logic (`dataset.py:91`). A bad/unparseable record is skipped with a
    `[WARN]`.

## TokenizerWrapper

`TokenizerWrapper` (`dataset.py:17`) wraps a trained `tokenizers.Tokenizer` so the rest of the
code only deals with `encode()`/`decode()` plus two ids:

```python
tok.pad_token_id   # "<pad>" id
tok.eos_token_id   # "<eos>" id
tok.get_vocab_size()
```

It **errors at construction** if the tokenizer lacks `<pad>`/`<eos>` (dataset.py:26) -- train it
with `tokenizer.py`, don't point at an unrelated `tokenizer.json`.

## Pretraining: `PretrainStream`

`PretrainStream` (`dataset.py:160`) is an `IterableDataset` that streams fixed-length
`(input_ids, labels)` chunks of size `ctx_len` for causal LM training:

- Tokenizes each record, appends the `eos` token, and buffers tokens across records so document
  boundaries don't create ragged chunks.
- Each training example is `chunk[:-1]` shifted to `chunk[1:]` (`x`/`y`), the standard causal
  LM next-token setup.
- Tracks `last_pos` (file + record index) as it goes, which `train.py` reads back for resume, plus
  carries `buffer_tokens` so a partially-consumed document isn't lost on restart.

## SFT: `SFTDataset`

`SFTDataset` (`dataset.py:280`) loads conversation JSON/JSONL and applies **loss masking** so only
assistant turns are trained on. Records must be shaped like:

```json
{"conversations": [{"from": "user", "value": "..."}, {"from": "assistant", "value": "..."}]}
```

- Records can live in nested subfolders and across any number of `.json`/`.jsonl` files
  (`discover_sft_records`, `dataset.py:252`).
- `_add_speaker_and_signal` (`dataset.py:197`) rewrites each turn to `"User/Assistant: <value>\n\n"`
  and `_preprocess_conversation` (`dataset.py:213`) sets every **user** token (and the
  `"Assistant: "` prefix) to `IGNORE_INDEX = -100` so they contribute nothing to the loss
  (`dataset.py:230`). Padding is filled with `IGNORE_INDEX` too.
- Every record is truncated/padded to `ctx_len` (`dataset.py:239`).

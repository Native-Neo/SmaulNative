# tokenizer.py

The repository uses a custom lightweight tokenizer stored as one JSON file. It is designed for the
project's English/Hindi training data and does not depend on the Hugging Face `tokenizers` BPE runtime.

## Train

```bash
python tokenizer.py train \
    --fromdataset ./datasets \
    --vocab-size 8000 \
    --word-budget 40000 \
    --output ./tokenizer.json
```

Important options:

- `--fromdataset`: file or directory containing training text.
- `--vocab-size`: maximum vocabulary size (must equal `train.py --vocab`; quickstart uses `8000`).
- `--word-budget`: maximum number of whole-word entries considered.
- `--max-records`: optional record limit; `0` means unlimited.
- `--output`: output JSON path (parent dirs created; written atomically).

## Vocabulary

The tokenizer reserves `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<|im_start|>`, `<|im_end|>`,
`<think>`, and `</think>` (ChatML tags are real tokens, preserved -- not stripped), plus
`<cap>` and `<upper>` case markers.
English words are normalized to lowercase with case markers retained (mixed case like
`hELLO`/`eBay` lowercases lossily by design). Devanagari text is represented using
whole words where possible, then Devanagari grapheme units (ZWNJ/ZWJ-aware) and individual
characters as fallbacks. Numbers cover ASCII and Devanagari digits. Hyphens split words
(`well-known` -> `well`, `-`, `known`); common multi-spaces (`"  "`, `"\n\n"`) are single tokens.
Invalid `<unused_*>` ids decode as `<unk>` instead of vanishing silently.

## Determinism

Directory inputs are traversed in sorted path order (symlinks escaping the corpus root are
skipped), so the same corpus and settings produce stable vocabulary ordering. Text is decoded
as UTF-8-SIG with undecodable bytes replaced (`U+FFFD`), so one bad file never kills training.
Oversized plain files (>10MB) and JSON files (>50MB) are skipped with a warning; Parquet reads
prefer a text column and otherwise scan string columns only. `case_stats` is no longer stored
(it bloated `tokenizer.json` and was never used).

## Encoding

```bash
python tokenizer.py encode --tokenizer ./tokenizer.json --text "Hello नमस्ते"
python tokenizer.py encode --tokenizer ./tokenizer.json --text-file ./prompt.txt
python tokenizer.py decode --tokenizer ./tokenizer.json --ids "1 2 3"
```

`--text-file` avoids shell-escaping pain for Hindi/newlines; bad `--ids` fail with a clear
error. `SmaulTokenizer.encode()` returns a list-like object with an `.ids` property for
compatibility with the training code.

## Automatic tokenizer

`train.py` builds the tokenizer itself via `ensure_tokenizer`: an existing file is reused
only when its vocabulary size matches `--vocab` and its format version is current
(version 7, `tokenizer.VERSION`); otherwise it is rebuilt from the training data. The
manual `train` command above uses the same builder with defaults `--vocab-size 32000`
and `--word-budget 20000`. Streaming builds (`stream_name != "none"`) require a positive
`--max-records` cap -- unbounded FineWeb downloads are refused.

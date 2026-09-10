# tokenizer.py

The repository uses a custom lightweight tokenizer stored as one JSON file. It is designed for the
project's English/Hindi training data and does not depend on the Hugging Face `tokenizers` BPE runtime.

## Train

```bash
python tokenizer.py train \
    --fromdataset ./datasets \
    --vocab-size 65536 \
    --word-budget 40000 \
    --output ./tokenizer.json
```

Important options:

- `--fromdataset`: file or directory containing training text.
- `--vocab-size`: maximum vocabulary size.
- `--word-budget`: maximum number of whole-word entries considered.
- `--max-records`: optional record limit; `0` means unlimited.
- `--output`: output JSON path.

## Vocabulary

The tokenizer reserves `<pad>`, `<unk>`, `<bos>`, and `<eos>`, plus `<cap>` and `<upper>` case markers.
English words are normalized to lowercase with case markers retained. Devanagari text is represented using
whole words where possible, then Devanagari grapheme units and individual characters as fallbacks.

## Determinism

Directory inputs are traversed in sorted path order, so the same corpus and settings produce stable
vocabulary ordering. Text is decoded with replacement rather than silently deleting invalid bytes.

## Encoding

```bash
python tokenizer.py encode --tokenizer ./tokenizer.json --text "Hello नमस्ते"
python tokenizer.py decode --tokenizer ./tokenizer.json --ids "1 2 3"
```

`SmaulTokenizer.encode()` returns a list-like object with an `.ids` property for compatibility with the
training code.

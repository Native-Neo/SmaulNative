# tokenizer.py

Byte-level codec: the vocabulary is exactly the 256 UTF-8 byte values. Text is
encoded once via UTF-8 (`encode_bytes`), the model consumes byte IDs directly,
and the output head produces exactly 256 logits. There is no BPE/WordPiece/
SentencePiece, no word tokenizer, and no vocabulary training step.

```bash
python tokenizer.py encode --text "Hello नमस्ते"
# 72 101 108 108 111 32 230 164 168 224 164 174 224 164 184 224 165 141 224 165 135
python tokenizer.py decode --ids "72 101 108 108 111"
# Hello
```

## Codec

- `encode_bytes(text)` -> `list[int]`: raw UTF-8 bytes, every value 0-255.
- `decode_bytes(ids)` -> `str`: invalid ids raise; undecodable bytes decode
  with `errors="replace"` so arbitrary byte sequences never crash and never
  silently drop bytes.
- `IncrementalByteDecoder`: streaming decode for generation. Complete prefixes
  emit immediately; a trailing incomplete UTF-8 sequence is buffered until more
  bytes arrive (`flush()` replaces a dangling tail). Split multi-byte chars
  across chunk boundaries are preserved, not dropped.
- `SmaulTokenizer` (also `ByteTokenizer`): fixed 256-entry file
  `{"version": 9, "kind": "byte", "vocab_size": 256}`. Loading a legacy
  word-level file fails with a clear error telling you to delete it and
  re-run -- old embedding rows index a different token space and cannot be
  migrated.

## Corpus readers (kept)

`read_texts` and the record coercers (`_coerce_list_text`, `_string_values`,
`_record_text`) still turn heterogeneous dataset files (plain text, CSV,
JSON/JSONL, Parquet) into plain strings. Directory inputs are traversed in
sorted path order (symlinks escaping the corpus root are skipped). Text is
decoded as UTF-8-SIG with undecodable bytes replaced (`U+FFFD`). Oversized
plain files (>10MB) and JSON files (>50MB) are skipped with a warning; Parquet
reads prefer a text column and otherwise scan string columns only.

## Automatic tokenizer

`train.py` writes the byte file itself via `ensure_tokenizer`: an existing
valid byte file is reused, a legacy/unreadable one is rewritten. No training
data is needed for the codec -- `--data` is still required for the corpus.
`--vocab` must be `256`.

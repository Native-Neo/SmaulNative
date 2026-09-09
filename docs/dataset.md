# dataset.py

`dataset.py` provides file discovery, text extraction, tokenizer wrapping, pretraining streaming, and SFT
dataset loading.

## Supported files

Recursive discovery supports `.txt`, `.text`, JSON/JSONL, CSV, Parquet, and common source/document
extensions such as Python, C/C++, Rust, JavaScript, Java, Go, shell, Markdown, YAML, TOML, XML, and SQL.

Structured records prefer `text`, `content`, `document`, `body`, or `code`. Records containing both
`prompt` and `completion` preserve both fields in the extracted training text.

## Pretraining

`PretrainStream` tokenizes records, appends `<eos>`, carries tokens across record boundaries, and emits
fixed-length causal-LM pairs of `ctx_len` tokens. Resume state is tracked by source file/record plus the
remaining token buffer.

## SFT

SFT records must contain a `conversations` list. Each valid turn needs string `from` and `value` fields.
`user`/`human` turns are normalized to `User`; `assistant`/`gpt` turns are normalized to `Assistant`.
Unknown roles are retained in the input but are completely masked from the loss.

Only assistant response text is a training target. Speaker prefixes and all non-assistant tokens use
`IGNORE_INDEX = -100`. Padding is also masked.

Malformed JSONL records and malformed SFT turns are skipped rather than terminating the whole dataset
load. A dataset containing no valid SFT conversations raises an explicit error.

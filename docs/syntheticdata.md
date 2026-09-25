# syntheticdata.py

Generates a synthetic bilingual (English/Hindi) instruction-following dataset -- math, algorithms,
data structures, cyber security -- with zero duplicate prompts, as JSONL and/or Parquet.

## Run it

```bash
python syntheticdata.py --count 250000 --format both --output-dir ./datasets/synthetic
```

| Flag | Default | What it does |
|---|---|---|
| `--count` | `250000` | number of unique samples to generate |
| `--format` | `both` | `jsonl`, `parquet`, or `both` |
| `--output-dir` | `./datasets/synthetic` | where `synthetic_bilingual.jsonl` / `.parquet` land (separate from real data) |
| `--seed` | none | random seed for reproducible generation |
| `--overwrite` | off | allow replacing existing dataset files |

## What it generates

Six kinds of records, each produced by a dedicated generator picked at random per sample:

- **`gen_linear_equation`** (`syntheticdata.py:43`) -- `ax + b = c` algebra in English **and**
  Hindi, with a worked step-by-step solution. Answers teach the exact fraction plus an
  explicitly approximate decimal (`x = (c-b)/a ≈ …`), never a rounded value as exact.
- **`gen_quadratic_equation`** (`syntheticdata.py:74`) -- `ax^2 + bx + c = 0`, roots constructed
  from random factors, solved via the discriminant/quadratic formula.
- **`gen_system_linear_equations`** (`syntheticdata.py:102`) -- 2x2 systems with verification.
- **`gen_sorting_algorithm_code`** (`syntheticdata.py:178`) -- Quick Sort in Python,
  JavaScript, C++, or Rust, with complexity analysis. Prompts carry a random example array
  and sort order so code prompts are unique (previously only 4 variants existed). C++/Rust
  samples are bounds-safe (no `j -= 1` underflow / unbounded `while` scans).
- **`gen_data_structure_code`** (`syntheticdata.py:266`) -- Stack/Queue implementations in
  Python, C++, or Java, with randomized capacity/extra requirements for uniqueness.
- **`gen_cyber_security_qa`** (`syntheticdata.py:279`) -- XSS, CSRF, MitM, buffer overflow,
  password hashing, with mitigations and randomized audience/scenario.

The prompt language/code in each sample always matches the emitted code: the fenced
language tag comes from `_LANG_FENCE` (`syntheticdata.py:175`) / `_DS_FENCE`
(`syntheticdata.py:263`) keyed by the same language the code body was written in.

## Zero-duplicate guarantee & format

- **Uniqueness**: every generated prompt is BLAKE2-hashed into a compact digest set
  (32 bytes/entry, not full strings); a duplicate is thrown away and regenerated, so all
  `--count` records are distinct. It stops hard at `target_count * 10` attempts, so don't
  ask for more than the combinatorial space can realistically supply. `--seed` reproduces a run.
- **Streaming**: `main()` generates and writes incrementally (`iter_unique_dataset` +
  `export_dataset_iter`, 10k-row Parquet batches) so million-scale runs never hold the whole
  dataset in RAM. Writes are atomic (tmp + rename) and refuse to overwrite without `--overwrite`.
- **ChatML output**: each record's `text` field is ChatML-formatted
  (`<|im_start|>user/assistant ... <|im_end|>`, optional `<think>` chain-of-thought)
  via `format_chatml`, which escapes embedded control tags in generated content, alongside
  the raw `instruction`/`response`/`think`
  fields and a `domain` tag (e.g. `math_algebra_hi`, `code_datastructures`).
- Parquet export (compressed with zstd) needs `pyarrow` installed; JSONL always works.
- Output defaults to `./datasets/synthetic`, separate from `download.py` output, so synthetic
  files are not silently mixed into real-data training passes.

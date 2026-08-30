# syntheticdata.py

Generates a synthetic bilingual (English/Hindi) instruction-following dataset -- math, algorithms,
data structures, cyber security -- with zero duplicate prompts, as JSONL and/or Parquet.

## Run it

```bash
python syntheticdata.py --count 250000 --format both --output-dir ./datasets
```

| Flag | Default | What it does |
|---|---|---|
| `--count` | `250000` | number of unique samples to generate |
| `--format` | `both` | `jsonl`, `parquet`, or `both` |
| `--output-dir` | `./datasets` | where `synthetic_bilingual.jsonl` / `.parquet` land |

## What it generates

Six kinds of records, each produced by a dedicated generator picked at random per sample:

- **`gen_linear_equation`** (`syntheticdata.py:46`) -- `ax + b = c` algebra in English **and**
  Hindi, with a worked step-by-step solution.
- **`gen_quadratic_equation`** (`syntheticdata.py:82`) -- `ax^2 + bx + c = 0`, roots constructed
  from random factors, solved via the discriminant/quadratic formula.
- **`gen_system_linear_equations`** (`syntheticdata.py:116`) -- 2x2 systems with verification.
- **`gen_sorting_algorithm_code`** (`syntheticdata.py:206`) -- Quick/Merge/Bubble/Insertion/
  Selection sort in Python, JavaScript, C++, or Rust, with complexity analysis.
- **`gen_data_structure_code`** (`syntheticdata.py:314`) -- Stack/Queue implementations in
  Python, C++, or Java.
- **`gen_cyber_security_qa`** (`syntheticdata.py:332`) -- XSS, CSRF, MitM, buffer overflow,
  password hashing, with mitigations.

The prompt language/code in each sample always matches the emitted code: the sorting and
data-structure generators were fixed so the fenced language tag and the actual code body line up
(`syntheticdata.py:143`, `:225`).

## Zero-duplicate guarantee & format

- **Uniqueness**: every generated prompt is hashed into a `seen_prompts` set
  (`syntheticdata.py:372`); a duplicate is thrown away and regenerated, so all `--count` records
  are distinct. It stops hard at `target_count * 10` attempts (`syntheticdata.py:387`), so don't
  ask for more than the combinatorial space can realistically supply.
- **ChatML output**: each record's `text` field is ChatML-formatted
  (`<|im_start|>user/assistant ... <|im_end|>`, optional `thinking`/`response` chain-of-thought)
  via `format_chatml` (`syntheticdata.py:361`), alongside the raw `instruction`/`response`/`think`
  fields and a `domain` tag (e.g. `math_algebra_hi`, `code_datastructures`).
- Parquet export (compressed with ZSTD) needs `pyarrow` installed; JSONL always works.
- Output lands in the same `./datasets` dir `download.py` uses, so both can feed one `dataset.py`
  pass.

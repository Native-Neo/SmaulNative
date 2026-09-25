# filter_data.py

Lightweight stdin/stdout filters for streamed training records (no dataset stored).

```bash
python stream_data.py --dataset hindi --max_records 1000 | \
    python filter_data.py --dataset hindi --min_chars 20 --max_chars 1000000 > kept.jsonl
```

| Flag | Default | What it does |
|---|---|---|
| `--dataset` | `auto` | `auto`, `hindi`, `english`, `openthoughts` (`hindi`/`english` enforce script ratios) |
| `--min_chars` / `--max_chars` | `20` / `1000000` | length gates (validated) |
| `--min_unique` | `8` | min distinct chars (sampled over first 50k chars, not whole 1M strings) |
| `--min_dev` / `--min_latin` | `8` / `12` | min Devanagari/Latin counts for `hindi`/`english` |
| `--dev_ratio` / `--latin_ratio` | `0.20` / `0.50` | min script-to-letter ratios |

## Behavior

- Recognizes `text`/`content`/`document`/`body`/`code`/`prompt`/`completion` (same keys as
  `dataset.py`), including list-valued prompts (chat history) -- unknown shapes return `None`.
- Cheap length pre-check runs *before* NFKC normalization (huge-input DoS guard); code
  indentation is preserved (only blank edge lines stripped).
- `auto`/`openthoughts` apply no script gate (lenient); mixed bilingual text that fails both
  script gates should use `--dataset auto` or tuned thresholds.

# convert_linear_to_gguf.py

Export a SmaulLinear checkpoint to GGUF (FP8 weights dequantized per tile at export).

```bash
python convert_linear_to_gguf.py ./runs/linear ./runs/linear.gguf --dtype f16 --overwrite
```

| Arg | Default | What it does |
|---|---|---|
| `input_dir` | *(required)* | checkpoint dir (`config.json` + `model.safetensors` + `tokenizer.json`) |
| `output` | *(required)* | GGUF path (parent dirs created) |
| `--dtype` | `f16` | `f32` or `f16` tensor storage |
| `--overwrite` | off | refuse to clobber an existing output unless set |

## Behavior

- Tokenizer ids are capped (`MAX_VOCAB_IDS`): a crafted `tokenizer.json` with a giant id is
  rejected instead of allocating a huge list. Gaps and negative ids raise.
- Every `.w8` must have a matching `.sc` with sane shapes; tile width must divide `in_f` and
  match `config.tile` (zero-tile writes are rejected, not silently exported as raw codes).
- Config values are range-checked; `context_length` comes from the checkpoint (`ctx_len` /
  `context_length`, default 512) instead of a hardcoded 2048.
- Export streams one tensor at a time (peak RAM ~1 tensor, not 2x the model) and writes via
  tmp + atomic rename.
- Needs the optional `gguf` package (`pip install gguf`).

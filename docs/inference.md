# inference.py + infer_linear.py

`inference.py` is the `LinearInference` engine all frontends share; `infer_linear.py` is the
single-prompt CLI built on it.

## Engine

```python
from inference import LinearInference
engine = LinearInference("./runs/linear", device="auto", dtype="auto")
print(engine.generate("Hello world", max_new_tokens=64))
for chunk in engine.stream("Hello world", max_new_tokens=64, seed=0):
    print(chunk, end="", flush=True)
```

- Init validates that checkpoint `vocab_size` matches `tokenizer.json` (mismatch raises
  `ValueError` instead of silently mistokenizing) and accepts `device` (`auto`/`cpu`/`cuda`)
  and `dtype` (`auto`/`fp32`/`bf16`).
- The model window is 512 tokens (`MODEL_WINDOW`): prompts longer than that warn and use the
  last 512; prompts over `MAX_PROMPT_TOKENS` (4096) raise `ValueError`.
- `max_new_tokens` must be in `[1, 4096]`. Over-aggressive `top_k`/`top_p` filtering that
  leaves no finite logits ends gracefully (EOS) instead of a `multinomial` crash.
- `chat_prompt(messages, system)` allowlists roles (`user`/`assistant`/`tool`; pass system
  instructions via `system=`, not a `system` role) and neutralizes fake `System:`/`Assistant:`
  header lines inside content.

## Single-prompt CLI

```bash
python infer_linear.py --model ./runs/linear --prompt "Hello world" --max 64 \
    --temp 0.7 --topk 50 --topp 0.95 --rep 1.05 --device auto --dtype auto
```

| Flag | Default | What it does |
|---|---|---|
| `--model` | `./runs/linear` | checkpoint dir |
| `--prompt` | `Hello world` | prompt text (capped at 200k chars) |
| `--max` | `64` | new tokens, `[1, 4096]` |
| `--temp` / `--topk` / `--topp` / `--rep` | `0.7` / `50` / `0.95` / `1.05` | sampling (validated) |
| `--device` / `--dtype` | `auto` | forwarded to `LinearInference` |

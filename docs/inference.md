# inference.py

`inference.py` is the `LinearInference` engine all frontends share (`infer_cli.py`,
`infer_server.py`). There is no `infer_linear.py`; `docs/inference.md` used to name one.

## Engine

```python
from inference import LinearInference
engine = LinearInference("./runs/linear", device="auto", dtype="auto")
print(engine.generate("Hello world", max_new_tokens=64))
for chunk in engine.stream("Hello world", max_new_tokens=64, seed=0):
    print(chunk, end="", flush=True)
```

- Init validates that checkpoint `vocab_size` matches `tokenizer.json` (mismatch raises
  `ValueError` instead of silently mistokenizing) and accepts `device` (`auto`/`cpu`/`cuda`),
  `dtype` (`auto`/`fp32`/`bf16`), plus optional `architecture` / `embedding_storage`
  overrides that are validated against the checkpoint rather than trusted.
- The model window is 262144 tokens (`MODEL_WINDOW`): prompts longer than that warn and use
  the last 262144; prompts over `MAX_PROMPT_TOKENS` (65536) raise `ValueError`.
- `max_new_tokens` must be in `[1, 65536]`. Over-aggressive `top_k`/`top_p` filtering that
  leaves no finite logits ends gracefully (EOS) instead of a `multinomial` crash.
- `chat_prompt(messages, system)` allowlists roles (`user`/`assistant`/`tool`; pass system
  instructions via `system=`, not a `system` role) and neutralizes fake `System:`/`Assistant:`
  header lines inside content.

## Incremental decoding

Generation does **not** re-run the prompt for every new token. Linear attention's recurrence
state is O(D²) per (batch, head) and independent of the sequence length, so the engine runs
the prefix once and then advances one token at a time:

```
prefill(ids[-MODEL_WINDOW:])  ->  logits, state
step(next_token, state)       ->  logits, state        (repeated)
```

`SmaulLinear.prefill` / `SmaulLinear.step` do the work; `LinearInference._prefill` / `_step`
wrap them. A step is used only while the carried state covers exactly the window the old
re-forward path would have used. The recurrence is a *sum over everything it has absorbed*,
so once the window slides (the state would still hold tokens the windowed forward excludes)
the engine re-prefills instead of stepping — which is precisely the truncation the old code
got for free by re-forwarding. Generating N tokens goes from O(N · window) forwards to
O(window + N · D²).

Measured, 2 threads, rawr at `--rawr-sparsity 0.9`, generating N tokens after a prompt of T:

| model | T | N | re-forward | prefill + N steps | speedup |
|---|---|---|---|---|---|
| V512 d128 L2 | 256 | 8 | 171 ms | 71 ms | 2.4x |
| V512 d128 L2 | 1024 | 32 | 1849 ms | 150 ms | 12.3x |
| V8000 d512 L8 | 256 | 8 | 3231 ms | 600 ms | 5.4x |
| V8000 d512 L8 | 1024 | 32 | 74.2 s | 3.45 s | **21.5x** |

**A per-token step is not bit-identical to a batched forward, and cannot be.** Torch selects
a different GEMM/SpMM kernel for a `[1, d]` input than for a `[T, d]` one, so the same row of
an activation sums in a different order. Measured: the same row of `x @ W.T` differs by
4.7e-07 relative at T=12 and 4.1e-07 at T=128 (d=512, V=8000); the Rawr `SparseLinear` head
differs from its dense equivalent by 6.0e-07. That is fp32 epsilon, far below anything a
trained model would notice, but on a randomly-initialised model the top logits are nearly
tied and a perturbation that small can flip an `argmax` — so generated text is *statistically*
the same, not byte-identical. `tests/test_last_token.py` pins the logits to 1e-4 relative
rather than comparing text, for that reason.

The engine falls back to the per-token re-forward if the model has no `prefill`/`step` (a
`__new__`-constructed test double, or a model predating them), so output and behaviour are
unchanged either way.

## Single-prompt CLI

```bash
python infer_cli.py --model ./runs/linear --prompt "Hello world" --max 64 \
    --temp 0.7 --topk 50 --topp 0.95 --rep 1.05 --device auto --dtype auto
```

| Flag | Default | What it does |
|---|---|---|
| `--model` | `./runs/linear` | checkpoint dir |
| `--prompt` | `Hello world` | prompt text (capped at 200k chars) |
| `--max` | `64` | new tokens, `[1, 65536]` |
| `--temp` / `--topk` / `--topp` / `--rep` | `0.7` / `50` / `0.95` / `1.05` | sampling (validated) |
| `--device` / `--dtype` | `auto` | forwarded to `LinearInference` |

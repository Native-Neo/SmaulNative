# infer_cli.py

Interactive chat CLI over `LinearInference`.

```bash
python infer_cli.py --model ./runs/linear --device auto --dtype auto \
    --temperature 0.7 --top-k 50 --top-p 0.95 --repeat-penalty 1.05 \
    --max-tokens 256 --max-history 40
```

Commands inside the session: `/clear`, `/system <text>`, `/exit` (`Ctrl-D`/`Ctrl-C` exits).

## Behavior

- Sampling flags are validated up front (`--max-tokens` in `[1, 4096]`, `--temperature >= 0`,
  `0 < --top-p <= 1`, `--repeat-penalty > 0`).
- History is capped at `--max-history` turns (default 40): oldest turns are dropped with a
  `[history trimmed]` notice. This matches the engine reality -- the model only attends to the
  last 512 tokens, so unbounded history would silently forget early context while slowing
  every turn.
- `Ctrl-C` mid-answer stops generation: an empty interruption discards the pending user turn
  (nothing saved); a partial answer is kept with a ` [stopped]` suffix instead of masquerading
  as a complete turn.

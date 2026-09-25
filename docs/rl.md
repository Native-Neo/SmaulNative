# rl.py

Human-preference collection + GRPO training for SmaulLinear (`SmaulRL`).

```bash
python rl.py --model_dir ./runs/linear --work_dir ./rl \
    --prompt "Explain gravity" --responses 8 --max_new_tokens 256 \
    --temperature 0.8 --top_k 50 --top_p 0.95 \
    --rl_lr 1e-6 --clip 0.2 --kl_coef 0.02 --device auto
```

## Flow

1. `candidates()` samples `count` (1-64) responses per prompt.
2. `_pick()` shows them; you choose the best (`Ctrl-D` keeps the first).
3. `_save()` appends `{prompt, responses, chosen, source: "human"}` to
   `work_dir/preferences.jsonl` (append + fsync).
4. `grpo_step()` trains: chosen gets reward +1, others -1.

## Guarantees

- Inputs validated: `max_new_tokens` in `[1, 4096]`, `count` in `[1, 64]`, `lr`/`clip`/`kl_coef`
  ranges checked; bad device strings fail fast.
- Prompts truncate to the 512-token window with a warning (same window as sampling);
  `_logprob()` uses the identical window so training matches generation, truncating long
  responses instead of OOMing.
- Single-candidate calls raise (std of 1 is NaN); empty-candidate runs raise instead of
  returning a fake `0.0`; non-finite loss raises **without** saving, so checkpoints are never
  poisoned.
- One `Lion` optimizer persists across `grpo_step()` calls (momentum is kept, not discarded
  per step).
- All-`-inf` sampling falls back to greedy instead of a `multinomial` crash.
- If `work_dir/policy` exists it resumes from there (a notice is printed); its tokenizer must
  match the model vocab or init raises.

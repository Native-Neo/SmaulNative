# rl.py --auto

Verified automatic RL: a small reward model (`PreferenceModel`) ranks candidates, GRPO trains
the policy, and the reward model retrains on each confirmed label.

```bash
python rl.py --auto --model_dir ./runs/linear --work_dir ./rl \
    --prompt "Explain gravity" --responses 8 --max_new_tokens 256 \
    --temperature 0.8 --top_k 50 --top_p 0.95 \
    --preference_epochs 3 --preference_lr 1e-3 \
    --rl_lr 1e-6 --clip 0.2 --kl_coef 0.02 --device auto
# add --no-verify only after collecting human prefs (see below)
```

## Safety rule

Unattended auto-labeling (`--no-verify`) is **refused** unless at least
`MIN_HUMAN_PREFS_FOR_AUTO` (4) human preference records exist and the reward model has
trained on them. Without this, the randomly-initialized reward model would label its own
`argmax` as ground truth and GRPO would collapse onto noise. Collect seeds first with
`rl.py`, or run with verification (default: you confirm/correct each pick).

## Training details

- `train_preferences()` streams `preferences.jsonl` (malformed lines skipped with a count),
  ignores `epochs <= 0` without touching the checkpoint, skips non-finite losses, and never
  marks a random model as trained (`valid == 0` leaves the checkpoint alone).
- Resume is hash-based: the meta file stores `{records, sha256}`; edited/reordered/truncated
  preference files retrain from scratch with a warning instead of training the wrong slice.
  Each round mixes a small deterministic replay sample with the new records to limit forgetting.
- Reward batches truncate pairs to `MAX_PREF_PAIR_LEN` (2048) so one huge prompt cannot OOM the
  dense batch tensor; empty inputs raise.
- Corrupt `preference_model.pt` checkpoints are ignored with a warning (metadata errors reset
  the counter); prompts/`EOF` on closed stdin keep the predicted choice.

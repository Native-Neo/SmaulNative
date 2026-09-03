# train.py

The one entrypoint for both pretraining and SFT -- a plain PyTorch loop (no Lightning/DeepSpeed),
with Ctrl-C checkpointing, full resume, and Hugging Face-style export.

## Run it

```bash
# pretrain from scratch (auto-trains a tokenizer if --tokenizer_path doesn't exist)
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M

# SFT on top of a pretrained checkpoint (point --output_dir at the SAME dir to continue in place,
# or a new dir to keep the base checkpoint untouched -- see notes below)
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT --tokenizer_path ./RWKV-X-256M/tokenizer.json
```

## Key flags

| Flag | Default | What it does |
|---|---|---|
| `--mode` | *(required)* | `pretrain` or `sft` |
| `--dataset_dir` | `./datasets` | training data (pretrain: any text; SFT: conversation JSON/JSONL) |
| `--output_dir` | `./RWKV-X-256M` | model checkpoint dir (`config.json`, `model.safetensors`, bundled `tokenizer.json`) |
| `--checkpoint_dir` | `./checkpoints` | optimizer state + resume position (separate from `--output_dir` on purpose) |
| `--tokenizer_path` | `./tokenizer.json` | auto-trained here if missing |
| `--tokenizer_vocab_size` | `32768` | only used if auto-training |
| `--n_embd` | `768` | hidden size (new model only) -- model size is fixed at 256M params (268,435,456, binary/Mebi convention; see below); this is the only size knob left |
| `--head_size` | `64` | must divide `--n_embd` |
| `--n_moba_layer` | `3` | how many blocks use MOBA sparse attention instead of pure RWKV-7 |
| `--ctx_len` | `512` | train-time BPTT window -- **keep this small on CPU** |
| `--batch_size` | `1` | |
| `--epochs` | `3` | SFT only |
| `--learning_rate` | `1e-4` | |
| `--optimizer` | `lion` | `lion` or `adamw` |
| `--log_every` | `10` | steps between log lines |
| `--save_every` | `200` | steps between checkpoint saves |
| `--new_data` | off | reset dataset position, **keep** model/optimizer weights |
| `--qat` | off | enable 3-bit QAT on the Channel-Mix linears (see [qat.md](qat.md)) |
| `--qat_calib_batches` | `64` | batches used to settle fake-quant ranges before training starts |
| `--qat_export_dir` | none | after training, convert+save an int3 checkpoint here (separate from `--output_dir`) |

## Notes

- **Resuming**: just re-run the same command with the same `--output_dir`/`--checkpoint_dir` --
  it picks up model weights, optimizer state, and exact dataset position automatically
  (`build_model` reloads from `config.json` when present; `ResumeState.load` restores the step /
  token count / file / record / buffer, `train.py:76`).
- **Ctrl-C is safe**: `SIGINT` sets a flag (`train.py:35`); the loop finishes the current step and
  the `finally:` block (`train.py:338`) saves before exiting -- including truncated-BPTT buffer
  tokens, so you never lose a partially-consumed document.
- **New dataset, same model**: add `--new_data` to reset the dataset read-position *without*
  losing trained weights (e.g. switching from `download.py` data to `syntheticdata.py` data
  mid-project). It resets `ResumeState` but keeps the optimizer (`train.py:320`, `:328`).
- **SFT after pretrain**: point `--tokenizer_path` at the pretrained checkpoint's bundled
  `tokenizer.json` (`<output_dir>/tokenizer.json`) so vocab stays consistent, and use a *different*
  `--output_dir`/`--checkpoint_dir` for the SFT run unless you deliberately want to overwrite the
  pretrained checkpoint in place. The checkpointer bundles the exact tokenizer used into
  `output_dir` (`train.py:127`) so later steps never guess which one goes with which model.
- **Model size is fixed, not a flag**: new models always target `TARGET_PARAMS = 268_435_456`
  (256M binary/Mebi convention) at whatever `--n_embd`/`--head_size`/`--n_moba_layer` you pass --
  depth (`n_layer`) is solved automatically via `config_for_target_params`. There's no
  `--target_params` flag anymore; edit `TARGET_PARAMS` in `train.py`, or call
  `rwkv_x_core.config_for_target_params(your_target, ...)` yourself (`train.py:30`).
- **Optimizer**: `lion` (default) keeps only one momentum buffer vs Adam's two, lighter on CPU
  RAM (`train.py:46`); `--optimizer adamw` is available. Gradients are clipped to norm 1.0, and a
  non-finite loss skips the step rather than poisoning the optimizer.
- **QAT**: `--qat` wraps the Channel-Mix FFN linears in fake-quant (see [qat.md](qat.md)),
  calibrates on `--qat_calib_batches`, trains normally, and if `--qat_export_dir` is set, converts
  a deep-copied model to real packed int3 weights and saves that as a separate checkpoint
  (`train.py:341`).
- CPU is genuinely slow here (the RWKV recurrence runs as a plain Python loop -- no compiled WKV
  kernel for CPU, `train.py:288`) -- start with a small `--ctx_len` to sanity-check your pipeline
  before a long run.

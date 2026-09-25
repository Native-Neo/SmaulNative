# train.py

`train.py` is the SmaulLinear FP8 pretraining entry point: pretraining with tiled E4M3
weights and an FP32 Lion optimizer, writing resume-free checkpoints.

## Run it

```bash
python train.py --data ./datasets --out ./runs/linear \
    --tokenizer ./runs/linear/tokenizer.json \
    --d 512 --layers 8 --heads 8 --vocab 8000 \
    --ctx 256 --batch 2 --steps 1000 --threads 2
```

| Flag | Default | What it does |
|---|---|---|
| `--data` | `./datasets` | dataset directory scanned by `dataset.discover_files` |
| `--out` | `./runs/linear` | checkpoint directory |
| `--tokenizer` | `<out>/tokenizer.json` | tokenizer path (default follows `--out`; auto-trained if missing/mismatched) |
| `--vocab` | `8000` | vocabulary size; must match the tokenizer |
| `--d` | `512` | model width (`d_model`) |
| `--layers` | `8` | block count (`n_layer`) |
| `--heads` | `8` | linear-attention head count |
| `--precision` | `fp8` | weight precision: tiled-E4M3 `fp8` or plain `fp32` |
| `--ctx` | `256` | training sequence length |
| `--batch` | `2` | sequences per optimizer step |
| `--steps` | `1000` | optimizer steps |
| `--lr` | `2e-4` | Lion learning rate |
| `--wd` | `0.01` | weight decay |
| `--grad_clip` | `1.0` | global grad-norm clip |
| `--log_every` | `10` | log cadence (steps) |
| `--save_every` | `200` | checkpoint cadence (steps) |
| `--tok_records` | `200000` | max records for automatic tokenizer training (`0` = unlimited) |
| `--threads` | `2` | CPU threads (via `compute.get_backend().configure()`) |

There are no SFT, streaming, or resume flags: the trainer only pretrains.

In `fp32` mode every projection is a plain FP32 linear (`fp8_modules` is empty) and
Lion runs its standard parameter path; checkpoints, inference, and GGUF export work
identically, with `.weight` tensors instead of packed `w8`/`sc` pairs.

## Tokenizer

The tokenizer is built automatically via `tokenizer.ensure_tokenizer`: an existing file is
reused only when its vocabulary size equals `--vocab` and its format version is current
(version 7, `tokenizer.VERSION`); otherwise it is rebuilt from `--data` (up to
`--tok_records` records) and saved to `--tokenizer`.

## Optimizer

`Lion` keeps FP32 momentum per parameter/FP8 module (betas `0.9`/`0.99`). Each step:

1. Clears parameter grads and per-module FP8 weight grads (`_gw`).
2. Rejects non-finite grads (they would poison quantized weights via `sign()`).
3. Clips the global grad norm (float64 accumulation) to `--grad_clip`.
4. Applies a sign update scaled by `--lr`, with decay `lr * wd` folded into the FP8
   `requant` for quantized layers and multiplicative decay for the rest.

Steps with non-finite loss or grads are skipped (50 consecutive failures stop training
instead of looping forever). `Ctrl-C` (`SIGINT`) or `SIGTERM` finishes the current step,
saves, and exits. A zero-step run does not overwrite any existing checkpoint.

## Checkpoints

Each save writes `model.safetensors` + `config.json` (now carrying `tokenizer_sha256` and
`dataset_fingerprint`, via `SmaulLinear.save_pretrained`),
the tokenizer, and `optimizer.json` holding only `{lr, wd, betas, clip}`. No optimizer momentum
or dataset position is stored, so every run trains forward from step 0 -- checkpoints are
restart points for inference/continued training setups, not exact training resume.

Progress lines report step, loss, tokens/sec, and stored parameter size in MiB (including
Lion momentum).

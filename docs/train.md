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
| `--tokenizer` | `./runs/linear/tokenizer.json` | tokenizer path (auto-trained if missing/mismatched) |
| `--vocab` | `8000` | vocabulary size; must match the tokenizer |
| `--d` | `512` | model width (`d_model`) |
| `--layers` | `8` | block count (`n_layer`) |
| `--heads` | `8` | linear-attention head count |
| `--ctx` | `256` | training sequence length |
| `--batch` | `2` | sequences per optimizer step |
| `--steps` | `1000` | optimizer steps |
| `--lr` | `2e-4` | Lion learning rate |
| `--wd` | `0.01` | weight decay |
| `--log_every` | `10` | log cadence (steps) |
| `--save_every` | `200` | checkpoint cadence (steps) |
| `--tok_records` | `200000` | reserved; currently unused |
| `--threads` | `2` | CPU threads (`torch` + `OMP_NUM_THREADS`) |

There are no SFT, streaming, precision, or resume flags: the trainer only pretrains.

## Tokenizer

The tokenizer is built automatically via `tokenizer.ensure_tokenizer`: an existing file is
reused only when its vocabulary size equals `--vocab` and its format version is current
(version 6); otherwise it is rebuilt from `--data` and saved to `--tokenizer`.

## Optimizer

`Lion` keeps FP32 momentum per parameter/FP8 module (betas `0.9`/`0.99`). Each step:

1. Clears parameter grads and per-module FP8 weight grads (`_gw`).
2. Clips the global grad norm to `1.0`.
3. Applies a sign update scaled by `--lr`, with decay `lr * wd` folded into the FP8
   `requant` for quantized layers and multiplicative decay for the rest.

Steps with non-finite loss are skipped. `Ctrl-C` (`SIGINT`) finishes the current step,
saves, and exits.

## Checkpoints

Each save writes `model.safetensors` + `config.json` (via `SmaulLinear.save_pretrained`),
the tokenizer, and `optimizer.pt` holding only `{lr, wd, betas}`. No optimizer momentum
or dataset position is stored, so every run trains forward from step 0 -- checkpoints are
restart points for inference/continued training setups, not exact training resume.

Progress lines report step, loss, tokens/sec, and stored parameter size in MiB.

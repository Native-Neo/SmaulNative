# train.py

`train.py` is the SmaulLinear FP8 pretraining entry point: pretraining with tiled E4M3
weights and an FP32 Lion (default) or SmaulOpt optimizer. Lion checkpoints are
resume-free; SmaulOpt checkpoints save full optimizer state and resume exactly.

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
| `--lr` | `2e-4` | learning rate (`learning_rate`) |
| `--wd` | `0.01` | weight decay (`weight_decay`) |
| `--optimizer` | `lion` | `lion` (default) or `smaul` (SmaulOpt v1) |
| `--beta-m` | `0.9` | SmaulOpt `beta_m` (momentum decay) |
| `--beta-v` | `0.999` | SmaulOpt `beta_v` (magnitude-EMA decay) |
| `--epsilon` | `1e-8` | SmaulOpt `epsilon` (must be positive) |
| `--state-dtype` | `bf16` | SmaulOpt state storage: `bf16` (2 B, default), `fp16` (2 B), `fp32` (4 B, lossless) |
| `--grad_clip` | `1.0` | global grad-norm clip |
| `--log_every` | `10` | log cadence (steps) |
| `--save_every` | `200` | checkpoint cadence (steps) |
| `--tok_records` | `200000` | max records for automatic tokenizer training (`0` = unlimited) |
| `--threads` | `2` | CPU threads (via `compute.get_backend().configure()`) |

There are no SFT, streaming, or resume flags: the trainer only pretrains.

In `fp32` mode every projection is a plain FP32 linear (`fp8_modules` is empty) and both
optimizers run their standard parameter path; checkpoints, inference, and GGUF export work
identically, with `.weight` tensors instead of packed `w8`/`sc` pairs.

## Tokenizer

The tokenizer is built automatically via `tokenizer.ensure_tokenizer`: an existing file is
reused only when its vocabulary size equals `--vocab` and its format version is current
(version 7, `tokenizer.VERSION`); otherwise it is rebuilt from `--data` (up to
`--tok_records` records) and saved to `--tokenizer`.

## Optimizer

Two optimizers are selectable with `--optimizer`; the default is unchanged.

### `lion` (default, resume-free)

`Lion` keeps FP32 momentum per parameter/FP8 module (betas `0.9`/`0.99`). Each step:

1. Clears parameter grads and per-module FP8 weight grads (`_gw`).
2. Rejects non-finite grads (they would poison quantized weights via `sign()`).
3. Clips the global grad norm (float64 accumulation) to `--grad_clip`.
4. Applies a sign update scaled by `--lr`, with decay `lr * wd` folded into the FP8
   `requant` for quantized layers and multiplicative decay for the rest.

### `smaul` (SmaulOpt v1)

`SmaulOpt` (`train.SmaulOpt`) is a small deterministic FP32 adaptive optimizer holding
exactly two persistent FP32 states per trainable parameter/FP8 module — `m` (momentum) and
`v` (EMA of `|g|`) — plus a global step counter for bias correction. For gradient `g_t`:

```
m_t     = beta_m * m_{t-1} + (1 - beta_m) * g_t
v_t     = beta_v * v_{t-1} + (1 - beta_v) * |g_t|
m_hat   = m_t / (1 - beta_m**t)
v_hat   = v_t / (1 - beta_v**t)
u_t     = m_hat / (v_hat + epsilon)
theta_t = theta_{t-1} - lr * u_t - lr * wd * theta_{t-1}   # decoupled decay
```

Defaults: `lr=1e-4`, `beta_m=0.9`, `beta_v=0.999`, `epsilon=1e-8`, `weight_decay=0.01`.
These are reproducible v1 defaults, not tuned or optimal values.

Notes:

- State arithmetic is FP32. SmaulOpt adds no clipping or normalization of its own: it
  consumes the same globally clipped gradients as Lion (the trainer's `--grad_clip` path).
- Non-finite grads are rejected and the step is skipped without advancing the counter, so
  bias correction stays aligned with the updates actually applied.
- FP8 weights are updated through the existing RQT path: state update in FP32, then
  `FP8Linear._requant_block` per 64-row block, so no FP32 master weight copy is introduced
  and no full-matrix update transient is built. `test_no_fp32_master_weights` still holds.
- Every trainable parameter (embeddings, norms, LM head, dense projections) uses the same
  update; no architecture-specific rules.
- SmaulOpt is fully resumable. See below.

#### State storage width (`--state-dtype`, v2)

`--state-dtype` narrows only the **stored** `m`/`v` buffers. The equations above are
always evaluated in FP32 for every setting — each step dequantizes the state to FP32, runs
the identical FP32 update, then requantizes. Weight storage (FP8 E4M3) is never affected.

| `--state-dtype` | bytes/element | storage | status |
|---|---|---|---|
| **`bf16`** | **2** | `bfloat16` | **default** — ~0.07% error vs `fp32` |
| `fp16` | 2 | `float16` | ~0.01% error; clips at 65504, flushes magnitudes below ~6e-8 |
| `fp32` | 4 | `float32` | lossless reference (v1 behavior) |

**Version boundary.** SmaulOpt **v1** specified FP32 state, and `--state-dtype fp32`
still reproduces it exactly. **v2** made the 2-byte `bf16` state the default: it halves
optimizer state at a measured ~0.07% relative parameter error against `fp32`, which is the
one deliberate deviation from the v1 spec. The update math, the equations, the FP8
integration, and the checkpoint format are otherwise unchanged.

For FP8 modules the state is widened to FP32 one 64-row block at a time (matching the
requant blocks), so no full-matrix FP32 state transient is introduced.

The 1-byte integer state (`int8`/`uint8` with per-block scales) was implemented, measured,
and **removed**. It was the wrong trade on every axis:

| | cost per 64×512 block | state | error vs `fp32` | weight blowup under a heavy tail |
|---|---|---|---|---|
| **`bf16`** | **336 µs** | 4 B/param | ~0.07% | none |
| `int8` | 1772 µs (**5.3x**) | 2 B/param | ~18% | `\|theta\|` 0.11 → 1467 |

Narrowing a float state is just a cast, whereas an integer state also needs an absmax
reduction, a scale, a divide, a round, a clamp, and a non-finite guard before the integer
cast — hence 5.3x the cost for only 2x less memory. It also had a real divergence mode: one
large-gradient element sets the block scale, every other element's `v` rounds to zero, and
`u = m_hat / (0 + epsilon)` ran to `|u| ~ 1.7e10`. Re-adding a 1-byte state would need
outlier handling, not just a wider integer.

#### Why the dequant/requant round trip stays

It is not a quantization error for `bf16`/`fp16` — widening to FP32 is exact. It is the
cast that lets the update be computed in FP32 at all, with one unavoidable rounding back to
storage. It also cannot be removed: `torch.lerp` *is* the exact fused EMA (`b*m + (1-b)*g`)
in one op, but it rejects dtype promotion, so it cannot read a `bf16` state and produce
`fp32`. Four alternatives measured on this repo, all worse than the current form:

| Implementation | Roundings | Accuracy | bf16 / fp16 (4096 elems) |
|---|---|---|---|
| dequant → `mul_`+`add_` → requant (current) | 1 | reference | **88 / 110 µs** |
| `lerp` after dequant (4 passes → 3) | 1 | bit-identical | 100 / 124 µs |
| delta form `m += (1-b)(g-m)` | 1 | bit-identical | 96 / 140 µs |
| in-place `mul_().add_()`, no round trip | 2 | 4x worse in `v` | fastest, 0 temps |

Narrowing buys memory, not speed. Note also that the error is dominated by `m`, not `v`:
`m` is the numerator, so its rounding error passes straight into the update, whereas `v` is
the denominator, where errors partly cancel between `m_hat` and `v_hat`.

#### `update_clip` (defensive guard)

In exact arithmetic `|EMA(g)| <= EMA(|g)|`, so `|u| < 1`. Narrow state can nudge `|u|` just
over that line, because `m_hat` and `v_hat` use different bias corrections early on, so
`update_clip` (default `10.0`) clamps marginally in normal operation. It is kept as a guard,
not a crutch: with it effectively disabled the narrow-float widths still track `fp32` and
never diverge (`test_update_clip_is_defensive_not_load_bearing`). The catastrophic mode it
would catch belonged to the removed integer path. `fp32` is never clamped, so the reference
path is bit-identical with or without it.

Checkpoints record `state_dtype` in `optimizer.json` and store `m`/`v` at that width, so a
resumed run keeps the same width. A checkpoint that names a removed width is rejected rather
than silently reinterpreted.

Steps with non-finite loss or grads are skipped (50 consecutive failures stop training
instead of looping forever). `Ctrl-C` (`SIGINT`) or `SIGTERM` finishes the current step,
saves, and exits. A zero-step run does not overwrite any existing checkpoint.

## Checkpoints

Each save writes `model.safetensors` + `config.json` (now carrying `tokenizer_sha256` and
`dataset_fingerprint`, via `SmaulLinear.save_pretrained`) and the tokenizer, plus
`optimizer.json`:

- **Lion**: `{name, lr, wd, betas, clip}` — no momentum, so Lion checkpoints are restart
  points for inference/continued training, not exact training resume.
- **SmaulOpt**: `optimizer.json` carries `{name, step, learning_rate, beta_m, beta_v,
  epsilon, weight_decay, clip, state_dtype}` and `optimizer_state.safetensors` carries the
  `m`/`v` tensors keyed `m.param.<name>` / `v.param.<name>` / `m.fp8.<module>` /
  `v.fp8.<module>`, stored at the checkpoint's own state width. Loading with `train._load_optimizer(dir, opt, model)`
  restores the step counter, states, scales, and hyperparameters, so training continues
  exactly where it stopped. Loading a non-SmaulOpt (or state-less) checkpoint into
  SmaulOpt fails with a clear error rather than pretending the `m`/`v` state exists. Lion
  checkpoints stay loadable as before.

Progress lines report step, loss, tokens/sec, and stored parameter size in MiB (including
optimizer state).

## Benchmark

`python benchmark.py --mode opt` compares the available optimizers on identical tensor
sizes, gradients, clipping, and thread count, reporting update time, persistent
optimizer-state bytes, and CPU throughput across `lion` and `smaul` at each
`--state-dtype`. It is a measurement, not a claim that SmaulOpt converges better than Lion.

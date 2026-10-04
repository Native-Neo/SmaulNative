# train.py

`train.py` is the SmaulLinear FP8 pretraining entry point: pretraining with tiled E4M3
weights and an FP32 SmaulOpt (default) or Lion optimizer. Lion checkpoints are
resume-free; SmaulOpt checkpoints save full optimizer state and resume exactly.

## Run it

```bash
python train.py --data ./datasets --out ./runs/linear \
    --tokenizer ./runs/linear/tokenizer.json \
    --d 512 --layers 8 --heads 8 --vocab 256 \
    --ctx 256 --batch 2 --steps 1000 --threads 2
```

Finite training:

```bash
python train.py --data ./datasets --out ./runs/finite \
    --d 512 --layers 8 --heads 8 --ctx 256 --batch 2 --steps 1000 \
    --moe-experts 4 --moe-top-k 2
```

Continual stream training with replay and retention measurement:

```bash
python train.py --data ./datasets/new-domain --old-data ./datasets/old-domain \
    --out ./runs/cont --continual --domains ./datasets/new-domain \
    --replay-size 512 --replay-rate 0.1 --trunk-lr-mult 0.3 \
    --d 512 --layers 8 --heads 8 --ctx 256 --batch 2 --steps 10000
```

Inference:

```bash
python infer_cli.py --model ./runs/linear --prompt "Hello world" --max 64
```

| Flag | Default | What it does |
|---|---|---|
| `--data` | `./datasets` | dataset directory scanned by `dataset.discover_files` |
| `--out` | `./runs/linear` | checkpoint directory |
| `--tokenizer` | `<out>/tokenizer.json` | tokenizer path (default follows `--out`; auto-trained if missing/mismatched) |
| `--preset` / `--list-presets` | none | named size preset, overriding `--vocab/--d/--layers/--heads/--ffn_mult`; `--list-presets` prints them and exits |
| `--vocab` | `256` | vocabulary size; byte-level, must be `256` |
| `--d` | `512` | model width (`d_model`) |
| `--layers` | `8` | block count (`n_layer`) |
| `--heads` | `8` | linear-attention head count |
| `--ffn_mult` | `2.5` | FFN width multiplier (also the MoE expert hidden-dim knob) |
| `--architecture` | `rawr` | `rawr` (sparse, default) or `plain` (dense baseline) |
| `--embedding-storage` | `ram` | embedding table: `ram` (`nn.Embedding`) or `mmap` (file-backed) |
| `--rawr-sparsity` | `0.9` | Rawr: fraction of connections omitted; see the warning below |
| `--rawr-min-degree` | `4` | Rawr: connectivity floor per token (>= 1) |
| `--rawr-dict` | none | Rawr: extra dictionary file (one word per line) on top of built-ins |
| `--rawr-graph-out` | none | Rawr: also export the connectivity graph JSON here |
| `--rawr-max-docs` | `2000` | Rawr: max corpus docs sampled for graph edges (`0` = unlimited) |
| `--rawr-max-tokens-per-doc` | `1024` | Rawr: max tokens read per corpus doc for graph edges |
| `--precision` | `fp8` | weight precision: tiled-E4M3 `fp8` or plain `fp32` |
| `--moe-experts` | `1` | total MoE experts per block (`1` = dense FFN, no routing) |
| `--moe-top-k` | `1` | active experts per token (only these execute) |
| `--moe-balance-weight` | `0.01` | router load-balancing aux-loss weight |
| `--continual` | off | continual stream training with replay and per-group LRs |
| `--domains` | `--data` | comma-separated dataset dirs, trained in order without auto state reset |
| `--old-data` | none | reference domain for retention eval (old loss before/after) |
| `--replay-size` | `512` | bounded replay reservoir, in chunks (`0` disables storage) |
| `--replay-rate` | `0.1` | fraction of continual steps interleaved from replay, `[0, 1]` |
| `--trunk-lr-mult` | `0.3` | shared-trunk LR multiplier in continual mode (conservative) |
| `--expert-lr` | `--lr` | MoE expert LR in continual mode |
| `--router-lr` | `--lr` | router LR in continual mode |
| `--reset-state` | off | reset the stream carry-over buffer between domains |
| `--retention-batches` | `20` | eval batches per domain for retention reports |
| `--ctx` | `256` | training sequence length |
| `--batch` | `2` | sequences per optimizer step |
| `--steps` | `1000` | optimizer steps |
| `--lr` | `2e-4` | learning rate (`learning_rate`) |
| `--wd` | `0.01` | weight decay (`weight_decay`) |
| `--optimizer` | `smaul` | `smaul` (default, SmaulOpt) or `lion` (resume-free, fixed betas) |
| `--beta-m` | `0.9` | SmaulOpt `beta_m` (momentum decay) |
| `--beta-v` | `0.999` | SmaulOpt `beta_v` (magnitude-EMA decay) |
| `--epsilon` | `1e-8` | SmaulOpt `epsilon` (must be positive) |
| `--state-dtype` | `bf16` | SmaulOpt state storage: `bf16` (2 B, default), `fp16` (2 B), `fp32` (4 B, lossless) |
| `--factor-v` / `--no-factor-v` | on | SmaulOpt: store `v` factored (row/col marginals) for 2-D parameters |
| `--grad-dtype` | `bf16` | SmaulOpt: dtype `p.grad` is stored at after `backward()`; `fp32` keeps them wide. Lion ignores it. |
| `--grad_clip` | `1.0` | global grad-norm clip |
| `--log_every` | `10` | log cadence (steps) |
| `--save_every` | `200` | checkpoint cadence (steps) |
| `--threads` | `2` | CPU threads (via `compute.get_backend().configure()`) |

There are no SFT or resume flags: the trainer pretrains (finite) or runs the
continual stream described below.

In `fp32` mode every projection is a plain FP32 linear (`fp8_modules` is empty) and both
optimizers run their standard parameter path; checkpoints, inference, and GGUF export work
identically, with `.weight` tensors instead of packed `w8`/`sc` pairs.

## `--rawr-sparsity` is a column-count knob, not a compression ratio

`hidden_cols` keeps `K = d_model * (1 - rawr_sparsity)` columns per `SparseLinear` row, so
`K` -- not the sparsity fraction -- is what sets the cost. `SparseLinear` now dispatches its
forward and `d/dx` as CSR sparse products (`_SparseLinearFn`), so it visits exactly the
`out_f * K` nonzeros and allocates only the output; the earlier `x[..., cols]` gather
materialised a `[B, T, out_f, K]` block and was measured **23x slower than the dense FP32
GEMM it replaced**, despite doing 10x less arithmetic. At the 32M preset that gather was 41%
of a 35.5 s step (8.3 tok/s).

The graph-derived part of the sparsity is real, but the headline number is not the model's.
`RawrGraph.stats()` reports the fraction of the `vocab x vocab` token-edge space the graph
occupies; the graph only drives the **FFN and the LM head**. The attention q/k/v/o
projections stay dense `FP8Linear`, and at the 32M preset they dominate. So `train.py` prints
both: `SmaulLinear.compute_profile()` walks the built model and reports what is actually
executed. V8000, d512, 8 layers, ffn 2.5, batch 2, ctx 256, 2 threads:

| `--rawr-sparsity` | K | model sparsity | dense share of per-token MAC | graph reports | full step | tok/s |
|---|---|---|---|---|---|---|
| `plain` (dense) | -- | 0.0% | 100% | 99.91% | 10726 ms | 47.7 |
| `0.5` | 256 | 35.1% | 45.8% | 99.91% | 9487 ms | 54.0 |
| **`0.9` (default)** | **51** | **63.3%** | **80.9%** | 99.91% | **5244 ms** | **97.6** |
| `0.99` | 5 | 69.6% | 97.7% | 99.91% | 3619 ms | 141.5 |

Read that as: the 99.91% "sparsity" buys 2.2x over dense, because 81-98% of the arithmetic
is dense attention regardless. Going from `0.5` to `0.9` is the change that mattered -- at
`0.5` the "sparse" model stored *more* trainable values than the dense one (14.0M vs 8.2M)
while doing only half the compute, i.e. the worst of both.

`LinearConfig` keeps `rawr_sparsity=0.5` as its code-level default so library callers and
legacy checkpoints (which record the value they were built with) are unaffected -- same
pattern as `architecture` defaulting to `plain` in code but `rawr` on the CLI.

Deriving the columns used to cost 20.6 s per 8-layer model build (160 s at the 256M preset):
`hidden_cols` scored all `in_f` columns per row and sorted. It now takes only the answer
(adjacency + the nearest non-edges, merged), which is **15-22x faster** and produces
byte-identical columns -- which matters, because `cols` is what every existing Rawr
checkpoint is interpreted through. `tests/test_rawr.py` pins it against a verbatim copy of
the old implementation.

## Where the time goes

Measured with `torch.profiler` and component timings, 32M preset, batch 2, ctx 256, 2 threads,
`--rawr-sparsity 0.99` (4005 ms/step, 128 tok/s):

| phase | share |
|---|---|
| linear attention forward+backward (dense FP8 q/k/v/o + the O(D^2) recurrence) | **~44%** |
| FP8 requantization in the optimizer step (`decode_tile` + `quantize_tiles`, 256 calls) | ~16% |
| everything else in the optimizer step (dense embedding update, SparseLinear values) | ~14% |
| Rawr FFN + LM head, sparse | ~7% |
| logits + cross-entropy | ~1% |
| global grad norm | ~0.02% |

The Rawr sparse layers are now ~1% of the step; the dense FP8 attention is the cost. On this
CPU the FP8 kernels are *slower* than a plain MKL FP32 GEMM of the same shape
(`benchmark.py` measures 1.92x forward and 2.20x backward), because AVX1 without FMA
sustains 7-13 GFLOP/s where MKL SGEMM sustains 39-48. Memory is not the constraint at this
scale: peak RSS 615 MB, live gradients 32.5 MiB, checkpoint 25.0 MiB, optimizer `m` 24.2 MiB,
`cols` index RAM 1.5 MiB.

## Tokenizer
The tokenizer is a fixed byte codec written automatically via
`tokenizer.ensure_tokenizer`: an existing valid byte file is reused, a
legacy/unreadable one is rewritten. No training data is needed for the codec
itself. `--vocab` must be `256`.

## Continual stream training

`--continual` trains indefinitely over `--domains` (default: `--data` alone) as
one byte stream: the carry-over buffer threads through domains with no
auto-reset on domain change (`--reset-state` clears it explicitly between
domains). Every new chunk enters a bounded replay reservoir (`--replay-size`
chunks, uniform reservoir sampling, constant memory); a `--replay-rate`
fraction of steps is interleaved from replay so new data never fully replaces
old data and old domains keep exercising their experts.

Three optimizer groups keep the shared RAWR/recurrent trunk comparatively
stable while experts adapt faster: trunk LR is `--lr * --trunk-lr-mult`
(default `0.3 * --lr`), experts use `--expert-lr`, the router `--router-lr`
(both default to `--lr`). Each group carries its own momentum state; FP8
weight grads are isolated per group while it steps (`continual.GwStash`), so
no optimizer change was needed. Step lines log `moe_inactive` (experts with no
tokens) and `max_use` (largest expert usage fraction) when MoE is on.

Retention is measured, not claimed: with `--old-data`, the run evaluates
old-domain loss before training, then old-domain and new-domain loss after,
and prints `[retention] old a -> b (delta +d) new c`. A positive delta is
forgetting; replay, the conservative trunk rate, and expert routing are what
reduce it. Catastrophic forgetting is not claimed solved.

## MoE training

`--moe-experts N --moe-top-k K` builds `SwiFFN_MoE` blocks with `RawrFFN`
experts on `architecture="rawr"` (dense `SwiFFN` experts on `"plain"`).
`--moe-balance-weight` scales the Switch-style load-balancing aux loss added
to the training loss. `merge_moe.py` can additionally upcycle per-domain dense
checkpoints into MoE experts; see `merge_moe.md`.

## Optimizer

Two optimizers are selectable with `--optimizer`. The default is `smaul`.

### `lion` (resume-free)

`Lion` keeps FP32 momentum per parameter/FP8 module (betas `0.9`/`0.99`). Each step:

1. Clears parameter grads and per-module FP8 weight grads (`_gw`).
2. Rejects non-finite grads (they would poison quantized weights via `sign()`).
3. Clips the global grad norm to `--grad_clip`. The norm is accumulated in **float64**, in
   `_NORM_BLOCK` (2^19) element blocks so no full-size float32/float64 copy of any gradient
   is ever built -- the old form built an FP32 copy of every gradient and held them all,
   ~10x the gradient bytes in transients (314 MiB for one 31 MiB bf16 gradient). float32 is
   *not* a drop-in here: `torch.linalg.vector_norm` accumulates linearly, so on a 16M-element
   tensor its float32 result is 6.4e-4 relative off (bf16 3.5e-4) -- and this number *is* the
   clip threshold. Measured 0.196 s / 314 MiB -> 0.081 s / 10.7 MiB for one 8000x2048 bf16
   gradient, at 1.1e-16 relative error. A non-finite gradient propagates through the norm, so
   `_clip` needs no separate `isfinite` scan over every gradient (that scan was 0.130 s,
   149% of the norm computation itself, because it re-reads all the bytes).
4. Applies a sign update scaled by `--lr`, with decay `lr * wd` folded into the FP8
   `requant` for quantized layers and multiplicative decay for the rest.

### `smaul` (SmaulOpt, default)

`SmaulOpt` (`model.SmaulOpt`) is a small deterministic FP32 adaptive optimizer holding
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

#### Factored `v` (`--factor-v`, on by default)

For a 2-D state `[R, C]`, the full `v` is replaced by two BF16 vectors `v_row[R]` and
`v_col[C]`. `m` is always full-size, and the update equations are unchanged.

**The equation**

```
v_hat[i, j]  ~=  R_hat[i] * C_hat[j] / G_hat        G_hat = mean(R_hat)
```

**Why that form.** The EMA is linear, so it commutes with means:

```
rowmean_i(v_t) = beta_v * rowmean_i(v_{t-1}) + (1 - beta_v) * rowmean_i(|g_t|)
```

So maintaining an EMA of the row means and of the column means gives the **exact** row and
column marginals of the true `v_t` -- no approximation at that stage. The only approximation
is dropping the rank/interaction term. Given exact marginals `R`, `C` and grand mean
`G = mean(R) = mean(C)`, the unique rank-1 field consistent with both marginals and the
grand mean is exactly `R_i C_j / G`.

This is **not** AdaFactor's form. AdaFactor factors `EMA(g^2)` as `R_i * C_j` with no
division, because it only ever needs the result under a `sqrt`. SmaulOpt's statistic is
`EMA(|g|)` -- non-negative, and used directly as a denominator -- so the division is what
keeps the reconstruction consistent with the stored marginals. Measured on random gradients,
dividing roughly halves the reconstruction error: 0.30 relative versus 0.64 un-divided, and
the reconstruction is exact (2e-7) for a rank-1 `v`.

**Which tensors get factored** -- by shape only, no module or architecture names:

| shape | form | reason |
|---|---|---|
| `[R, C]`, `R,C >= 2` | factored | `R + C <= R * C`, so it is never larger |
| `[C]`, `[]` | full `v` | cannot be row/column factored |
| `[1, C]`, `[R, 1]` | full `v` | `R + C > R * C` |

**Memory.** For a 4096x4096 state, `v` goes from `R*C*2` = 32 MB to `(R+C)*2` = 16 KB.
Since `m` is full-size, the *total* optimizer state shrinks by less: on a 128-wide 2-layer
model, bf16 total state goes 3.40 MiB -> 1.72 MiB (1.98x), with `v` itself 1.70 MiB ->
0.02 MiB (~85x). The saving grows with model size, since the largest tensors (embedding and
LM head) are exactly the 2-D ones.

**Cost.** No full `[R, C] v` is ever allocated. Marginal means are computed with an L1
reduction (`torch.linalg.vector_norm(..., ord=1, dim)`), which is a fused reduction -- the
profiler shows 0.000 MiB allocated versus a full-size temporary for `g.abs().mean(dim)`. The
reconstruction is materialized one 64-row block at a time, matching the existing requant
blocking. Isolated `opt.step()` latency is unchanged within noise: 1.03x (bf16) and 0.95x
(fp32) versus full `v`.

**Accuracy: the measured tradeoff.** Factored `v` is an approximation and is *not* equivalent
to full `v`. Relative error of factored versus full, after 60 steps on a 24x40 grid:

| gradient distribution | `v` rel err | update rel err | parameter rel err |
|---|---|---|---|
| rank-1 (separable) | 0.0000 | 0.0000 | 0.0000 |
| heavy-tailed | 0.043 | 0.017 | 0.040 |
| uniform | 0.126 | 0.050 | 0.121 |
| unbalanced columns | 0.111 | 0.052 | 0.126 |
| unbalanced rows | 0.119 | 0.054 | 0.131 |
| normal | 0.167 | 0.066 | 0.159 |
| **sparse (10% nonzero)** | 0.542 | 0.211 | **0.458** |
| **mostly-zero (0.5% nonzero)** | 0.922 | 0.938 | **1.436** |

The update error runs ~2.5-3x *below* the `v` error on dense grids, because `u` is
scale-invariant in `v` and the marginals are preserved exactly. But **a rank-1 field is
dense, so sparse `v` is the weak spot**: on a 10%-nonzero gradient the reconstruction puts
`v` about 1.8x too *small* at the positions that actually have gradient, which makes the step
about 1.8x too large there, and smears non-zero mass into positions whose true `v` is zero.
On a 0.5%-nonzero gradient the parameter error exceeds 1.0. Since LLM gradients are often
sparse -- and the tensors that dominate memory here are the embedding and head -- this is a
real risk, not a corner case. AdaFactor, for the same reason, does not factor embedding
matrices; this implementation factors by shape only, so it *will* factor them. If sparse
gradients turn out to dominate a real run, `--no-factor-v` is the switch back, and excluding
particular tensors is a shape/threshold decision rather than a name-based rule.

**Checkpoints.** `factor_v` is recorded in `optimizer.json`. Factored states are stored under
`v_row.*` / `v_col.*` keys, full states under `v.*`. A model holding both forms at once (2-D
plus 1-D parameters) round-trips normally. Loading rules:

- factored checkpoint + `factor_v` on -> loads the marginals.
- factored checkpoint + `--no-factor-v` -> **refused**; a factored state cannot be expanded
  into a full one without inventing the rank term.
- full checkpoint + `factor_v` on -> **migrated explicitly**, with a printed notice. The
  marginals of a stored `v` are exactly recoverable, so `R` and `C` are preserved exactly;
  only the rank term is dropped, which is what factoring approximates anyway.
- checkpoint with no `factor_v` key (written before factoring existed) -> treated as full-v.

Nothing is ever silently reinterpreted or discarded.

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

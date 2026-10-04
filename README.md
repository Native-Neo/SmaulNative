# SmaulNative

A compact SmaulLinear training and inference repository with English-Hindi data tooling, Mixture of Experts (MoE) upcycling, real quantized FP8 training, and native CPU acceleration.

## Overview

- **SmaulLinear Architecture**: Linear-attention blocks with SwiGLU FFN/MoE and tiled E4M3 FP8 weights (`model.py`, `kernel/fp8_tile.py`).
- **Native CPU Backend**: Tiled FP8 kernels with a torch fallback plus a training-step benchmark (`kernel/compute.py`, `kernel/fp8_tile.py`, `kernel/fp8_cpu.cpp`, `kernel/attn_cpu.cpp`, `kernel/sparse_cpu.cpp`, `kernel/quant_cpu.cpp`, `benchmark.py`).
- **Bilingual Tokenizer**: A custom word/character tokenizer with Devanagari grapheme fallback, case markers, and special tokens (`tokenizer.py`).
- **Unified Training Pipeline**: `train.py` pretrains SmaulLinear with tiled FP8 weights, an FP32 Lion or SmaulOpt optimizer (`--optimizer smaul`, default), automatic tokenizer builds, and resume-free (Lion) / resumable (SmaulOpt) checkpoints.
- **Incremental decoding**: `inference.py` decodes one token at a time from the linear-attention state instead of re-running the prompt per token (2.4x-21.5x for generation; see [docs/inference.md](docs/inference.md)).
- **MoE Upcycling**: Merge multiple dense SmaulLinear checkpoints into a sparse SwiGLU Mixture of Experts model (`merge_moe.py`).
- **RQT**: Real Quantized Training with tiled E4M3 FP8 weights and per-tile scales, requantized in place after every optimizer step. No FP32 master copy of an FP8 weight is kept; optimizer momentum stays FP32.

## Project Layout

```
├── kernel/                # Compute backends and native CPU kernels
│   ├── compute.py
│   ├── fp8_tile.py
│   ├── fp8_cpu.cpp
│   ├── attn_cpu.cpp
│   ├── sparse_cpu.cpp
│   └── quant_cpu.cpp
├── benchmark.py           # CPU benchmark suite
├── docs/                  # Detailed guides and command references
├── tests/                 # Unit and optimization regression tests
├── USEME.md               # CLI cheat sheet
├── model.py               # SmaulLinear model definition, presets, optimizers, build_model()
├── train.py               # SmaulLinear FP8 trainer
├── inference.py           # Inference engine for SmaulLinear checkpoints
├── infer_cli.py           # Interactive chat CLI
├── infer_server.py        # Local server with chat UI
├── merge_moe.py           # Dense-to-MoE upcycling
├── convert_linear_to_gguf.py  # Checkpoint export to GGUF
├── rl.py                  # Human preference collection + GRPO training
├── dataset.py             # Dataset loaders
├── download.py            # Dataset downloader
├── filter_data.py         # Lightweight filters for streamed records
├── stream_data.py         # Remote Hugging Face Parquet streaming
├── syntheticdata.py       # Synthetic bilingual data generator
└── tokenizer.py           # Custom bilingual tokenizer
```

## Quickstart

### 1. Installation

```bash
python -m pip install -r requirements.txt
```

For the native CPU FP8 extension, `ninja` and a working C++ compiler are also required.
Without them training still works through the torch fallback.

### 2. End-to-End Workflow

```bash
# 1. Download or generate data
python download.py --languages hindi english --max_rows 100000
# or: python syntheticdata.py --count 250000 --format both --output-dir ./datasets

# 2. Byte codec (no training step -- fixed 256 entries, written automatically)
#    Inspect it any time:
python tokenizer.py encode --text "Hello नमस्ते"

# 3. Pretraining (finite)
python train.py --data ./datasets --out ./runs/linear \
    --d 512 --layers 8 --heads 8 --vocab 256 \
    --ctx 256 --batch 2 --steps 1000 --threads 2

# 4. Continual stream training with replay (see docs/train.md)
python train.py --data ./datasets/new --old-data ./datasets/old \
    --out ./runs/cont --continual --replay-size 512 --replay-rate 0.1
```

The byte `tokenizer.json` is written automatically; a legacy word-level file is
refused with a clear error instead of being silently reused.

## RQT Training

RQT means **Real Quantized Training**: the model trains from the quantized
representation rather than a fake-quantized view of an FP32 master parameter.

Every `FP8Linear` stores `uint8` E4M3 codes plus `float32` per-tile scales (tile width
64 by default). The forward pass runs through the compute backend -- natively in AVX
when the extension is available, otherwise via a bounded torch fallback that never
materializes the full FP32 matrix. Weight gradients accumulate in a bf16 `_gw` buffer and
are folded back into the quantized storage by `requant` after every optimizer step, so the
packed weights themselves carry the training forward.

See [docs/rqt.md](docs/rqt.md) for the storage format, the training step, and GGUF export.

## CPU Training

The CPU backend is the default compute backend (`kernel.compute.get_backend()`). Threading is
configured explicitly or via `SMAUL_CPU_THREADS`:

```bash
SMAUL_CPU_THREADS=2 python train.py --data ./datasets --out ./runs/linear
```

`configure()` sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and the torch thread counts. The
native extension (`smaul_fp8_ivb`, built from `fp8_cpu.cpp` for Ivy Bridge-era CPUs)
loads lazily; if the build fails, a warning is issued once and the torch tiled fallback
takes over. Do not copy a built extension between different CPU architectures. Measure
before tuning -- see `python benchmark.py --help`.

## Configuration

The default training configuration is `--d 512 --layers 8 --heads 8 --vocab 256` with
`--ctx 256 --batch 2`. `--d` controls width, `--layers` controls depth, `--vocab` is
always the 256 byte values, and `d_model` must be divisible by `n_heads`. `LinearConfig` also
exposes `ffn_mult` (default `2.5`, also the MoE expert hidden-dim knob), `tile` (default `64`), `precision` (`fp8` or
`fp32`, default `fp8`), and the MoE fields
`is_moe`/`num_experts`/`num_experts_per_tok`/`moe_balance_weight` (set via
`--moe-experts`/`--moe-top-k`/`--moe-balance-weight` or `merge_moe.py`).

For CPU training, start with a small `--ctx`: the linear-attention state is compact,
but longer contexts still cost more per step.

## Remote Streaming

`train.py --continual` trains as a stream over `--domains` with replay (see
docs/train.md). To use remote data without downloading a full
dataset first, stream filtered records to stdout and consume them downstream:

```bash
python stream_data.py --dataset hindi --max_records 1000 > streamed.jsonl
```

`--dataset` accepts `hindi`, `english`, `openthoughts`, or `all`. Records are filtered
by `filter_data` rules (`--min_chars`, `--max_chars`); authentication uses `HF_TOKEN` /
`HUGGINGFACE_HUB_TOKEN`, and library callers can resume from a dataset/file/row
position via `start_dataset` / `start_file` / `start_record`.

## Resume

Lion (`--optimizer lion`) checkpoints are resume-free: each save writes
`model.safetensors` + `config.json` (with `tokenizer_sha256` + `dataset_fingerprint`), the
tokenizer, and `optimizer.json` holding only the Lion hyperparameters (`lr`, `wd`, betas,
`clip`). No optimizer momentum, RNG state, or dataset position is stored, so re-running a
Lion training command always starts from step 0.

SmaulOpt (`--optimizer smaul`, the default) checkpoints are resumable: `optimizer.json` additionally
records the step counter, `beta_m`, `beta_v`, `epsilon`, weight decay, and
`state_dtype`, and `optimizer_state.safetensors` stores the `m`/`v` states at that width,
so `train._load_optimizer(dir, opt, model)` continues training from the saved step.
`--state-dtype {bf16,fp16,fp32}` controls the optimizer's own state width while the update
math stays FP32: `bf16` (2 B/param) is the **default**, costing a measured ~0.07% error
against `fp32`, and `fp32` (4 B/param) reproduces the v1 lossless behavior. A 1-byte
integer state was tried and removed: ~5x slower per step, ~18% error, and it diverged on
heavy-tailed gradients. `--no-factor-v` disables SmaulOpt's factored `v` (row/column
marginals instead of a full `[R, C]` state, on by default for 2-D parameters): `v` shrinks
~85x with no measurable step-latency cost, at the cost of an approximation that is exact for
rank-1 gradients and weakest on sparse ones.

## Testing

Run the full suite (955 tests) with:

```bash
python -m pytest -q
```

This is the same command CI runs (`.github/workflows/test.yml`). For the FP8
training-step benchmark instead of the test suite:

```bash
python benchmark.py --d 512 --layers 4 --ctx 256 --batch 2 --iters 10
```

# SmaulNative

A compact RWKV-X training and inference repository with English-Hindi data tooling, Mixture of Experts (MoE) upcycling, low-bit floating-point QAT, and native CPU acceleration.

## Overview

- **RWKV-X Architecture**: RWKV-7 TimeMix blocks with interleaved MOBA attention (`rwkv_x_core.py`).
- **Native CPU Backend**: C++ WKV forward/backward kernels plus CPU-specific training support (`cpu/`).
- **Bilingual Tokenizer**: A custom word/character tokenizer with Devanagari grapheme fallback, case markers, and special tokens (`tokenizer.py`).
- **Unified Training Pipeline**: `train.py` supports pretraining, SFT, streaming resume, QAT, and router-only MoE fine-tuning.
- **MoE Upcycling**: Merge multiple dense domain checkpoints into a sparse Mixture of Experts model (`merge_moe.py`).
- **Low-bit floating-point QAT**: `qat.py` supports FP2, FP4, and FP8 modes. FP2/FP4 weights can be physically packed for export and CPU inference; QAT keeps FP32 master weights for training stability.

## Project Layout

```
├── cpu/               # Native WKV kernels and CPU benchmarks
├── datasets/          # Local training datasets and token streams
├── docs/              # Detailed guides and command references
├── tests/             # Unit and optimization regression tests
├── USEME.md           # CLI cheat sheet
├── dataset.py         # Dataset loaders
├── download.py        # Dataset downloader
├── merge_moe.py       # Dense-to-MoE upcycling
├── qat.py             # Low-bit floating-point QAT and packed weights
├── rwkv_x_core.py     # Core RWKV-X model definition
├── stream_data.py     # Remote Hugging Face Parquet streaming
├── syntheticdata.py   # Synthetic bilingual data generator
├── tokenizer.py       # Custom bilingual tokenizer
└── train.py           # Pretraining, SFT, and MoE router training
```

## Quickstart

### 1. Installation

```bash
pip install torch safetensors huggingface_hub pyarrow tqdm
```

For native CPU acceleration, a working C++ compiler and the Python packages used by `cpu/__init__.py` are required.

### 2. End-to-End Workflow

```bash
# 1. Download or generate data
python download.py
# or: python syntheticdata.py --output_file datasets/synthetic.jsonl

# 2. Train a tokenizer
python tokenizer.py train --fromdataset ./datasets --output ./SmaulNative/tokenizer.json --vocab-size 32768

# 3. Pretraining
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --ctx_len 256 \
    --tokenizer_path ./SmaulNative/tokenizer.json

# 4. SFT + low-bit floating-point QAT
python train.py --mode sft --dataset_dir ./datasets --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json --qt 4 --qat_export_dir ./SmaulNative-fp4
```

`--qt 2`, `--qt 4`, and `--qt 8` select FP2, FP4, and FP8 respectively. The `--qt` compatibility option is translated by `qat.py`; `train.py --qat` remains available as the generic QAT switch.

If `tokenizer.json` is absent, `train.py` can train it automatically. Supplying an existing tokenizer is recommended for reproducible training and resume runs.

### 3. CPU Training

The CPU backend uses the native WKV implementation automatically when `--cpu` is enabled:

```bash
SMAUL_CPU_THREADS=2 python train.py --cpu --mode pretrain --dataset_dir ./datasets \
    --output_dir ./SmaulNative --optimizer lion
```

The native extension is compiled for the host CPU with `-march=native`; do not copy a built extension between different CPU architectures. On an i3-3220, 2 threads are recommended over all 4 hardware threads because Hyper-Threading can reduce throughput for this workload.

## Configuration

The default model configuration is approximately 256M parameters for a 65K vocabulary. `--n_embd` controls width, `--n_layer` controls depth, and `--n_moba_layer` controls the number of MOBA blocks. `--head_size` must divide `--n_embd`, and at least one RWKV block must remain.

For CPU training, start with a small `--ctx_len`. MOBA attention uses causal scaled-dot-product attention and becomes increasingly expensive as sequence length grows.

## Remote Streaming

`--stream_dataset` can use `hindi`, `english`, `openthoughts`, or `all` to stream filtered Parquet records directly from Hugging Face without downloading the dataset first. Streaming checkpoints preserve the dataset/file/row position and the partially filled token buffer.

## Resume and QAT

Training checkpoints preserve model weights, optimizer state, RNG state, dataset position, token count, and streaming buffer state. Re-running the same training command resumes from the saved checkpoint. `--new_data` resets the dataset position while keeping the model and optimizer state.

QAT supports FP2, FP4, and FP8 floating-point modes. FP2 and FP4 use physically packed sub-byte weight storage when converted/exported. QAT training retains FP32 master parameters and uses fake quantization in the forward pass; packed FP2/FP4 weights are intended for the converted/exported inference path. FP8 uses PyTorch's `float8_e4m3fn` representation where supported.

For FP2/FP4 CPU inference, the packed path processes weights in chunks to avoid materializing the complete low-bit weight matrix as FP32 at once. This reduces temporary memory use, but it is not a hardware FP2/FP4 GEMM kernel; performance should be benchmarked against the normal FP32 path.

## Testing

Run the optimization and model tests with:

```bash
python3 tests/test_optimizations.py
```

The native WKV regression test is:

```bash
python3 tests/test_wkv_native.py
```

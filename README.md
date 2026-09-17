# SmaulNative

A compact RWKV-X training and inference repository with English-Hindi data tooling, Mixture of Experts (MoE) upcycling, real packed low-bit RQT, and native CPU acceleration.

## Overview

- **RWKV-X Architecture**: RWKV-7 TimeMix blocks with interleaved MOBA attention (`rwkv_x_core.py`).
- **Native CPU Backend**: C++ WKV forward/backward kernels plus CPU-specific training support (`cpu/`).
- **Bilingual Tokenizer**: A custom word/character tokenizer with Devanagari grapheme fallback, case markers, and special tokens (`tokenizer.py`).
- **Unified Training Pipeline**: `train.py` supports pretraining, SFT, streaming resume, RQT, and router-only MoE fine-tuning.
- **MoE Upcycling**: Merge multiple dense domain checkpoints into a sparse Mixture of Experts model (`merge_moe.py`).
- **RQT**: Real Quantized Training with physically packed FP4/FP6 weights and native FP8 storage. RQT does not keep FP32 master weights for RQT linear layers.

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
├── rqt.py             # Real packed FP4/FP6/FP8 training
├── rwkv_x_core.py     # Core RWKV-X model definition
├── stream_data.py     # Remote Hugging Face Parquet streaming
├── syntheticdata.py   # Synthetic bilingual data generator
├── tokenizer.py       # Custom bilingual tokenizer
└── train.py           # Pretraining, SFT, RQT, and MoE router training
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

# 3. Normal pretraining
python train.py --cpu --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --ctx_len 256 \
    --tokenizer_path ./SmaulNative/tokenizer.json
```

If `tokenizer.json` is absent, `train.py` can train it automatically. Supplying an existing tokenizer is recommended for reproducible training and resume runs.

## RQT Training

RQT means **Real Quantized Training**: the model trains from the physically quantized representation rather than a fake-quantized view of an FP32 master parameter.

```bash
# Pure FP4
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 4

# Pure FP6
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 6

# Pure FP8
python train.py --cpu --mode pretrain --dataset_dir ./datasets --rqt --rqt_bits 8

# Mixed FP4/FP6/FP8
python train.py --cpu --mode pretrain --dataset_dir ./datasets --mixed_rqt
```

Pure RQT applies the selected precision to every `nn.Linear`. Mixed RQT uses FP8 for attention/head projections, FP4 for FFN projections, and FP6 for the remaining linear layers.

RQT packs two FP4 codes per byte and four FP6 codes per three bytes. FP8 uses PyTorch `float8_e4m3fn`. RQT Lion keeps its optimizer averages in FP32, while the model weights themselves remain packed after every optimizer step.

The current implementation decodes the packed weights to FP32 for the matrix multiplication. This gives genuine packed model storage and genuine quantized forward weights, but it is not yet a dedicated FP4/FP6 CPU GEMM kernel.

See [docs/rqt.md](docs/rqt.md) for the implementation details and resume format.

## CPU Training

The CPU backend uses the native WKV implementation automatically when `--cpu` is enabled:

```bash
SMAUL_CPU_THREADS=2 python train.py --cpu --mode pretrain --dataset_dir ./datasets \
    --output_dir ./SmaulNative
```

The native extension is compiled for the host CPU with `-march=native`; do not copy a built extension between different CPU architectures. On an i3-3220, 2 threads are recommended over all 4 hardware threads because Hyper-Threading can reduce throughput for this workload.

## Configuration

The default model configuration is approximately 256M parameters for a 65K vocabulary. `--n_embd` controls width, `--n_layer` controls depth, and `--n_moba_layer` controls the number of MOBA blocks. `--head_size` must divide `--n_embd`, and at least one RWKV block must remain.

For CPU training, start with a small `--ctx_len`. MOBA attention uses causal scaled-dot-product attention and becomes increasingly expensive as sequence length grows.

## Remote Streaming

`--stream_dataset` can use `hindi`, `english`, `openthoughts`, or `all` to stream filtered Parquet records directly from Hugging Face without downloading the dataset first. Streaming checkpoints preserve the dataset/file/row position and the partially filled token buffer.

## Resume

Training checkpoints preserve model weights, optimizer state, RNG state, dataset position, token count, and streaming buffer state. Re-running the same training command resumes from the saved checkpoint. RQT optimizer state is keyed by stable module and parameter names and validates tensor shapes during restore.

## Testing

Run the optimization and model tests with:

```bash
python3 tests/test_optimizations.py
```

For RQT specifically:

```bash
python3 -m pytest tests/test_rqt.py
```

The native WKV regression test is:

```bash
python3 tests/test_wkv_native.py
```

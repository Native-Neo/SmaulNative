# SmaulNative

A compact RWKV-X training and inference repository with English-Hindi data tooling, Mixture of Experts (MoE) upcycling, 3-bit Quantization-Aware Training (QAT), and native CPU acceleration.

## Overview

- **RWKV-X Architecture**: RWKV-7 TimeMix blocks with interleaved MOBA attention (`rwkv_x_core.py`).
- **Native CPU Backend**: C++ WKV forward/backward kernels plus a fused AVX Lion optimizer and CPU-specific training path (`cpu/`).
- **Bilingual Pipeline**: Tools for downloading web text, generating synthetic instruction pairs, and training a byte-level BPE tokenizer.
- **Unified Training Pipeline**: `train.py` supports pretraining, SFT, streaming resume, QAT, and router-only MoE fine-tuning.
- **MoE Upcycling**: Merge multiple dense domain checkpoints into a sparse Mixture of Experts model (`merge_moe.py`).
- **3-bit QAT**: Train with fake-quantized Channel-Mix linears and optionally export packed int3 weights (`qat.py`).

## Project Layout

```
├── cpu/               # Native WKV and Lion kernels, benchmarks, and CPU trainer
├── datasets/          # Local training datasets and token streams
├── docs/              # Detailed guides and command references
├── tests/             # Unit and optimization regression tests
├── USEME.md           # CLI cheat sheet
├── dataset.py         # Streaming dataset loaders
├── download.py        # Dataset downloader
├── merge_moe.py       # Dense-to-MoE upcycling
├── qat.py             # 3-bit QAT modules
├── rwkv_x_core.py     # Core RWKV-X model definition
├── syntheticdata.py   # Synthetic bilingual data generator
├── tokenizer.py       # Byte-level BPE tokenizer trainer
└── train.py           # Pretraining, SFT, and router training entrypoint
```

## Quickstart

### 1. Installation

```bash
pip install torch tokenizers safetensors huggingface_hub pyarrow datasets tqdm
```

For native CPU acceleration:

```bash
pip install -r cpu/cpu_native_requirements.txt
```

### 2. End-to-End Workflow

```bash
# 1. Download or generate data
python download.py
# or: python syntheticdata.py --output_file datasets/synthetic.jsonl

# 2. Train a tokenizer (optional: train.py can auto-train one if missing)
python tokenizer.py --dataset_dir ./datasets --output ./SmaulNative/tokenizer.json --vocab_size 32768

# 3. Pretraining
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --ctx_len 256 \
    --tokenizer_path ./SmaulNative/tokenizer.json

# 4. SFT + 3-bit QAT
python train.py --mode sft --dataset_dir ./datasets --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json --qat --qat_export_dir ./SmaulNative-int3
```

### 3. CPU Training

The CPU backend uses the native WKV implementation automatically when `--cpu` is enabled. For the dedicated optimized CPU trainer:

```bash
SMAUL_CPU_THREADS=2 python cpu/cpu_train.py --mode pretrain --dataset_dir ./datasets \
    --output_dir ./SmaulNative --optimizer lion
```

The native extension is compiled for the host CPU with `-march=native`; do not copy a built extension between different CPU architectures. On an i3-3220, 2 threads are recommended over all 4 hardware threads because Hyper-Threading can reduce throughput for this workload.

## Configuration

New models target 256M parameters by default. `--n_embd` controls width, while depth is selected automatically to stay near the target. `--head_size` must divide `--n_embd`, and `--n_moba_layer` must leave at least one RWKV layer.

For CPU training, start with a small `--ctx_len`. MOBA attention uses full causal scaled-dot-product attention on CPU and therefore becomes expensive at long sequence lengths.

## Resume and QAT

Training checkpoints preserve model weights, optimizer state, dataset position, token count, and streaming buffer state. Re-running the same training command resumes from the saved checkpoint. `--new_data` resets the dataset position while keeping model and optimizer state.

With `--qat`, Channel-Mix linears use fake quantization during training. `--qat_export_dir` converts a copy of the trained model to packed 3-bit weights without replacing the normal checkpoint.

## Documentation

Full documentation is available in [`docs/`](docs/):

- [Quickstart Guide](docs/quickstart.md)
- [Dataset Pipeline](docs/dataset.md)
- [Data Downloader](docs/download.md)
- [Synthetic Data Generation](docs/syntheticdata.md)
- [Tokenizer Training](docs/tokenizer.md)
- [Model Architecture](docs/rwkv_x_core.md)
- [Training & SFT](docs/train.md)
- [CPU Optimizations](docs/cpu.md)
- [MoE Upcycling](docs/merge_moe.md)
- [3-bit QAT](docs/qat.md)
- [CLI Reference](USEME.md)

## Testing

Run the optimization and model tests with:

```bash
python3 tests/test_optimizations.py
```

The native WKV regression test is:

```bash
python3 tests/test_wkv_native.py
```

# SmaulNative

A compact, high-efficiency RWKV-X training and inference repository featuring bilingual English-Hindi support, Mixture of Experts (MoE) upcycling, 3-bit Quantization-Aware Training (QAT), and fused native CPU acceleration.

## Overview

- **RWKV-X Architecture**: Linear attention with multi-head orthogonal bidirectional attention (`rwkv_x_core.py`).
- **CPU Optimizations**: Fused native C++ AVX Lion optimizer and thread-optimized training loop in `cpu/`.
- **Bilingual Pipeline**: Tools for downloading web text (`download.py`), generating synthetic bilingual instruction pairs (`syntheticdata.py`), and byte-level BPE tokenizer training (`tokenizer.py`).
- **Unified Training Pipeline**: Single training entrypoint (`train.py`) supporting pretraining, SFT, resumed streaming, QAT, and router-only MoE fine-tuning.
- **MoE Upcycling**: Merge multiple dense domain experts into a sparse Mixture of Experts model (`merge_moe.py`).
- **Quantization (QAT)**: 3-bit weight quantization for memory-constrained deployment (`qat.py`).

## Project Layout

```
├── cpu/               # Native C++ Lion optimizer kernels, benchmark, and CPU trainer
├── datasets/          # Local training datasets and token streams
├── docs/              # Detailed guides and command references for each component
├── tests/             # Unit and optimization regression tests
├── USEME.md           # Quick CLI cheat sheet for every command
├── dataset.py         # Streaming dataset loaders (PretrainStream, SFTDataset)
├── download.py        # Dataset downloader (FineWeb English / Hindi)
├── merge_moe.py       # Upcycle dense checkpoints into MoE
├── qat.py             # 3-bit Quantization-Aware Training modules
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

For native CPU acceleration (optional C++ extension):
```bash
pip install -r cpu/cpu_native_requirements.txt
```

### 2. End-to-End Workflow

```bash
# 1. Download or generate data
python download.py
# or: python syntheticdata.py --output_file datasets/synthetic.jsonl

# 2. Train tokenizer (optional: train.py will auto-train one if missing)
python tokenizer.py --dataset_dir ./datasets --output ./SmaulNative/tokenizer.json --vocab_size 32768

# 3. Pretraining
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --ctx_len 256

# 4. Supervised Fine-Tuning (SFT) + 3-bit QAT
python train.py --mode sft --dataset_dir ./datasets --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json --qat --qat_export_dir ./SmaulNative-int3
```

### 3. CPU Training

To use the host-native fused Lion optimizer on CPU:

```bash
python cpu/cpu_train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --optimizer lion
```

## Documentation

Full documentation for each module is located in [`docs/`](docs/):

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

Run optimization and model tests:

```bash
python3 tests/test_optimizations.py
```

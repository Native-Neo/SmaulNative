# SmaulNative

A compact RWKV-X training and inference repository with English-Hindi data tooling, Mixture of Experts (MoE) upcycling, low-bit floating-point QAT, GGUF conversion/quantization, ONNX packaging, and native CPU acceleration.

## Overview

- **RWKV-X Architecture**: RWKV-7 TimeMix blocks with interleaved MOBA attention.
- **Native Rust Runtime**: The main CLI dynamically loads workflow `.so` libraries and dispatches to dedicated native workflow executables.
- **Native CPU Backend**: Native WKV forward/backward kernels plus CPU-specific training support.
- **Bilingual Tokenizer**: A custom word/character tokenizer with Devanagari grapheme fallback, case markers, and special tokens.
- **Unified Training Pipeline**: Native Rust training supports pretraining and SFT, with streaming/resume support and Lion or AdamW optimizers.
- **MoE Upcycling**: Merge multiple dense domain checkpoints into a sparse Mixture of Experts model.
- **Low-bit floating-point QAT**: FP2, FP4, and FP8 modes are supported by the low-bit training/export code.
- **GGUF tooling**: Safetensors -> GGUF conversion plus native classic GGUF quantization.
- **ONNX export**: Native Rust writes a valid ONNX package containing the model weights and a SmaulNative RWKV-X custom operator.

## Rust CLI

Build the native applications and workflow shared libraries with:

```bash
cargo build
```

The build produces the launcher and dedicated workflow executables in `target/debug/`, together with the seven workflow libraries:

```text
target/debug/
├── smaul-native
├── smaul-finetune
├── smaul-sft
├── smaul-pretrain
├── smaul-quantize-gguf
├── smaul-convert-gguf
├── smaul-export-onnx
├── smaul-infer
├── libsmaul_finetune.so
├── libsmaul_sft.so
├── libsmaul_pretrain.so
├── libsmaul_quantize-gguf.so
├── libsmaul_safetensors-gguf.so
├── libsmaul_export-onnx.so
└── libsmaul_inference.so
```

The launcher is a CLI rather than an interactive menu. Use a workflow name as the first argument; all following arguments are passed to that workflow:

```bash
./target/debug/smaul-native --help
./target/debug/smaul-native finetune --model ./SmaulNative --dataset ./datasets/train.jsonl
./target/debug/smaul-native sft --model ./SmaulNative --dataset ./datasets/sft.jsonl
./target/debug/smaul-native pretrain --model ./SmaulNative --dataset ./datasets/train.jsonl
./target/debug/smaul-native quantize-gguf model.gguf model-q.gguf --type q4_0
./target/debug/smaul-native safetensors-gguf ./SmaulNative ./model.gguf --dtype f16
./target/debug/smaul-native export-onnx ./SmaulNative ./model.onnx
./target/debug/smaul-native inference --model ./SmaulNative
```

Aliases are available for convenience: `fine-tune`, `quantize`, `convert-gguf`, `onnx`, and `infer`. `--version` prints the launcher version.

The launcher loads the selected `libsmaul_*.so` from the same directory as the launcher. If it is not there, it also checks the Cargo build output directory recorded at compile time. The shared library then dispatches to the corresponding native workflow executable.

## GGUF quantization

Native GGUF quantization reads an existing GGUF model, dequantizes each supported tensor to FP32, then writes 32-element classic GGUF quantization blocks. The current native quantizers are:

```text
q4_0
q4_1
q5_0
q5_1
q8_0
```

Example:

```bash
./target/debug/smaul-quantize-gguf model.gguf model-q4.gguf --type q4_0
```

Tensors whose element count is not divisible by 32 are preserved as FP16 instead of being incorrectly packed. Existing GGUF metadata and tokenizer strings are retained where the native writer supports their metadata types.

The current implementation deliberately targets the classic 32-value GGUF formats first. K-quants such as Q2_K/Q3_K/Q4_K/Q5_K/Q6_K are not silently approximated as another format; they remain a separate implementation target.

## ONNX export

The native exporter produces a valid ONNX protobuf containing the Safetensors weights and a `smaulnative::RWKVX` custom operator. The operator receives token IDs plus the exported tensors and carries the original `config.json` as an ONNX string attribute.

```bash
./target/debug/smaul-export-onnx ./SmaulNative ./model.onnx
```

This makes the model representation self-contained at the ONNX format level, but execution requires a runtime that registers the `smaulnative::RWKVX` operator. It is not claimed to be a standard-operator-only ONNX graph yet; standard ONNX operator lowering for the full RWKV-X/MOBA recurrence is a separate backend task.

## Project Layout

```text
├── cpu/               # Native WKV kernels and CPU benchmarks
├── datasets/          # Local training datasets and token streams
├── docs/              # Detailed guides and command references
├── tests/             # Unit and optimization regression tests
├── workflow_plugins/  # Workflow shared-library entry points
├── src/bin/           # Dedicated native workflow applications
├── src/gguf_quantize.rs
├── src/onnx_export.rs
├── USEME.md           # CLI cheat sheet
├── dataset.py         # Dataset loaders
├── download.py        # Dataset downloader
├── merge_moe.py       # Dense-to-MoE upcycling
├── qat.py             # Low-bit floating-point QAT and packed weights
├── rwkv_x_core.py     # Core RWKV-X model definition
├── stream_data.py     # Remote Hugging Face Parquet streaming
├── syntheticdata.py   # Synthetic bilingual data generator
├── tokenizer.py       # Custom bilingual tokenizer
└── train.py           # Legacy Python training entry point
```

## Python Quickstart

The Python pipeline remains available for the existing data and model tooling:

```bash
pip install torch safetensors huggingface_hub pyarrow tqdm

python download.py
# or: python syntheticdata.py --output_file datasets/synthetic.jsonl

python tokenizer.py train --fromdataset ./datasets --output ./SmaulNative/tokenizer.json --vocab-size 32768

python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --ctx_len 256 \
    --tokenizer_path ./SmaulNative/tokenizer.json
```

## Configuration

The default model configuration is approximately 256M parameters for a 65K vocabulary. `--n_embd` controls width, `--n_layer` controls depth, and `--n_moba_layer` controls the number of MOBA blocks. `--head_size` must divide `--n_embd`, and at least one RWKV block must remain.

For CPU training, start with a small context length. MOBA attention becomes increasingly expensive as sequence length grows.

## Resume and QAT

Training checkpoints preserve model weights, optimizer state, RNG state, dataset position, token count, and streaming buffer state. The native Rust trainer supports Lion and AdamW. The Python QAT pipeline supports FP2, FP4, and FP8 floating-point modes.

## Testing

For the native Rust project:

```bash
cargo test --lib
cargo build
```

The Python optimization tests remain available:

```bash
python3 tests/test_optimizations.py
python3 tests/test_wkv_native.py
```

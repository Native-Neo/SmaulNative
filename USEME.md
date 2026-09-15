# USEME

Practical commands for running SmaulNative. Run commands from the repository root.

## Native Rust CLI

Build the native launcher, workflow applications, and workflow shared libraries:

```bash
cargo build
```

Show the CLI help:

```bash
./target/debug/smaul-native --help
```

Run a workflow directly:

```bash
./target/debug/smaul-native finetune --model ./SmaulNative --dataset ./datasets/train.jsonl
./target/debug/smaul-native sft --model ./SmaulNative --dataset ./datasets/sft.jsonl
./target/debug/smaul-native pretrain --model ./SmaulNative --dataset ./datasets/train.jsonl
./target/debug/smaul-native quantize-gguf --input model.gguf --output model-q.gguf
./target/debug/smaul-native safetensors-gguf ./SmaulNative ./model.gguf --dtype f16
./target/debug/smaul-native export-onnx --model ./SmaulNative --output model.onnx
./target/debug/smaul-native inference --model ./SmaulNative
```

Aliases: `fine-tune`, `quantize`, `convert-gguf`, `onnx`, and `infer`.

All arguments after the workflow name are forwarded to the selected workflow application.

The launcher dynamically loads the matching `libsmaul_*.so` before dispatching to the workflow executable. The libraries are built into `target/debug/` beside the native binaries.

## Native Applications

The workflow applications can also be run directly:

```bash
./target/debug/smaul-finetune
./target/debug/smaul-sft
./target/debug/smaul-pretrain
./target/debug/smaul-quantize-gguf
./target/debug/smaul-convert-gguf
./target/debug/smaul-export-onnx
./target/debug/smaul-infer
```

## Python Entry Points

The Python tooling remains available for data preparation and the existing Python training/QAT pipeline:

- [Quickstart](docs/quickstart.md) -- setup + end-to-end example
- [download.py](docs/download.md)
- [syntheticdata.py](docs/syntheticdata.md)
- [tokenizer.py](docs/tokenizer.md)
- [dataset.py](docs/dataset.md)
- [train.py](docs/train.md)
- [qat.py](docs/qat.md)
- [rwkv_x_core.py](docs/rwkv_x_core.md)
- [merge_moe.py](docs/merge_moe.md)
- [CPU Optimizations](docs/cpu.md)

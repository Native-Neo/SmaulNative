# Quickstart

Run commands from the repository root.

## Install

```bash
python -m pip install -r requirements.txt
```

For the native CPU WKV backend, `ninja` and a working C++ compiler are required.

## Pretraining

Create or provide data under `./datasets`. The tokenizer is trained automatically when the configured
`tokenizer_path` does not exist.

```bash
python train.py --cpu --mode pretrain \
    --dataset_dir ./datasets \
    --output_dir ./SmaulNative \
    --ctx_len 512
```

The default model configuration is 832 hidden dimensions, 17 layers, 64-wide heads, and 3 MOBA layers.
For CPU training, keep `ctx_len` modest because MOBA attention is quadratic in sequence length.

## SFT

Use conversation JSON/JSONL records shaped like:

```json
{"conversations":[{"from":"user","value":"Hello"},{"from":"assistant","value":"Hi!"}]}
```

Reuse the pretrained tokenizer:

```bash
python train.py --cpu --mode sft \
    --dataset_dir ./sft_data \
    --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json
```

Only assistant response tokens contribute to SFT loss.

## Streaming datasets

For supported Hugging Face Parquet sources, pretraining can stream without storing the source dataset:

```bash
python train.py --cpu --mode pretrain \
    --stream_dataset hindi \
    --output_dir ./SmaulNative
```

Supported stream names are `hindi`, `english`, `openthoughts`, and `all`.

## QAT

```bash
python train.py --mode sft \
    --dataset_dir ./sft_data \
    --output_dir ./SmaulNative-SFT \
    --tokenizer_path ./SmaulNative/tokenizer.json \
    --qat --qat_calib_batches 64 \
    --qat_export_dir ./SmaulNative-int3
```

The training checkpoint remains fake-quantized FP32; `--qat_export_dir` writes the converted packed int3
checkpoint.

## Tokenizer

To train one manually:

```bash
python tokenizer.py train \
    --fromdataset ./datasets \
    --vocab-size 65536 \
    --output ./tokenizer.json
```

The tokenizer is the project's custom word/grapheme/character vocabulary format, not a Hugging Face
`tokenizers` BPE tokenizer.

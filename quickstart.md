# Quickstart

The fastest way to "actually run this". Several one-time `pip` packages are needed; run the
export from the repo root.

## Setup

```bash
# All the dependencies for every branch
pip install torch transformers tokenizers safetensors datasets pyarrow tqdm pyyaml psutil pandas huggingface_hub 

# All dependencies for Main
pip install torch tokenizers safetensors huggingface_hub pyarrow
```

A Hugging Face token is only required for the gated/rate-limited downloads in
[download.md](download.md) (`huggingface-cli login`).

## Suggested first run-through

The pieces chain together, but most are optional depending on what you want:

1. **Get data** -- either pull real web text with [download.md](download.md), or generate a
   synthetic bilingual instruction set with [syntheticdata.md](syntheticdata.md).
2. **Tokenizer** -- [tokenizer.md](tokenizer.md) trains a byte-level BPE tokenizer. You usually
   don't run this by hand: `train.py` auto-trains one if `--tokenizer_path` doesn't exist.
3. **Pretrain** -- [train.md](train.md) `--mode pretrain`.
4. **SFT** -- [train.md](train.md) `--mode sft` on top of the pretrained checkpoint.
5. **Optional** -- [merge_moe.md](merge_moe.md) to combine SFT domains into one MoE model;
   [qat.md](qat.md) to quantize (via `train.py --qat`).

## Full end-to-end

```bash
python download.py
python tokenizer.py --dataset_dir ./datasets --output ./SmaulNative/tokenizer.json --vocab_size 131072
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M --ctx_len 256
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT \
    --tokenizer_path ./RWKV-X-256M/tokenizer.json --qat --qat_export_dir ./RWKV-X-SFT-int3
```

Per-step flags and behavior are documented on each module's page: [download.md](download.md),
[syntheticdata.md](syntheticdata.md), [tokenizer.md](tokenizer.md), [dataset.md](dataset.md),
[train.md](train.md), [qat.md](qat.md), [merge_moe.md](merge_moe.md).

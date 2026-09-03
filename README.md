# SmaulNative

SmaulNative is an experimental language-model training project built around small, CPU-friendly implementations and custom data tooling.

The active root implementation is a pure-PyTorch RWKV-X-style model with RWKV-7 TimeMix and MOBA attention, tokenizer training, pretraining/SFT, MoE upcycling, and optional 3-bit QAT.

## Active root pipeline

```text
download.py -> tokenizer.py -> dataset.py -> train.py -> qat.py
                                      |
                                      v
                                 merge_moe.py
```

## Core files

- `rwkv_x_core.py` — RWKV-X-style model implementation.
- `train.py` — pretraining and SFT entrypoint.
- `tokenizer.py` — byte-level BPE tokenizer training.
- `dataset.py` — recursive dataset loading and streaming.
- `download.py` — dataset acquisition.
- `syntheticdata.py` — synthetic instruction-data generation.
- `merge_moe.py` — Channel-Mix MoE upcycling and tokenizer union merging.
- `qat.py` — 3-bit quantization-aware training/export.
- `router_utils.py` — router-only training utilities.

## Training architecture

The trainer accepts the architecture directly instead of selecting layers from a parameter target:

```bash
python train.py --mode pretrain \
  --dataset_dir ./datasets \
  --n_layer 20 \
  --n_embd 832 \
  --head_size 64 \
  --n_moba_layer 5
```

`n_layer`, `n_embd`, `head_size`, and `n_moba_layer` are forced exactly as supplied. The tokenizer vocabulary is used as the model vocabulary size and therefore determines the embedding and output-head vocabulary dimension; it does not change the requested architecture depth or width.

When resuming, the trainer validates the checkpoint architecture and tokenizer vocabulary instead of silently loading a different configuration.

## Tokenizer

```bash
python tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json --vocab_size 65536
```

If `train.py` cannot find the requested tokenizer, it trains one automatically. Checkpoints bundle the tokenizer they were trained with.

## Data and SFT

`dataset.py` supports text, JSON/JSONL, CSV, Parquet, and source-code files. Pretraining uses fixed-length token streaming with resumable dataset position tracking. SFT supports ChatML-style conversation records and loss masking.

## QAT and MoE

QAT can be enabled with `--qat`. The Channel-Mix FFN projections are fake-quantized during training and can optionally be exported as packed int3 weights.

`merge_moe.py` combines compatible checkpoints by turning Channel-Mix FFNs into experts and can union their BPE tokenizers, resizing vocabulary-dependent layers when required.

## Repository branches

- `main` — primary root implementation.
- `developement` — active development line.
- `forced-architecture-training` — development of explicit, forced training architecture configuration.
- `transformer` — legacy Transformer implementation; no longer actively updated.
- `transformers-based-RWKV` — legacy Transformers-based RWKV implementation; no longer actively updated.

The legacy branches are retained for historical/reference purposes rather than as additional active root implementations.

## Requirements

Typical dependencies:

```bash
pip install torch transformers tokenizers safetensors datasets pyarrow tqdm pyyaml psutil pandas huggingface_hub
```

For CPU-only training, keep sequence length and batch size appropriate for available RAM and CPU performance.

## Status

SmaulNative is experimental. Architecture, checkpoint formats, training behavior, and data tooling may change between development revisions. Check the branch and model configuration before reusing checkpoints.

## License

### ARR (All-Rights-Reserved)

- You may use the repository in your own open-source projects without asking for permission.
- Derivative projects must remain open-source and may not add paid/monetised versions of this code.
- Credit the rightful owners of this repository.
- Do not distribute the project as your own product.

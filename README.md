# SmaulNative

- SmaulNative is an experimental language-model development project focused on building and training models from scratch with custom data pipelines, tokenizer training, memory-aware training loops, RWKV-style architectures, Transformer variants, and Mixture of Experts model merging.

- a pure-PyTorch reimplementation of the real howard-hou/RWKV-X model math (RWKV-7 TimeMix + MOBA sparse attention), CPU-safe, with pretraining/SFT, tokenizer training, and MoE upcycling that can also merge tokenizers.

SmaulNative is intended as a practical experimentation environment for training and modifying language models without depending entirely on a large external training framework.

## Repository
The project is hosted at:
https://github.com/Native-Neo/SmaulNative

## Current Repository Layout
```text
SmaulNative/
├── .gitignore
├── README.md
├── data_g.py           # PyArrow-accelerated dataset cleaning/filtering/dedup
├── download_g.py             # Streaming dataset downloader with per-dataset GB caps
├── syntheticdata_g.py  # Synthetic bilingual instruction-data generator
├── rwkv_x_core.py            # Pure-PyTorch RWKV-X model math (RWKV-7 + MOBA hybrid)
├── train.py            # Single entrypoint for pretraining + SFT
├── dataset.py                # Universal recursive dataset loader (pretrain stream + SFT)
├── tokenizer.py        # Byte-level BPE tokenizer training -> single tokenizer.json
└── merge_moe.py              # MoE upcycling + tokenizer union-merging
```
This directory reimplements the **real** howard-hou/RWKV-X training model math in pure PyTorch so it runs on a CPU-only box. The WKV-7 recurrence here is not guessed — it is upstream's own CUDA-free reference loop made batched and differentiable, algebraically verified against their CUDA kernel's decay formula (`w_eff = exp(-exp(w_raw))` closed form).

```text
download_g.py    data_g.py    syntheticdata_g.py    Tokenizer.py    train.py    merge.py
```

### `download_g.py`

Streams pre-training datasets from Hugging Face into local JSONL files with a configurable per-dataset size cap (default 20 GB), resumable across runs. Default corpus set includes FineWeb-Edu, English/Hindi wiki data, and Hindi PDFs.

### `data_g.py`

A PyArrow C++ accelerated cleaning/filtering pipeline operating at million-row scale:

- Reads Parquet, JSON, JSONL, CSV, TSV, text/docs, and common source-code files.
- Vectorized length filtering and table-level exact SHA256/string deduplication.
- Exports sharded ZSTD-compressed Parquet.

Usage:

```bash
python3 data_g.py --input ./datasets --output ./cleaned_datasets --workers 4
```

### `syntheticdata_g.py`

Generates millions of unique bilingual (English/Hindi) instruction-response pairs across math, computer science, algorithms, cyber security, science, and reasoning:

- Combinatorial prompt space (> 10^12 variations) with zero-duplicate hash enforcement.
- Chain-of-thought `<think>...</think>` reasoning traces.
- ChatML-style formatting (`<|im_start|>`, `<|im_end|>`).
- High-throughput ZSTD Parquet / JSONL export (a sample lives in `datasets/synthetic_bilingual.parquet`).

### `rwkv_x_core.py`

- `RWKV_Tmix_x070`: exact RWKV-7 TimeMix with value-residual, gated decay, and O(1)-per-token recurrent state (streaming/unlimited-context generation at inference; training still uses a finite BPTT window).
- MOBA sparse-attention blocks interleaved among RWKV blocks (CPU falls back to full causal SDPA — correct but O(T^2)).
- `config_for_target_params()`: solves layer count to hit a parameter target (default ~256M).
- HF-style `save_pretrained()` / `from_pretrained()` (config.json + model.safetensors) plus an upstream-shaped `.pth` export loadable by the real `rwkv-x` pip package on CUDA boxes.

### `tokenizer.py`

Trains a byte-level BPE tokenizer directly on the dataset directory using the same text extraction logic as training. Outputs a single merged `tokenizer.json` (vocab + merges + special tokens), replacing upstream's fixed-vocab TRIE tokenizer setup entirely.

### `dataset.py`

Universal dataset loader: recursively walks the dataset dir for txt/jsonl/json/csv/parquet/source files and yields fixed-length chunks for pretraining or `-100`-masked pairs for SFT.

### `train.py`

Single entrypoint for both pretraining and SFT:

```bash
python actual_rwkv_proto/train.py --mode pretrain --dataset_dir ./datasets
python actual_rwkv_proto/train.py --mode sft     --dataset_dir ./sft_data
```

- Auto-trains the tokenizer if none exists yet.
- Lion optimizer by default (AdamW available); gradient clipping; non-finite-loss skip.
- Ctrl-C-only checkpointing: finishes the current step, then saves model + optimizer + dataset position.
- Full resume across runs; `--new_data` resets dataset accounting while keeping weights.
- Bundles the trained tokenizer into every checkpoint so later stages never guess.

Honest caveat: pure-Python recurrence is slow on CPU (no compiled kernel). Keep `--ctx_len` modest.

### `merge_moe.py`

Combines same-architecture checkpoints (e.g. one base pretrain + several domain SFTs) into an MoE checkpoint, **and merges their tokenizers**:

- Everything except the Channel-Mix FFN stays shared from the base.
- Each branch's FFN becomes an expert with a learned top-k router.
- Tokenizers are union-merged: base token ids preserved unchanged, new tokens appended without duplicates, BPE merge rules de-duplicated with base order preserved.
- If the merged vocab grows, embeddings/head are resized before assembly.

Note: MoE-upcycled checkpoints are this project's own extension and load only via `RWKVXModel` (with `is_moe: true` in config.json), not the upstream pip package.

---

# Typical Workflow

```text
Dataset Sources
      |
      v
download_g.py  -->  data_g.py (clean/filter/dedup)  -->  syntheticdata_g.py
      |                                                        |
      v                                                        v
Tokenize ( tokenizer.py )  <-----------  Synthetic corpora
      |
      v
Pretrain: train.py 
      |
      v
Fine-tune domain branches (SFT)
      |
      v
Merge: merge_moe.py
      |
      v
MoE / Merged Model
```

The exact scripts used depend on the architecture being trained.

---

# Model Checkpoints

SmaulNative uses Hugging Face-style model directories where supported.

A typical checkpoint may contain:

```text
model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── tokenizer_config.json
```

Large checkpoints may use multiple Safetensors shards with a `model.safetensors.index.json`. All merge pipelines are designed to process checkpoints in a memory-aware manner rather than loading every tensor simultaneously.

This additionally exports an upstream-shaped `.pth` (`rwkvx_upstream_compatible.pth`) so real `rwkv-x` package inference can run on a CUDA box later.

---

# Design Goals

- Training language models from scratch, including on modest CPU-only hardware.
- Supporting experimentation with multiple architectures (`custom RWKV-style, RWKV-X, Llama-style` **See other branches**).
- Keeping model tooling understandable and modifiable.
- Resumable training and checkpointing everywhere (model + optimizer + dataset position).
- Safetensors and Hugging Face-style model formats.
- Self-trained tokenizers bundled with the checkpoints that use them.
- Combining specialized checkpoints through model merging and MoE upcycling.
- Custom data acquisition, cleaning, and synthetic-data generation alongside model code.

---

# Requirements

Dependencies vary by pipeline. Commonly required packages:

```bash
pip install torch transformers tokenizers safetensors datasets pyarrow tqdm pyyaml psutil pandas
```

Additional dependencies may be required by individual scripts.

---

# Running the Project

The repository does not use a single universal command because it contains multiple independent tools.

Root pipeline:

```bash
python download_g.py                       # fetch datasets
python data_g.py --input ./datasets --output ./cleaned_datasets
python syntheticdata_g.py --count 250000 --format both
python tokenizer.py                        # train tokenizer -> ./SmaulNative
python train.py --mode pretrain            # Runs pretraining with the RWKV-X achitecture 
python train.py --mode sft                 # Runs SFT (SUper vised FineTuning)
python merge_moe.py --help                 # MoE upcycle
python rwkv_x_core.py                      # Just a core requirement of train.py
```

The available command-line arguments depend on the individual script (`--help` works everywhere).

---

# Project Status

SmaulNative is an experimental and actively evolving project. The repository contains multiple architecture implementations (See other branches) and independent tooling paths; model formats, training behavior, merging logic, dataset pipelines, and configuration formats may change as development continues. Compatibility between checkpoints depends on the model architecture and the version of the corresponding training or merging implementation.

Recent work has centered on verified RWKV-X math, self-training tokenizers, tokenizer bundling in checkpoints, and tokenizer-aware MoE merging.

---

# Contributing

Contributions, experiments, architecture improvements, training optimizations, dataset tooling improvements, and bug reports are welcome.

Because the project contains separate architecture paths, changes should clearly indicate whether they target:

- The root RWKV-X implementation.
- The `trnasformer` implementation.
- The `transformers-based-RWKV` implementation.
- Shared data or tokenizer tooling.
- Model merging infrastructure.

---

# License

### ARR (All-Rights-Reserved) and you must:
- Not need to ask for permission to use the repo for your own open-source projects.
- Keep the project open-source without any paid/monitised content (such as enhanced versions of this code but paid).
- Always give credit to this repository's rightful owners.
- Don't distribute as your own product.

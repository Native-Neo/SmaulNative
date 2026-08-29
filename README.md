# SmaulNative

- SmaulNative is an experimental language-model development project focused on building and training models from scratch with custom data pipelines, tokenizer training, memory-aware training loops, RWKV-style architectures, Transformer variants, Mixture of Experts model merging, and 3-bit quantization-aware training.

- A pure-PyTorch reimplementation of the real howard-hou/RWKV-X model math (RWKV-7 TimeMix + MOBA sparse attention), CPU-safe, with pretraining/SFT, tokenizer training, MoE upcycling that can also merge tokenizers, and 3-bit (int3) QAT for the Channel-Mix FFN layers.

SmaulNative is intended as a practical experimentation environment for training and modifying language models without depending entirely on a large external training framework.

## Repository
The project is hosted at:
https://github.com/Native-Neo/SmaulNative

## Current Repository Layout
```text
SmaulNative/
├── .gitignore
├── README.md
├── dataset.py        # Recursive multi-format dataset loader (pretrain stream + SFT)
├── download.py       # FineWeb English & Hindi streaming dataset downloader
├── merge_moe.py      # MoE upcycling + tokenizer union-merging
├── qat.py            # 3-bit (int3) quantization-aware training for the Channel-Mix (FFN) linears
├── rwkv_x_core.py    # Pure-PyTorch RWKV-X model math (RWKV-7 + MOBA hybrid)
├── syntheticdata.py  # Synthetic bilingual instruction-data generator
├── tokenizer.py       # Byte-level BPE tokenizer training -> single tokenizer.json
└── train.py           # Single entrypoint for pretraining + SFT (+ optional QAT)
```

This directory reimplements the **real** howard-hou/RWKV-X training model math in pure PyTorch so it runs on a CPU-only box. The WKV-7 recurrence here is upstream's own CUDA-free reference loop made batched and differentiable, algebraically verified against their CUDA kernel's decay formula (`w_eff = exp(-exp(w_raw))` closed form).

```text
download.py    dataset.py    syntheticdata.py    tokenizer.py    train.py    qat.py    merge_moe.py
```

### `download.py`

Streams FineWeb English (`HuggingFaceFW/fineweb`, `data/100BT`) and FineWeb-2 Hindi (`HuggingFaceFW/fineweb-2`, `data/hin_Deva/train`) from Hugging Face into `./datasets/raw/` with a configurable size cap per language (default 40 GiB), resumable across runs.

### `dataset.py`

Universal recursive multi-format dataset loader supporting plain text (`.txt`), JSON/JSONL (`.json`, `.jsonl`), CSV (`.csv`), Parquet (`.parquet`), and source code files (`.py`, `.cpp`, `.rs`, `.ts`, etc.):

- **Pretraining**: `PretrainStream` yields fixed-length token chunks (`ctx_len`) for causal LM training with record-based dataset position tracking for seamless resumes.
- **SFT**: `SFTDataset` processes ChatML conversation records (`conversations` key) with loss masking (`-100`) applied to user turns.
- **Tokenizer Wrapper**: `TokenizerWrapper` provides unified `encode()`/`decode()` interfaces with explicit `<pad>` and `<eos>` special token management.

### `syntheticdata.py`

Generates millions of unique bilingual (English/Hindi) instruction-response pairs across math, algebra, systems of linear equations, quadratic equations, sorting algorithms, data structures, cyber security, science, and reasoning:

- Combinatorial prompt space (> 10^12 variations) with zero-duplicate hash enforcement.
- Chain-of-thought `<think>...</think>` reasoning traces.
- ChatML-style formatting (`<|im_start|>`, `<|im_end|>`).
- Sorting-algorithm and data-structure samples are generated with matching language-specific code (Python/JavaScript/C++/Rust for algorithms; Python/C++/Java for Stack/Queue data structures), so the prompt's stated language always matches the code block.
- High-throughput PyArrow ZSTD Parquet / JSONL export (`--count`, `--format`, `--output-dir`).

Usage:

```bash
python syntheticdata.py --count 250000 --format both --output-dir ./datasets
```

### `rwkv_x_core.py`

- `RWKV_Tmix_x070`: Exact RWKV-7 TimeMix with value-residual, gated decay (`w_eff = exp(-0.606531 * sigmoid(w0 + g))`), and O(1)-per-token recurrent state (streaming/unlimited-context generation at inference; training uses a finite BPTT window).
- `MOBABlock`: Sparse-attention blocks interleaved among RWKV blocks (CPU falls back to full causal SDPA).
- `config_for_target_params()`: Solves layer count to hit a parameter target (default ~256M).
- HF-style `save_pretrained()` / `from_pretrained()` (`config.json` + `model.safetensors`) plus an upstream-shaped `.pth` (`rwkvx_upstream_compatible.pth`) loadable by the `rwkv-x` pip package on CUDA boxes.

### `tokenizer.py`

Trains a byte-level BPE tokenizer directly on the dataset directory using Hugging Face `tokenizers`. Outputs a single merged `tokenizer.json` (vocab + merges + special tokens `<pad>`/`<eos>`), replacing upstream's fixed-vocab TRIE setup.

Usage:

```bash
python tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json --vocab_size 32768
```

### `train.py`

Single entrypoint for both pretraining and SFT:

```bash
python train.py --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M
python train.py --mode sft      --dataset_dir ./sft_data  --output_dir ./RWKV-X-SFT
```

- Auto-trains `tokenizer.json` if none exists at `--tokenizer_path`.
- Lion optimizer by default (AdamW available); gradient clipping; non-finite-loss skipping.
- Ctrl-C signal handler: finishes current step, then saves model + optimizer state + dataset position.
- Full resume across runs; `--new_data` resets dataset position while retaining model weights.
- Automatically bundles `tokenizer.json` into every checkpoint directory.
- `--qat`: enables 3-bit (int3) quantization-aware training via `qat.py` (see below) -- calibrates on `--qat_calib_batches` batches, then fine-tunes with fake-quant noise in the loop. `--qat_export_dir` converts the trained model to real packed int3 weights and saves that separately, leaving the fake-quantized/still-fine-tunable checkpoint in `--output_dir` untouched.

Honest caveat: pure-Python recurrence is slow on CPU (no compiled CUDA kernel). Keep `--ctx_len` modest during CPU training.

### `qat.py`

3-bit (int3) quantization-aware training for the Channel-Mix (FFN) `key`/`value` linears -- the dense, high-parameter-count projections, and the intended QAT target. `emb`, `head`, RWKV-7 TimeMix (attention), and MOBA's `CausalSelfAttention` all stay FP32, since those are the layers most sensitive to precision loss.

- **Scheme**: weight = per-channel symmetric int3, 8 levels in `[-4, 3]` (one scale per output row); activation = per-tensor asymmetric int3, 8 levels in `[0, 7]`, ranges settled by a moving-average min/max observer.
- **Fake quantization**: implemented with `torch.ao.quantization.FakeQuantize` restricted to the 3-bit range -- the forward pass sees realistic (aggressive) quantization noise while gradients still flow at full precision (straight-through estimator), so the model adapts to quantization *during* fine-tuning instead of taking an accuracy hit only at the end.
- **Calibration**: `calibrate()` runs a handful of forward-only passes over real dataset text so observer ranges aren't starting from zero when fine-tuning begins.
- **Conversion**: `convert_qat()` bakes the calibrated ranges into *real* int3 weights, hand-packed 8 codes -> 3 bytes (no native sub-byte tensor dtype in torch), giving a genuine ~10.7x reduction vs. FP32 (~2.67x vs. int8) rather than just clamping values into a smaller range while still storing a full byte each. `QuantizedLinear` unpacks and dequantizes on the fly each forward -- CPU-portable, no fbgemm/qnnpack dependency.

Used automatically by `train.py --qat`; can also be imported directly (`prepare_qat`, `calibrate`, `convert_qat`) for custom scripts.

### `merge_moe.py`

Combines same-architecture checkpoints (e.g., base pretrain + domain SFTs) into a Channel-Mix MoE model, **and merges their tokenizers**:

- Everything except Channel-Mix FFN stays shared from base.
- Each branch's FFN becomes an expert with a learned top-k router.
- Tokenizers are union-merged: base token IDs preserved unchanged, new tokens appended without duplicates, BPE merge rules deduplicated with base order preserved.
- If the merged vocabulary grows, embedding and output head matrices are automatically resized before assembly.

Usage:

```bash
python merge_moe.py --base ./RWKV-X-256M --branches ./sft_branch1 ./sft_branch2 --out ./RWKV-X-MoE --top_k 1
```

Note: MoE-upcycled checkpoints are this project's own extension and load via `RWKVXModel` (with `is_moe: true` in `config.json`).

---

# Typical Workflow

```text
Dataset Sources (FineWeb / HF)
      |
      v
 download.py  -->  syntheticdata.py
      |                  |
      +--------+---------+
               |
               v
      Tokenizer ( tokenizer.py )
               |
               v
          dataset.py
               |
               v
        Pretrain: train.py --mode pretrain [--qat]
               |
               v
     Fine-tune: train.py --mode sft [--qat]
               |
               v
      Merge: merge_moe.py
               |
               v
      MoE / Merged Model
               |
               v (optional, if --qat was used)
   qat.convert_qat() / --qat_export_dir --> int3 checkpoint
```

The exact scripts used depend on the pipeline phase being executed.

---

# Model Checkpoints

SmaulNative uses Hugging Face-style model directories where supported.

A typical checkpoint directory contains:

```text
model/
├── config.json
├── model.safetensors
├── tokenizer.json
└── rwkvx_upstream_compatible.pth
```

This additionally exports an upstream-shaped `.pth` (`rwkvx_upstream_compatible.pth`) so real `rwkv-x` package inference can run on CUDA boxes later.

A `--qat_export_dir` checkpoint has the same layout, except the Channel-Mix `key`/`value` weights inside `model.safetensors` are stored as packed 3-bit (int3) buffers (with a per-channel scale buffer) rather than FP32 -- see `qat.py` for the pack/unpack format.

---

# Design Goals

- Training language models from scratch, including on modest CPU-only hardware.
- Supporting experimentation with multiple architectures (`custom RWKV-style, RWKV-X, Llama-style` **See other branches**).
- Keeping model tooling understandable and modifiable.
- Resumable training and checkpointing everywhere (model + optimizer + dataset position).
- Safetensors and Hugging Face-style model formats.
- Self-trained tokenizers bundled with the checkpoints that use them.
- Combining specialized checkpoints through model merging and MoE upcycling.
- Custom data acquisition, multi-format dataset streaming, and synthetic-data generation alongside model code.
- Selective, aggressive (3-bit) quantization-aware training to shrink and speed up deployment without touching the layers most sensitive to precision loss.

---

# Requirements

Dependencies vary by pipeline. Commonly required packages:

```bash
pip install torch transformers tokenizers safetensors datasets pyarrow tqdm pyyaml psutil pandas huggingface_hub
```

`qat.py` uses `torch.ao.quantization`, which ships with `torch` itself -- no extra package needed.

---

# Running the Project

Root pipeline execution commands:

```bash
python download.py                                                    # fetch FineWeb English + Hindi datasets
python syntheticdata.py --count 250000 --format both                  # generate synthetic instruction dataset
python tokenizer.py --dataset_dir ./datasets --output ./tokenizer.json# train BPE tokenizer
python train.py --mode pretrain --dataset_dir ./datasets              # run pretraining (RWKV-X)
python train.py --mode sft --dataset_dir ./sft_data                   # run SFT (Supervised Fine-Tuning)
python train.py --mode sft --dataset_dir ./sft_data --qat \
    --qat_export_dir ./RWKV-X-SFT-int3                                # SFT with QAT, export an int3 checkpoint
python merge_moe.py --base ./RWKV-X-256M --branches ./b1 ./b2 --out ./RWKV-X-MoE # MoE upcycle merge
```

Command-line arguments are supported across scripts (`--help` works everywhere).

---

# Project Status

SmaulNative is an experimental and actively evolving project. The repository contains multiple architecture implementations (See other branches) and independent tooling paths; model formats, training behavior, merging logic, dataset pipelines, and configuration formats may change as development continues. Compatibility between checkpoints depends on the model architecture and the version of the corresponding training or merging implementation.

Recent work has centered on verified RWKV-X math, self-training tokenizers, tokenizer bundling in checkpoints, tokenizer-aware MoE merging, consolidating the dataset loader into a single `dataset.py`, and selective 3-bit QAT for the Channel-Mix FFN layers.

---

# Contributing

Contributions, experiments, architecture improvements, training optimizations, dataset tooling improvements, and bug reports are welcome.

Because the project contains separate architecture paths, changes should clearly indicate whether they target:

- The root RWKV-X implementation.
- The `transformer` implementation. (No longer maintained)
- The `transformers-based-RWKV` implementation. (No longer maintained)
- Shared data or tokenizer tooling.
- Model merging infrastructure.
- Quantization-aware training tooling.

---

# License

### ARR (All-Rights-Reserved) and you must:
- Not need to ask for permission to use the repo for your own open-source projects.
- Keep the project open-source without any paid/monetised content (such as enhanced versions of this code but paid).
- Always give credit to this repository's rightful owners.
- Don't distribute as your own product.

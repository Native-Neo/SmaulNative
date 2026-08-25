# SmaulNative

SmaulNative is an experimental language-model development project focused on building and training models from scratch with custom data pipelines, tokenizer training, memory-aware training loops, RWKV-style architectures, Transformer variants, and Mixture of Experts model merging.

The repository contains two primary architecture paths:

- The root pipeline, centered around the RWKV-X / recurrent architecture and its MoE upcycling tools.
- `Transformer-Varient`, an alternative Transformer-based implementation with its own training, merging, synthetic-data, tokenizer, data-processing, and downloading tools.

SmaulNative is intended as a practical experimentation environment for training and modifying language models without depending entirely on a large external training framework.

## Repository

The project is hosted at:

https://github.com/Native-Neo/SmaulNative

## Current Repository Layout

```text
SmaulNative/
├── .gitignore
├── README.md
│
├── Data.py
├── Tokenizer.py
├── download.py
├── merge.py
├── train.py
│
└── Transformer-Varient/
    ├── Data.py
    ├── Merge.py
    ├── SyntheticData.py
    ├── Tokenizer.py
    ├── Train.py
    └── download.py
```

The root and `Transformer-Varient` directories represent separate model-development paths. Some utility scripts currently appear in both pipelines because each architecture path maintains its own workflow.

---

# Architecture Paths

## 1. RWKV-X / Recurrent Path

The repository root contains the primary RWKV-style implementation:

```text
Data.py
Tokenizer.py
download.py
merge.py
train.py
```

This path covers the workflow from dataset acquisition through tokenizer training, model pre-training, checkpoint handling, and Mixture of Experts upcycling.

### `download.py`

Handles dataset acquisition and downloading for the root training pipeline.

The goal is to provide a dedicated dataset-entry point rather than mixing downloading logic directly into training.

### `Data.py`

Processes datasets for training.

This stage is responsible for preparing source data before tokenizer training or model pre-training. Depending on the current implementation, this can include filtering, cleaning, normalization, conversion, and dataset organization.

### `Tokenizer.py`

Trains or prepares the tokenizer used by the root model pipeline.

The tokenizer is part of the model format, so vocabulary compatibility matters when training models and when merging checkpoints.

### `train.py`

The main RWKV-X training implementation.

The current training path supports a custom RWKV-style architecture with recurrent state, Hugging Face-compatible configuration, Safetensors checkpoints, dataset resume logic, and dense or MoE Channel-Mix execution.

The dense architecture follows the general structure:

```text
Tokens
  |
  v
Embedding
  |
  v
RWKV-X Blocks
  |
  +-- LayerNorm
  |
  +-- Time-Mix
  |      |
  |      +-- recurrent / sequence mixing
  |
  +-- LayerNorm
  |
  +-- Channel-Mix
  |
  v
Final LayerNorm
  |
  v
Language Model Head
```

The model can operate as a standard dense architecture or as a Channel-Mix Mixture of Experts model.

### Dense Channel-Mix

A normal block contains one Channel-Mix module:

```text
blocks.N.channel_mix
```

With parameters such as:

```text
key.weight
value.weight
receptance.weight
time_mix_k
time_mix_r
```

### MoE Channel-Mix

The MoE path replaces a single Channel-Mix module with multiple experts:

```text
blocks.N.channel_mix.experts.0.*
blocks.N.channel_mix.experts.1.*
blocks.N.channel_mix.experts.2.*
...
```

Each MoE-enabled Channel-Mix layer also contains a router:

```text
blocks.N.channel_mix.gate.weight
```

Conceptually:

```text
                    Input
                      |
                      v
                 Router / Gate
                      |
              Top-K Expert Selection
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Expert 0    Expert 1    Expert N
          |           |           |
          +-----------+-----------+
                      |
                      v
                   Output
```

The Time-Mix path remains shared while Channel-Mix modules become specialized experts.

---

# RWKV-X MoE Upcycling

## `merge.py`

`merge.py` is the dedicated RWKV-oriented model-merging tool.

Its purpose is to combine multiple compatible branch checkpoints derived from a shared base model into a Mixture of Experts architecture.

A typical workflow is:

```text
                         Base Model
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
         Code Branch      Math Branch    Language Branch
              |               |               |
              +---------------+---------------+
                              |
                              v
                         merge.py
                              |
                              v
                    SmaulNative MoE Model
```

During upcycling:

- Shared model parameters remain based on the base checkpoint.
- Time-Mix components remain shared.
- Layer normalization remains shared.
- Embeddings remain shared.
- Output layers remain shared.
- Channel-Mix parameters from each branch become separate experts.
- A router is created for every MoE-enabled Channel-Mix layer.

The generated checkpoint layout follows the pattern:

```text
blocks.0.channel_mix.experts.0.key.weight
blocks.0.channel_mix.experts.0.value.weight
blocks.0.channel_mix.experts.0.receptance.weight
blocks.0.channel_mix.experts.0.time_mix_k
blocks.0.channel_mix.experts.0.time_mix_r

blocks.0.channel_mix.experts.1.key.weight
blocks.0.channel_mix.experts.1.value.weight
blocks.0.channel_mix.experts.1.receptance.weight
blocks.0.channel_mix.experts.1.time_mix_k
blocks.0.channel_mix.experts.1.time_mix_r

blocks.0.channel_mix.gate.weight
```

The router supports top-k routing based on the configured expert density.

Example configuration:

```yaml
target_directory: "./SmaulNative-Merged"

base_model: "./SmaulNative-Base"

algorithm: "moe_upcycle"

density: 0.5
weight: 1.0

max_shard_size: "256MB"

seed: 42

addedexperts:
  - path: "./branches"
    weight: 1.0
```

Run:

```bash
python merge.py --config config.yaml
```

Branch directories are discovered in alphanumerical order when a parent branch directory is provided.

---

# 2. Transformer Variant

The `Transformer-Varient` directory contains an alternative model-development pipeline.

```text
Transformer-Varient/
├── Data.py
├── Merge.py
├── SyntheticData.py
├── Tokenizer.py
├── Train.py
└── download.py
```

This is a separate architecture path rather than simply an extension of the root RWKV-X model.

## `Transformer-Varient/Train.py`

Contains the training implementation for the Transformer variant.

This path is intended for experimentation with a Transformer-based architecture separately from the recurrent RWKV-X implementation.

## `Transformer-Varient/Merge.py`

Contains model merging functionality for the Transformer variant.

This allows the Transformer path to maintain its own merge implementation rather than requiring RWKV-specific parameter rules.

## `Transformer-Varient/SyntheticData.py`

Provides synthetic-data generation or processing functionality for the Transformer pipeline.

This is useful when experimenting with generated training examples, augmentation pipelines, or additional synthetic corpora.

## `Transformer-Varient/Data.py`

Handles data processing for the Transformer variant.

## `Transformer-Varient/Tokenizer.py`

Contains tokenizer functionality for the Transformer architecture path.

## `Transformer-Varient/download.py`

Handles dataset downloading for the Transformer pipeline.

---

# Typical Workflow

A typical SmaulNative experiment can follow this pipeline:

```text
Dataset Sources
      |
      v
download.py
      |
      v
Data.py
      |
      v
Tokenizer.py
      |
      v
train.py / Train.py
      |
      +--------------------+
      |                    |
      v                    v
RWKV-X Path          Transformer Path
      |                    |
      v                    v
Fine-tuned Branches  Fine-tuned Branches
      |                    |
      v                    v
merge.py             Merge.py
      |                    |
      v                    v
MoE / Merged Model   Merged Transformer Model
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
├── tokenizer_config.json
└── special_tokens_map.json
```

Large checkpoints may use multiple Safetensors shards:

```text
model/
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
└── model.safetensors.index.json
```

The RWKV-X merge pipeline is designed to process checkpoints in a memory-aware manner rather than loading every model tensor simultaneously.

---

# Design Goals

SmaulNative focuses on:

- Training language models from scratch.
- Supporting experimentation with multiple architectures.
- Keeping model tooling understandable and modifiable.
- Using custom training pipelines instead of hiding everything behind a single framework.
- Supporting resumable training and checkpointing.
- Supporting Safetensors and Hugging Face-style model formats.
- Supporting dataset and tokenizer pipelines alongside model code.
- Combining specialized checkpoints through model merging.
- Exploring Mixture of Experts architectures.
- Maintaining separate RWKV-style and Transformer-style development paths.

---

# Requirements

The exact dependencies depend on which pipeline is being used.

The root RWKV-X path commonly requires packages such as:

```text
torch
transformers
safetensors
tokenizers
datasets
pyarrow
tqdm
pyyaml
psutil
```

A typical installation is:

```bash
pip install torch transformers safetensors tokenizers datasets pyarrow tqdm pyyaml psutil
```

Additional dependencies may be required by individual scripts.

---

# Running the Project

The repository does not use a single universal command because it contains multiple independent tools.

Examples:

Train the root model:

```bash
python train.py
```

Merge RWKV-X branches:

```bash
python merge.py --config config.yaml
```

Run the Transformer training path:

```bash
python Transformer-Varient/Train.py
```

Run the Transformer merge path:

```bash
python Transformer-Varient/Merge.py
```

Train or prepare a tokenizer:

```bash
python Tokenizer.py
```

For the Transformer variant:

```bash
python Transformer-Varient/Tokenizer.py
```

The available command-line arguments depend on the individual script.

---

# Project Status

SmaulNative is an experimental and actively evolving project.

The repository contains multiple architecture implementations and independent tooling paths. Model formats, training behavior, merging logic, dataset pipelines, and configuration formats may change as development continues.

Compatibility between checkpoints depends on the model architecture and the version of the corresponding training or merging implementation.

---

# Contributing

Contributions, experiments, architecture improvements, training optimizations, dataset tooling improvements, and bug reports are welcome.

Because the project contains separate architecture paths, changes should clearly indicate whether they target:

- The root RWKV-X / recurrent implementation.
- The `Transformer-Varient` implementation.
- Shared data or tokenizer tooling.
- Model merging infrastructure.

---

# License

See the repository for the current license information.

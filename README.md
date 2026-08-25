# SmaulNative AI Engine

An end-to-end, memory-optimized framework for dataset streaming and cleaning, 2-bit Quantization-Aware Training (QAT) on low-resource hardware, custom BPE tokenization, synthetic dataset generation, and domain-expert Mixture of Experts (MoE) model merging.

---

## Key Features

- **PyArrow Accelerated Data Cleaning (`Data.py`)**: High-throughput C++ SIMD vectorized dataset and source code filtering supporting dataset formats (.parquet, .jsonl, .json, .csv) and source code (.py, .js, .cpp, .java, .rs, .go, etc.) with exact SHA256 and PyArrow string deduplication, PII redaction, and ZSTD compressed Parquet export.
- **Low-Resource Pre-Training (`Train.py`)**: CPU-tailored pre-training engine for ~256M Llama models featuring simulated 2-bit per-channel Quantization-Aware Training (QAT) with a Straight-Through Estimator (STE), custom FP32 Lion optimizer, sliding-window attention, and signal-safe checkpoint recovery.
- **1.8B Mixture of Experts Merger (`Merge.py`)**: Merges 10 specialized domain branches (cyber, math, code, tool_calling, thinking, defense, offense, biology, chemistry, quantum) into a 10-expert Top-2 MoE architecture using memory-efficient tensor-by-tensor safetensors streaming.
- **Byte-Level BPE Tokenizer (`Tokenizer.py`)**: Custom 128k vocabulary tokenizer trainer with multi-format batch streaming and round-trip verification suite.
- **Bilingual Synthetic Data Generator (`SyntheticData.py`)**: Generates bilingual (English and Hindi) instruction-response pairs with step-by-step reasoning (<think>...</think> blocks), math generators, and coding exercises.
- **Streaming Dataset Downloader (`download.py`)**: Resume-capable streaming reader supporting FineWeb-Edu, Wikipedia, and FinePDFs with byte caps and custom directory parameters.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| [`Data.py`](./Data.py) | PyArrow C++ accelerated dataset and source code filtering pipeline. |
| [`Train.py`](./Train.py) | Single-file 2-bit QAT CPU pre-training engine. |
| [`Merge.py`](./Merge.py) | Merges 10 domain checkpoints into a 1.8B Top-2 MoE model. |
| [`Tokenizer.py`](./Tokenizer.py) | Custom Byte-Level BPE tokenizer trainer and verifier. |
| [`SyntheticData.py`](./SyntheticData.py) | Bilingual synthetic data generator for English and Hindi. |
| [`download.py`](./download.py) | CLI streaming dataset downloader. |

---

## Quick Start

### 1. Download Datasets
```bash
python3 download.py --output-dir ./datasets --target-gb 20.0
```

### 2. Generate Synthetic Bilingual Data
```bash
python3 SyntheticData.py --count 5000 --format both
```

### 3. Clean and Deduplicate Data
```bash
python3 Data.py \
    --input ./datasets \
    --output ./cleaned_datasets \
    --workers 4
```

### 4. Train Tokenizer
```bash
python3 Tokenizer.py
```

### 5. Run Pre-Training (2-bit QAT)
```bash
python3 Train.py
```

### 6. Merge Domain Branches into MoE Model
```bash
python3 Merge.py
```

---

## License

MIT License.

## Authors

- **Code Author**: [Native](https://github.com/Native-Neo/)
- **README**: GEMINI 3.6 FLASH PROVIDED BY GOOGLE VERTEX AI (ANTIGRAVITY CLI)

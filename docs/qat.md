# qat.py

Quantization-Aware Training for RWKV-X at **2-bit (FP2), 4-bit (FP4), and 8-bit (FP8)** floating-point precision. It is normally used through `train.py`; the module also exposes helpers for preparing, calibrating, and converting a model.

## Run through train.py

```bash
# FP2
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT \
    --qt 2 --qat_calib_batches 64 --qat_export_dir ./RWKV-X-SFT-fp2

# FP4
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT \
    --qt 4 --qat_calib_batches 64 --qat_export_dir ./RWKV-X-SFT-fp4

# FP8
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT \
    --qt 8 --qat_calib_batches 64 --qat_export_dir ./RWKV-X-SFT-fp8
```

`--qt 2`, `--qt 4`, and `--qt 8` select FP2, FP4, and FP8. `qat.py` translates the `--qt N` compatibility option into the generic QAT mode before `train.py` parses its arguments. `--qat` remains available when the default QAT mode is sufficient.

- `--output_dir` keeps the normal fine-tunable checkpoint. QAT uses FP32 master weights and simulates the selected low-bit representation during the forward pass.
- `--qat_export_dir` is a separate conversion/export path. FP2 and FP4 weights are physically packed into sub-byte storage; FP8 weights use PyTorch's `float8_e4m3fn` representation where supported.
- Packed FP2/FP4 storage is primarily a memory/bandwidth optimization. It should not be assumed to outperform a tuned FP32 GEMM on CPUs without native low-bit floating-point instructions.

## Scripting it directly

```python
import qat
qat.prepare_qat(model, bits=4)
qat.calibrate(model, tokenizer, some_texts, ctx_len=512, device=device)
# ... fine-tune model as normal ...
qat.convert_qat(model)
```

## How it works

- **Floating-point levels**: FP2 and FP4 use compact custom floating-point codebooks represented by low-bit codes; FP8 uses PyTorch's `float8_e4m3fn`. FP2/FP4 are not IEEE interchange formats.
- **QAT training**: `FloatQATLinear` keeps the original FP32 parameter as the master weight and applies a straight-through fake-quantized weight in the forward pass. This avoids replacing optimizer state with 2/4-bit values during training.
- **Packed FP2/FP4 weights**: conversion encodes the quantized codes into bytes instead of storing one byte or one FP32 value per code. FP2 stores four codes per byte and FP4 stores two codes per byte.
- **CPU packed path**: converted FP2/FP4 weights use a chunked CPU matmul path. It decodes only the weight values needed for each input chunk before calling the normal CPU linear operation, avoiding full FP32 materialization of the packed matrix at once.
- **Current scope**: QAT wrapping currently targets the Channel-Mix `key` and `value` projections, including MoE experts. Time-Mix, MOBA attention, embeddings, and the model head are not automatically converted by `prepare_qat`.
- **Scale**: FP2/FP4 weight quantization uses per-input-column scaling. The packed representation stores the quantized codes separately from the scale values.
- **Calibration**: `qat.calibrate` runs representative forward passes without gradients or optimizer updates so the selected quantization ranges can settle before fine-tuning.
- **Conversion**: `qat.convert_qat` replaces prepared QAT linears with packed low-bit weight modules. The exported/converted model is intended for inference rather than continued QAT training.

## CPU notes

The packed FP2/FP4 implementation is CPU-portable and does not require fbgemm or qnnpack. On older CPUs such as Ivy Bridge, there is no native FP2/FP4 arithmetic, so the implementation decodes the compact representation and accumulates through normal floating-point CPU operations. Physical packing can reduce memory traffic, but a dedicated native low-bit GEMM kernel is required for a substantial compute-speed improvement.

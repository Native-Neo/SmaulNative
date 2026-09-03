# qat.py

Quantization-Aware Training for RWKV-X at **3-bit (int3)** precision. Not run directly -- it's
what `train.py --qat` uses. Only the Channel-Mix (FFN) `key`/`value` linears get fake-quantized;
embeddings, head, and both attention variants (RWKV time-mix, MOBA) stay FP32.

## Run through train.py

```bash
python train.py --mode sft --dataset_dir ./sft_data --output_dir ./RWKV-X-SFT \
    --qat --qat_calib_batches 64 --qat_export_dir ./RWKV-X-SFT-int3
```

- `--output_dir` keeps a **fake-quantized, still fine-tunable** checkpoint (FP32 storage, but
  every forward pass simulates 3-bit noise) -- keep training/resuming from this normally.
- `--qat_export_dir` is a **separate**, one-shot export: real packed int3 weights (~10.7x smaller
  than FP32 for those layers), not meant to be resumed/fine-tuned further.

## Scripting it directly

```python
import qat
qat.prepare_qat(model)                                   # wrap the FFN linears
qat.calibrate(model, tokenizer, some_texts, ctx_len=512, device=device)  # settle ranges
# ... fine-tune model as normal ...
qat.convert_qat(model)                                    # bake in real int3 weights
```

## How it works

- **Which linears**: every `RWKV_CMix_x070` FFN's `key` (expand, `C -> 4C`) and `value`
  (contract, `4C -> C`) projections are replaced, including each MoE expert
  (`_iter_cmix_modules`, `qat.py:107`). All are bias-free, so a bias-free `QATLinear` is an exact
  drop-in and the `state_dict` key stays `weight` (`qat.py:59`).
- **Ranges** (`qat.py:19`): weights are symmetric signed int3 in `[-4, 3]` per-channel; activations
  are asymmetric unsigned int3 in `[0, 7]` per-tensor. A `FakeQuantize` with a moving
  min/max observer wraps each (`_weight_fake_quant` / `_activation_fake_quant`).
- **Calibration** (`qat.calibrate`, `qat.py:145`): before fine-tuning, a handful of forward passes
  over a representative slice of the training data lets the observers' ranges settle
  (no gradients, no optimizer step). `train.py` pulls `--qat_calib_batches` worth from `--dataset_dir`.
- **Conversion** (`qat.convert_qat` -> `QATLinear.to_quantized`, `qat.py:77`): bakes the calibrated
  weight range into real int3 codes and packs them sub-byte (8 codes / 3 bytes, MSB-first) into a
  `QuantizedLinear` (`qat.py:86`). On each forward it unpacks, dequantizes `q * scale`, and runs a
  normal `F.linear` -- CPU-portable with no fbgemm/qnnpack dependency, but the on-the-fly unpacking
  costs some forward time (fine for this CPU-first project, not a packed-kernel inference engine).
- **`prepare_qat`** wraps whatever `nn.Linear` is present, so it's safe to call on a freshly
  constructed model *or* one loaded via `from_pretrained` -- loaded weights are preserved
  (`qat.py:117`).

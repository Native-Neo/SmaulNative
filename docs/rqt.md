# RQT

SmaulNative trains directly from quantized weights: every `FP8Linear` stores tiled E4M3
codes plus per-tile scales, with no FP32 master copy of the weight. Gradients are captured
from that quantized forward, and the packed storage is requantized in place after every
optimizer step. Optimizer momentum stays FP32 -- it needs more numerical resolution than
the stored weight.

## Storage

`fp8_tile.py` quantizes each `(out_f, in_f)` matrix in tiles of 64 columns (`TILE = 64`,
overridable per layer via `LinearConfig.tile` / `FP8Linear(..., tile)`):

- `w8`: `uint8` E4M3 codes, shape `(out_f, in_f)`.
- `sc`: `float32` per-tile scales, shape `(out_f, n_tiles)`, where each scale is the
  tile's `amax / 448.0` (`E4M3_MAX = 448.0`).
- Exact zeros quantize to code `0`, so sparsity patterns survive quantization.
- Subnormals are representable through the 256-entry E4M3 codebook.

`FP8Linear.from_float(nn.Linear)` converts an existing float layer; `err_stats()` reports
`amax_fp8` and `mean_scale` for inspecting quantization health.

## Forward / backward

The forward pass calls `compute.get_backend()` (default `cpu`, see [cpu.md](cpu.md)):

- With the native extension, tiled E4M3 forward accumulates from the codes in AVX.
- Without it, the torch fallback decodes one bounded tile block at a time -- the full
  FP32 weight matrix is never materialized.

For the backward pass, input gradients flow through the same backend path, while weight
gradients accumulate in FP32 into the module's `_gw` buffer (`g.T @ x`, summed over
micro-batches) and are consumed by the Lion step.

## Training step

`train.Lion` clips the global grad norm to `1.0`, applies a sign update, and calls
`requant(update, lr * wd)` on each FP8 module: decode the current tiles, apply weight
decay and the update in FP32, then re-quantize. `_gw` is cleared afterwards.

## Export

`convert_linear_to_gguf.py` dequantizes each tile at export time and writes plain
`f32`/`f16` tensors:

```bash
python convert_linear_to_gguf.py ./runs/linear ./runs/linear.gguf --dtype f16
```

The exported model is intended for inference rather than continued quantized training.

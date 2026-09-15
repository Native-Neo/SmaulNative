# Native GGUF tooling

SmaulNative has two native GGUF workflows:

- `smaul-convert-gguf`: Safetensors checkpoint -> GGUF.
- `smaul-quantize-gguf`: existing GGUF -> classic quantized GGUF.

## Conversion

```bash
./target/debug/smaul-native safetensors-gguf ./SmaulNative ./model.gguf --dtype f16
```

Supported source output dtypes are `f16` and `f32`.

## Quantization

```bash
./target/debug/smaul-native quantize-gguf model.gguf model-q4.gguf --type q4_0
```

Native classic formats currently supported:

- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`

Each classic quantizer uses 32-value blocks. Tensors whose size is not divisible by 32 are written as FP16 rather than being truncated or padded silently.

The quantizer can read the GGUF formats already decoded by SmaulNative, so an already-quantized source can be re-quantized through FP32 internally.

K-quants (`q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`) are recognized by the reader but are not yet emitted by the native quantizer. They require their 256-value super-block packing and scale/minimum encoding rather than the classic 32-value layout.

# Native GGUF tooling

SmaulNative has two native GGUF workflows:

- `smaul-convert-gguf`: Safetensors checkpoint -> GGUF.
- `smaul-quantize-gguf`: existing GGUF -> quantized GGUF.

## Conversion

```bash
./target/debug/smaul-native safetensors-gguf ./SmaulNative ./model.gguf --dtype f16
```

Supported source output dtypes are `f16` and `f32`.

## Quantization

```bash
./target/debug/smaul-native quantize-gguf model.gguf model-q4.gguf --type q4_0
./target/debug/smaul-native quantize-gguf model.gguf model-q3k.gguf --type q3_k
./target/debug/smaul-native quantize-gguf model.gguf model-q6k.gguf --type q6_k
```

Native classic formats currently supported:

- `q4_0`
- `q4_1`
- `q5_0`
- `q5_1`
- `q8_0`

Native K formats currently supported:

- `q2_k`
- `q3_k`
- `q4_k`
- `q5_k`
- `q6_k`

Classic formats use 32-value blocks. K formats use 256-value super-blocks and their GGML-compatible scale/quant packing. Tensors whose size is not divisible by the required block size are written as FP16 rather than being truncated or padded silently.

The quantizer reads tensor values through SmaulNative's GGUF decoder and requantizes through FP32. The K-format encoders emit the standard GGUF tensor type IDs and block layouts used by the corresponding GGML formats.

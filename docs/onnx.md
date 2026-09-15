# Native ONNX export

SmaulNative can package a Rust-loaded Safetensors checkpoint as an ONNX protobuf file.

```bash
./target/debug/smaul-native export-onnx ./SmaulNative ./model.onnx
```

The exporter stores F32/F16 tensors as ONNX initializers, exposes `tokens` as the model input, exposes `logits` as the output, and embeds the original `config.json` in the custom operator attributes.

The graph uses the custom operator:

```text
smaulnative::RWKVX
```

This is intentional. RWKV-X in SmaulNative includes recurrent WKV state, TimeMix, CMix, and optional MOBA routing. The current exporter does not pretend those operations are equivalent to a Transformer graph made only from ordinary ONNX operators.

Consequently, the exported file is a valid ONNX model representation, but an inference runtime must register the `smaulnative::RWKVX` operator to execute it. The next interoperability step is lowering the full RWKV-X/MOBA computation into standard ONNX operators and control-flow primitives so generic ONNX runtimes can execute it without a SmaulNative custom kernel.

Only F32 and F16 Safetensors tensors are accepted by the exporter. This prevents silently changing already-quantized weights during ONNX export.

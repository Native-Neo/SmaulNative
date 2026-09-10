# SmaulNative Audit

## Scope

Audited the `developement` branch after merging `main` on 2026-09-10. No data-analysis tooling was used.

## Current findings

### High: MoE still executes inactive experts on the full sequence
`RWKV_CMix_MoE.forward()` computes every active expert on the complete `x` tensor and only masks the result afterward. Routing is correct, but sparse MoE compute savings are largely lost. The existing `tests/test_moe.py` is already intended to compare a sparse implementation against the dense reference, so this is the next model-core optimization target.

### High: MOBA cached decoding reallocates the KV cache
The cached attention path repeatedly uses `torch.cat((pk, k), 2)` and the equivalent V operation for every generated token. This copies the growing cache repeatedly and becomes increasingly expensive with long contexts.

### Fixed: inference repeatedly decoded the entire generated sequence
`inference.py` previously decoded the whole generated token list on every token, creating unnecessary O(N²) decoding work. This was fixed in commit `11bff604485790e53a6e2aa53fd08ade661d4841` with an incremental decoder that preserves the tokenizer's `<cap>` and `<upper>` state.

### Verified resolved: QAT serialization metadata
Current `QuantizedLinear` registers `packed`, `scale`, and `weight_shape` as buffers, and `RWKVXModel.from_pretrained()` explicitly reconstructs quantized layers before strict state-dict loading. The earlier QAT serialization finding was stale after the merge.

### Verified resolved: bounded remote row-group submission
`stream_data.py` submits at most `workers * 2` row groups at a time rather than creating futures for the entire Parquet file. The earlier unbounded-submission finding was stale after the merge.

### Verified resolved: remote stream resume positioning
Remote streaming now carries dataset/file/record positions and passes them back into the streaming loader. The earlier claim that remote resume always restarts from the beginning was stale.

### Verified: compiled QAT export path
`train.py` unwraps `model._orig_mod` when saving and when exporting QAT, so the compiled wrapper is not passed directly to the QAT conversion/export path.

## Areas checked

- Native WKV CPU backend and regression tests
- Lion and NativeLion optimizer paths
- RWKV-X recurrent state handling
- MOBA full-sequence and cached attention paths
- MoE routing and existing MoE tests
- QAT conversion and checkpoint loading
- training checkpoint/resume logic
- local and remote dataset streaming
- inference sampling and generation
- tokenizer encode/decode paths

## Priority order

1. Sparse-dispatch MoE execution while preserving exact previous-token inputs.
2. Reduce MOBA KV-cache copying during cached decoding.
3. Add QAT save/load regression coverage.
4. Add an inference regression test for incremental decoding equivalence.
5. Continue auditing the native CPU kernels and optimizer implementations.

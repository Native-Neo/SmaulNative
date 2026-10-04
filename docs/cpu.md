# Compute backend

The model (`model.py`) and FP8 autograd (`kernel/compute.py`) never touch extensions
directly. They call into a backend selected with `kernel.compute.get_backend()`; future backends
register via `compute.register_backend()` without changing model code.

## Backend selection

```python
from kernel.compute import get_backend
be = get_backend()  # or get_backend("cpu")
be.configure(threads=2)
```

- The name defaults to the `SMAUL_BACKEND` environment variable, falling back to `cpu`.
- Unknown names raise `ValueError` -- there are no silent fake backends.

## CPU backend

`CpuBackend` (`kernel/compute.py`) is the default: fully functional and independently testable.

- `configure(threads)`: defaults to `SMAUL_CPU_THREADS`, else half the logical CPUs;
  sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and the torch thread counts.
- `has_native` / `has_attn_native` / `has_sparse_native`: whether each compiled extension
  loaded. Each is loaded lazily on first use, not at `configure()`.
- `fp8_forward` / `fp8_backward_input`: route `(x, w8-codes, scales)` through the native
  extension on CPU, else through the torch tiled fallback (output blocks of 64 rows,
  one tile at a time).
- `fp8_quantize_tiles`: the tiled E4M3 quantizer, or `None` to let the caller use the torch
  path. **Must be bit-exact with the torch path** -- the codes are the persistent weights, so
  a difference silently changes the model rather than raising. See `rqt.md`.
- `sparse_grad_v`: the fused `d/dvalues` of the Rawr `SparseLinear`.
- `attn_forward(..., need_state=)` / `attn_step`: the linear-attention recurrence and its
  single-token decode step. `S`/`z` are materialized **only** when `need_state=True`
  (they are `B*H*D*D*4` bytes of memset, wasted on every training forward otherwise);
  otherwise the returned state is a zero-element placeholder.
- `attn_backward`: the two-pass recurrence backward.

## Native extension

The extension (`smaul_fp8_ivb`, built from `kernel/fp8_cpu.cpp`) exposes `fp8_forward` and
`fp8_backward_input` over CPU `float32` activations, `uint8` E4M3 codes, and `float32`
scales. A second extension (`smaul_attn`, built from `kernel/attn_cpu.cpp` with the same
flags) exposes `attn_forward` / `attn_backward` for the linear-attention recurrence
(FP32 state, exact reference math, O(D^2) state, nothing sequence-sized stored). A third
extension (`smaul_sparse`, built from `kernel/sparse_cpu.cpp`) exposes `sparse_grad_v`,
the fused `d/dvalues` of the Rawr sparse projection: for every nonzero it does one
length-`rows` dot product with `x` kept cache-resident, instead of materialising the
`rows * out_f * K` gathered activation in torch (836 MiB for an 8000x512 head at
rows=512, K=51, and ~13x slower). A fourth extension (`smaul_quant`, built from
`kernel/quant_cpu.cpp`) exposes `fp8_quantize_tiles`. All four are compiled for Ivy Bridge-era CPUs (`-mavx -mf16c`, explicitly *without*
AVX2/AVX512) and loads lazily on first use; if compilation fails, a `RuntimeWarning`
is issued once and the torch fallback is used. Do not copy a built extension between
different CPU architectures.

## Benchmark

End-to-end training-step benchmark for the current pipeline:

```bash
python benchmark.py --mode full --d 512 --layers 4 --ctx 256 --batch 2 --iters 10 --threads 2
```

It prints FP8-vs-FP32 timings for linear forward/backward, attention, FFN, RMSNorms,
residual adds, and requant, plus full-step milliseconds, tokens/sec, RSS, and stored
FP8 vs FP32 size in MiB. Do not assume FP8 is faster; this script measures it, and on
Ivy Bridge (AVX1, no FMA) it measures FP8 as **1.9x slower forward and 2.2x slower
backward** than plain FP32 at d=512 -- the custom kernels sustain 7-13 GFLOP/s where MKL
SGEMM sustains 39-48. The weights are still a quarter the size; the compute is not cheaper.

`--arch` (default `rawr`), `--step-optimizer` (default `smaul`) and `--rawr-sparsity`
(default `0.9`) exist because `LinearConfig`/`train.py`'s code-level defaults differ from the
CLI's, and a benchmark that does not pass them measures a configuration nobody trains. These
defaults changed what the bare `python benchmark.py --mode full ...` command means; pass them
explicitly when comparing against an older run.

`--mode opt` compares Lion against SmaulOpt (state widths and factored vs full `v`) on
identical tensors, gradients, clip and thread count.

## Architecture benchmark

The Rawr/Plain x RAM/mmap experiment is in the same file:

```bash
python benchmark.py --mode arch --out ./runs/arch_bench --steps 8
```

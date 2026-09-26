# Compute backend

The model (`smaul_linear.py`) and FP8 autograd (`compute.py`) never touch extensions
directly. They call into a backend selected with `compute.get_backend()`; future backends
register via `compute.register_backend()` without changing model code.

## Backend selection

```python
from compute import get_backend
be = get_backend()  # or get_backend("cpu")
be.configure(threads=2)
```

- The name defaults to the `SMAUL_BACKEND` environment variable, falling back to `cpu`.
- Unknown names raise `ValueError` -- there are no silent fake backends.

## CPU backend

`CpuBackend` (`compute.py`) is the default: fully functional and independently testable.

- `configure(threads)`: defaults to `SMAUL_CPU_THREADS`, else half the logical CPUs;
  sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` and the torch thread counts.
- `has_native`: whether the compiled FP8 extension loaded.
- `fp8_forward` / `fp8_backward_input`: route `(x, w8-codes, scales)` through the native
  extension on CPU, else through the torch tiled fallback (output blocks of 64 rows,
  one tile at a time).

## Native extension

The extension (`smaul_fp8_ivb`, built from `fp8_cpu.cpp`) exposes `fp8_forward` and
`fp8_backward_input` over CPU `float32` activations, `uint8` E4M3 codes, and `float32`
scales. A second extension (`smaul_attn`, built from `attn_cpu.cpp` with the same
flags) exposes `attn_forward` / `attn_backward` for the linear-attention recurrence
(FP32 state, exact reference math, O(D^2) state, nothing sequence-sized stored). It is compiled for Ivy Bridge-era CPUs (`-mavx -mf16c`, explicitly *without*
AVX2/AVX512) and loads lazily on first use; if compilation fails, a `RuntimeWarning`
is issued once and the torch fallback is used. Do not copy a built extension between
different CPU architectures.

## Benchmark

End-to-end training-step benchmark for the current pipeline:

```bash
python cpu/benchmark.py --mode full --d 512 --layers 4 --ctx 256 --batch 2 --iters 10 --threads 2
```

It prints FP8-vs-FP32 timings for linear forward/backward, attention, FFN, RMSNorms,
residual adds, and requant, plus full-step milliseconds, tokens/sec, RSS, and stored
FP8 vs FP32 size in MiB. Do not assume FP8 is faster; this script measures it.

## Architecture benchmark

The Rawr/Plain x RAM/mmap experiment is also in `cpu/benchmark.py`:

```bash
python cpu/benchmark.py --mode arch --out ./runs/arch_bench --steps 8
```

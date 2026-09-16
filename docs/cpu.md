# Native CPU backend

The CPU path provides a native C++ WKV forward/backward kernel and a fused C++ Lion update. The WKV
kernel uses `at::parallel_for`; Lion uses AVX when supported by the host CPU. Both are compiled locally.

## Enable native WKV

The normal trainer uses the native backend with `--cpu`:

```bash
python train.py --cpu --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M
```

`--cpu` configures PyTorch CPU threading and replaces the Python/TorchScript WKV path with the native
C++ implementation. The native kernel supports head sizes up to 128.

## Thread configuration

`SMAUL_CPU_THREADS` controls the configured CPU thread count. For CPUs with HyperThreading, fewer
threads can be faster than using every logical CPU. Benchmark the value on your machine rather than
assuming that the logical-core count is optimal.

Example:

```bash
SMAUL_CPU_THREADS=2 python train.py --cpu --mode pretrain --dataset_dir ./datasets --output_dir ./RWKV-X-256M
```

## Benchmarks

For a kernel-only test:

```bash
python cpu/benchmark_cpu.py --threads 2 --size 10000000
```

For an end-to-end training-step benchmark:

```bash
python cpu/benchmark_full.py --ctx_len 512 --steps 3
```

The native WKV path parallelizes independent batch/head work while each recurrence remains sequential
across time. MOBA attention on CPU still uses causal scaled-dot-product attention and is O(T²), so a
smaller `--ctx_len` can have a large effect on training speed.

The kernel is intentionally compiled for the host CPU. Do not copy a built extension between different
CPU architectures.

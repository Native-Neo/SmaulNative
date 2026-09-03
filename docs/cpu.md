# Native CPU backend

The CPU path adds a fused C++ Lion update and a faster pretraining loop. The kernel is compiled locally with `-O3 -march=native`, so the generated code matches the host CPU instead of assuming AVX2/AVX-512.

Use the optimized trainer with:

```bash
python cpu/cpu_train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative --checkpoint_dir ./SmaulNative --tokenizer_path ./SmaulNative/tokenizer.json --optimizer lion
```

`SMAUL_CPU_THREADS` controls PyTorch/OpenMP CPU parallelism. On a 2-core/4-thread Ivy Bridge CPU, start with `SMAUL_CPU_THREADS=4`; benchmark 2 vs 4 if needed.

For a kernel-only test:

```bash
python cpu/benchmark_cpu.py --threads 4
```

The normal `train.py` remains unchanged. `cpu_train.py` swaps in the native Lion optimizer and removes the unnecessary batch tensor stack for batch size 1. Existing checkpoints and optimizer state remain compatible because the native optimizer uses the same Lion state and update equations.

The kernel is intentionally host-compiled with `-march=native`; do not copy a built extension between different CPU architectures.

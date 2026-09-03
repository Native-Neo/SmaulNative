# Native CPU backend

The CPU path adds a fused C++ Lion update and a faster pretraining loop. The kernel is compiled
locally with `-O3 -march=native -mavx`, using explicit AVX (256-bit) intrinsics for the Lion hot
loop (~5x faster than the old scalar loop on Ivy Bridge CPUs such as the i3-3220).

Use the optimized trainer with:

```bash
python cpu/cpu_train.py --mode pretrain --dataset_dir ./datasets --output_dir ./SmaulNative \
  --checkpoint_dir ./SmaulNative --tokenizer_path ./SmaulNative/tokenizer.json --optimizer lion
```

**Thread configuration for the i3-3220:**

`SMAUL_CPU_THREADS` controls PyTorch/OpenMP CPU parallelism. The default is now **physical core
count ÷ 2** (= 2 on a 4-thread i3-3220). Using all 4 HW threads (old default) caused an 8x
*slowdown* due to HyperThreading oversubscription with MKL. Use `SMAUL_CPU_THREADS=2`:

```bash
SMAUL_CPU_THREADS=2 python cpu/cpu_train.py --mode pretrain ...
```

Do **not** set `SMAUL_CPU_THREADS=4` on an i3-3220 — it is measured to be 8x slower.

For a kernel-only test:

```bash
python cpu/benchmark_cpu.py --threads 2 --size 10000000
```

For a full end-to-end training step benchmark:

```bash
python cpu/benchmark_full.py --ctx_len 512 --steps 3
```

The normal `train.py` remains unchanged. `cpu_train.py` swaps in the native AVX Lion optimizer
and removes the unnecessary batch tensor stack for batch size 1. Existing checkpoints and optimizer
state remain compatible because the native optimizer uses the same Lion state and update equations.

The kernel is intentionally host-compiled with `-march=native -mavx`; do not copy a built
extension between different CPU architectures.

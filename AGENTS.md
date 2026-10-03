# AGENTS.md

SmaulNative: a CPU-first FP8 ("RQT") language-model trainer. Flat root-level scripts, no
install step, no package to import-install. **Run every command from the repo root.**

## Commits

**Only one file should be in each commit!** One file per commit, no exceptions -- not even
for a one-line typo, a doc fix, or a mechanical rename that happens to touch three files in
the same directory. This is the single hard rule on this repository.

- Do not batch unrelated changes, and do not batch related ones.
- If a change genuinely cannot be expressed as one file per commit (a rename that must land
  with its call sites, a generated artifact), stop and ask before committing.
- Commit messages are prose describing *why*; the file list should be self-explanatory.
- Verify each commit leaves the tree working where that is possible:
  `python -m pytest -q` after anything touching a kernel, `smaul_linear.py`,
  `train.py` or `rawr_graph.py`.
- Nothing is pushed unless asked.

## Commands

```bash
python -m pip install -r requirements.txt
python -m pytest -q                              # full suite
python -m pytest tests/test_train.py -q         # one file
python -m pytest tests/test_smaul_opt.py::test_constructor_defaults -q   # one test
python benchmark.py --mode full|opt|arch --d 512 --layers 4 --ctx 256 --batch 2 --iters 10
```

- **No lint / format / typecheck config exists** (no ruff, mypy, black, flake8, pre-commit,
  or `pyproject.toml`). Verification is `pytest` only. Do not introduce a linter.
- `pytest -q` collects and passes clean -- 955 tests, no `--ignore` needed. It
  did not until 4d81541; if you see `--ignore=tests/test_rl.py` anywhere, that
  advice is stale.
- `datasets/` is gitignored and usually absent. `train.py --data` needs a real directory;
  `syntheticdata.py` can generate one. Every other path is relative to cwd.
- The first FP8 forward JIT-compiles two C++ extensions (`smaul_fp8_ivb`, `smaul_attn`) via
  `torch.utils.cpp_extension`, needing `ninja` + a C++ compiler. Without them everything
  still runs on the torch fallback with one `RuntimeWarning`. Call
  `get_backend().configure(n)` early (as `train.py` does) so thread settings apply.

## Architecture that filenames do not explain

- **FP8 weights are buffers, not parameters.** `FP8Linear` stores `w8` (uint8 E4M3 codes) +
  `sc` (fp32 per-tile scales). There is deliberately **no FP32 master weight**.
  `tests/test_linear_fp8.py::test_no_fp32_master_weights` enforces this — do not add one.
- **FP8 weight gradients accumulate in `module._gw`, not `.grad`**, and accumulate across
  backward calls. Any optimizer must call `zero_grad(model)` *with the model* or `_gw`
  double-counts. Without `model` it warns instead of failing.
- `fp8_modules(model)` returns only `FP8Linear` instances, so it is **empty** when
  `precision="fp32"`. Every optimizer needs a dense `p.grad` path as well.
- Blocking constants: `TILE = 64` (quantization tile columns) and `_OB = 64` (output rows).
  Quantize, requant, and optimizer blocking are all keyed to these. Tile-local updates are
  the norm — full-matrix FP32 transients are treated as a bug (memory blowup at 4k+).
- `architecture` `rawr` vs `plain` changes state_dict keys (`SparseLinear.values` vs
  `head.weight`). `from_pretrained` is `strict=True` and refuses mismatched architecture —
  intentional, do not loosen it.
- **The graph's sparsity is not the model's.** `RawrGraph.stats()` reports the fraction of the
  `vocab x vocab` token-edge space, and the graph only drives the **FFN and the LM head**. The
  attention q/k/v/o projections stay dense `FP8Linear`, and at the 32M preset they are 81% of
  the per-token MACs at `--rawr-sparsity 0.9` (98% at 0.99). So the graph prints 99.91% while
  the model is 63% sparse. `SmaulLinear.compute_profile()` walks the built model and reports the
  real split; `train.py` prints both. Never quote `stats()["sparsity"]` as model sparsity.
- `SparseLinear.cols` is a **non-persistent** buffer, so it is rebuilt from `rawr_graph.json` on
  load. `hidden_cols` must therefore stay byte-identical across versions — it is pinned against
  a verbatim copy of the original full-sort implementation in `tests/test_rawr.py`.
- `SparseLinear.forward`/`d/dx` are CSR **sparse products**, not gathers: exactly `out_f*K`
  nonzeros. The earlier gather was measured 23x slower than the dense GEMM it replaced.
- Inference decodes one token at a time: `SmaulLinear.prefill`/`step` carry the O(D^2)
  linear-attention state, and `inference.py` re-prefills whenever the window would slide
  (stepping then would be wrong — the state would still hold evicted tokens). **A per-token step
  is not bit-identical to a batched forward and cannot be**: torch picks a different GEMM/SpMM
  kernel for `[1,d]` than `[T,d]`, worth ~5e-07 relative. Test logits, not generated text.
- Model forward returns `(logits, loss)`; loss is cross-entropy with `ignore_index=-100`.
- `kernel/` has no `__init__.py` (implicit namespace package); imports are
  `from kernel.compute import get_backend`.

## Optimizer contract (`train.py`)

Any optimizer added here must satisfy this, because the training loop depends on it:

- `zero_grad(model=None)` — clears `p.grad` **and** every `FP8Linear._gw`.
- `step(model) -> float` — returns the pre-clip global grad norm, and **must return
  `float("inf")` when any gradient is non-finite**. The loop uses that to skip the step and
  count consecutive failures (50 in a row aborts training). See `Lion._clip`.
- `state_dict() -> dict` / `load_state_dict(dict)`.
- Gradient clipping lives **in the optimizer**, not the training loop. Weight decay is
  decoupled from the gradient update.
- Checkpoint state goes to JSON (hyperparams) + safetensors (tensors) via
  `_save_optimizer` / `_load_optimizer`. **Never `torch.save` a checkpoint** — the repo
  treats pickle load as arbitrary code execution and says so in `_save_optimizer`.

## Conventions

- **Tests have no `conftest.py`.** Every test file must begin with
  `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` or its imports fail.
  All 16 existing test files do; follow suit in new ones.
- Tests must not require `./datasets` or network. Prefer tiny `LinearConfig` models.
- `train.py` installs SIGINT/SIGTERM handlers **only inside `main()`** via
  `install_handlers()`, deliberately, so that importing `train` (which `benchmark.py`
  does) cannot hijack process signals. Preserve this.
- Validate inputs eagerly and raise `ValueError` at boundaries (constructors and CLI
  parsing both do this). The house style is fail-fast over silent clamping or repair.
- Docs are one page per entry point in `docs/`, indexed by `USEME.md`. Touching `train.py`
  or the FP8 path means updating `docs/train.md` and `docs/rqt.md`.

## Known breakage / drift (pre-existing — do not "fix" silently, and don't blame your change)

- `tests/test_rl.py` used to import `autorl`, which does not exist (folded into
  `rl.py` in 511d585), which is why plain `pytest -q` failed at collection. Fixed
  in 4d81541 -- a one-word import, nothing else. `rl.py` is still only 26%
  covered and three of that file's five tests assert torch algebra without
  calling `rl.py` at all, so do not read a green `test_rl.py` as RL coverage.
- `.github/workflows/test.yml` used to run `from compute import get_backend`, but
  `compute.py` lives at `kernel/compute.py`. Fixed: the step imports from
  `kernel.compute`, like the tests do.
- Docs used to reference files that do not exist: `infer_linear.py`, `autorl.py`,
  `cpu/benchmark_full.py`, `cpu/benchmark_arch.py` (benchmarks were consolidated into
  `benchmark.py` behind `--mode`). Fixed: `docs/quickstart.md` names the real entry
  points, and the stale `cpu/*.py` strings in `benchmark.py`/`train.py` are gone.
- **JIT lock hang:** a stale `~/.cache/torch_extensions/py*/smaul_fp8_ivb/lock` left by a
  killed process makes every FP8 test hang forever inside `file_baton.wait()`. Symptom is
  pytest producing *no output* on the first FP8 test. Fix: delete that lock file.
- Editing a `kernel/*.cpp` file forces a ~90 s recompile on the next test run, which looks like
  a hang. It is not.

## The FP8 quantizer is a correctness trap

`quantize_tiles` is on the requant path, so its output *is* the persisted weight. The native
kernel (`kernel/quant_cpu.cpp`) must match the torch path **exactly**. It takes the codebook
permutation and midpoints as arguments from `kernel.fp8_tile._tables` for precisely this
reason: E4M3 has two codes for +448 (126/127) and two for -448 (254/255), so a C++-re-derived
`std::sort` picks the other code for every weight saturating to exactly 448 — invisible in the
loss, but it silently rewrote stored weights once. Do not re-derive the codebook in C++.

It is also a known perf dead end, measured: 45 ns/element, 1.05x from 1->2 threads, and
`std::lower_bound` / branchless / 4-way / SIMD-count searches all land within noise of each
other. The only remaining lever is re-deriving the mapping — i.e. the bug above. Don't.

## Where the FP8 kernels' time actually goes

`benchmark.py` measures FP8 as ~1.9x slower forward and ~2.2x slower backward than plain
FP32 at d=512, and the linear attention projections (dense FP8 q/k/v/o) are ~44% of a
32M-preset step. Diagnosis, so it does not have to be repeated:

- **This class of CPU (family 6, model 58) has no FMA and no AVX2** — only
  `avx f16c sse4_2 xsave`. `-mno-avx2` in `_native_cflags` is correct, and an
  FMA/AVX2 code path would be *dead code here*; it could only help Haswell+. Do not
  add one expecting a local speedup.
- **It is not the accumulate loop.** Holding the forward's `acc` in 8 registers
  instead of a stack array measured 0.99x — no change, bit-identical.
- **It is not thread count.** `fp8_backward_input` sustains 13.4 GFLOP/s at 1, 2
  *and* 4 threads while MKL SGEMM goes 24.2 -> 46.8. `fp8_forward` goes
  17.0 -> 14.5 -> 21.7. A kernel that is flat in thread count is bound by
  something shared between cores (a cache level), not by arithmetic.
- **The prime suspect is the codebook expansion, not the math.** Each task
  re-decodes the weights (`lut[wp[...]] * sp[...]`) for every `MR=32` rows
  (forward) or `RB=16` rows (backward). For the forward that is ~32768 decodes per
  1.05M MACs, and the isolated accumulate loop runs at 35.8 GFLOP/s against the
  kernel's 17.9 — consistent with roughly half the time being decode. The
  backward is worse (3.9 GFLOP/s at 8000x512) and `RB == 16` is baked into its
  AVX fast path by the `reduce8` accumulator layout.

Two decode micro-optimizations were tried and **both failed**; don't repeat them:

1. Reordering the decode to walk each output row's 64 codes contiguously. The
   loads do become sequential, but the `wt` stores become 64 scattered one-float
   writes, and that costs more than the reads it saves: measured **2x slower**
   (29.5 ms vs 15.0 ms at 512x512x512).
2. Hoisting the 64 per-output scales into L1. No speedup, and the first attempt
   indexed them by `j` instead of `o2 + j` and produced wrong weights (3.7e-02
   relative error) — an easy mistake to repeat, since the naive version looks
   right and the tests do catch it.

The remaining lever is raising `MR` / `RB` (fewer weight decodes per row) or
restructuring the task loop so the decode is hoisted out of the row loop. Both
trade against parallelism and register pressure — `acc[32][64]` is already 8 KiB,
and the backward's `RB == 16` is fixed by `reduce8`. Treat that as a kernel
project, not an incremental fix, and budget for measurement noise: on this
2-core box under load, the *backward* source -- unchanged in all three runs --
measured 23.1, 30.1 and 771.6 ms for the same 512x512x512 shape, so a
sub-20% claim from three samples means nothing here. `benchmark.py` is the
only harness that has ever been quiet enough to compare against.

Note also that the PyTorch torch-fallback path is now covered by tests
(`tests/test_linear_fp8.py`), because it is what runs when these extensions
cannot be built at all.
- Per-step wall-clock timings on this machine swing 450–700 ms run to run and are
  dominated by forward/backward, not the optimizer. Trust byte/state counts over ms.

# AGENTS.md

SmaulNative: a CPU-first FP8 ("RQT") language-model trainer. Flat root-level scripts, no
install step, no package to import-install. **Run every command from the repo root.**

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
- `pytest -q` **currently fails at collection** on `tests/test_rl.py` (see Known breakage).
  For a green run: `python -m pytest -q --ignore=tests/test_rl.py`.
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
  All 15 existing tests do; follow suit in new ones.
- Tests must not require `./datasets` or network. Prefer tiny `LinearConfig` models.
- `train.py` installs SIGINT/SIGTERM handlers **only inside `main()`** via
  `install_handlers()`, deliberately, so that importing `train` (which `benchmark.py`
  does) cannot hijack process signals. Preserve this.
- Validate inputs eagerly and raise `ValueError` at boundaries (constructors and CLI
  parsing both do this). The house style is fail-fast over silent clamping or repair.
- Docs are one page per entry point in `docs/`, indexed by `USEME.md`. Touching `train.py`
  or the FP8 path means updating `docs/train.md` and `docs/rqt.md`.

## Known breakage / drift (pre-existing — do not "fix" silently, and don't blame your change)

- `tests/test_rl.py` imports `autorl`, which does not exist (folded into `rl.py` in
  511d585). This is why plain `pytest -q` fails on a clean tree.
- `.github/workflows/test.yml` runs `from compute import get_backend`, but `compute.py`
  lives at `kernel/compute.py`; that CI step ImportErrors.
- Docs still reference files that do not exist: `infer_linear.py`, `autorl.py`,
  `cpu/benchmark_full.py`, `cpu/benchmark_arch.py` (benchmarks were consolidated into
  `benchmark.py` behind `--mode`).
- **JIT lock hang:** a stale `~/.cache/torch_extensions/py*/smaul_fp8_ivb/lock` left by a
  killed process makes every FP8 test hang forever inside `file_baton.wait()`. Symptom is
  pytest producing *no output* on the first FP8 test. Fix: delete that lock file.
- Per-step wall-clock timings on this machine swing 450–700 ms run to run and are
  dominated by forward/backward, not the optimizer. Trust byte/state counts over ms.

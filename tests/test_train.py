import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib
import signal

import pytest
import torch
from kernel.fp8_tile import fp8_modules
from smaul_linear import LinearConfig, SmaulLinear
from train import Lion


def test_lion_step_keeps_fp8_storage_and_finite_loss():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=1, n_heads=2, tile=32)
    model = SmaulLinear(cfg)
    opt = Lion(list(model.parameters()), lr=1e-4)
    idx = torch.randint(0, 128, (2, 16))
    opt.zero_grad(model)
    _, loss = model(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step(model)
    for _, m in fp8_modules(model):
        assert m.w8.dtype == torch.uint8
    _, loss2 = model(idx, idx)
    assert torch.isfinite(loss2)


def test_checkpoint_round_trip_uses_same_names(tmp_path):
    torch.manual_seed(1)
    cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=1, n_heads=2, tile=32)
    model = SmaulLinear(cfg)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()
    restored = SmaulLinear.from_pretrained(tmp_path)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert k1 == k2 and torch.equal(v1.cpu(), v2.cpu())


# ---------------------------------------------------------------------------
# train.py's argument handling had no coverage at all (60% for the file, with
# main() and every parser helper unexercised), even though it is the primary
# entry point: a broken --preset or a validation rule that stopped firing would
# only show up as a confusing failure in a long run, or not at all.
#
# These drive the pure helpers directly rather than main(), which needs a dataset
# and a tokenizer build. main() is exercised by tests/test_rawr.py and by
# benchmark.py, which imports it.
# ---------------------------------------------------------------------------

def _args(**kw):
    """A namespace with the defaults main() would have parsed."""
    import argparse
    base = dict(vocab=8000, d=512, layers=8, heads=8, ffn_mult=2.5, batch=2,
                ctx=256, steps=1000, lr=2e-4, wd=0.01, threads=2, log_every=10,
                save_every=200, optimizer="smaul", architecture="rawr",
                embedding_storage="ram", rawr_sparsity=0.9, rawr_min_degree=4)
    base.update(kw)
    return argparse.Namespace(**base)


def test_validate_args_accepts_the_documented_defaults():
    from train import _validate_args
    _validate_args(_args())          # must not raise


@pytest.mark.parametrize("kw,frag", [
    (dict(batch=0), "batch must be positive"),
    (dict(ctx=-1), "ctx must be positive"),
    (dict(steps=0), "steps must be positive"),
    (dict(d=0), "d must be positive"),
    (dict(threads=0), "threads must be positive"),
    (dict(log_every=0), "log_every must be positive"),
    (dict(save_every=0), "save_every must be positive"),
    (dict(heads=7), "divisible by --heads"),
    (dict(lr=0.0), "lr looks invalid"),
    (dict(lr=20.0), "lr looks invalid"),
    (dict(wd=-0.1), "wd looks invalid"),
    (dict(wd=20.0), "wd looks invalid"),
    (dict(optimizer="adam"), "must be lion/smaul"),
    (dict(beta_m=1.0), "beta-m must be in [0.0, 1.0)"),
    (dict(beta_v=-0.1), "beta-v must be in [0.0, 1.0)"),
    (dict(beta_m=float("nan")), "beta-m must be finite"),
    (dict(epsilon=0.0), "epsilon must be positive finite"),
    (dict(ffn_mult=0.0), "ffn_mult must be positive finite"),
])
def test_validate_args_rejects_bad_input(kw, frag):
    from train import _validate_args
    with pytest.raises(ValueError) as e:
        _validate_args(_args(**kw))
    assert frag in str(e.value), (kw, str(e.value))


def test_apply_preset_sets_every_dimension_and_leaves_the_rest_alone():
    from train import apply_preset, estimate_params, list_presets
    a = _args(preset="32M", d=1, layers=1, heads=1, ffn_mult=1.0, vocab=1)
    apply_preset(a)
    p = list_presets()["32M"]
    assert (a.vocab, a.d, a.layers, a.heads, a.ffn_mult) == (
        p["vocab"], p["d"], p["layers"], p["heads"], p["ffn_mult"])
    assert a.d == 512 and a.layers == 8 and a.ffn_mult == 2.5
    # Untouched dimensions are not clobbered by a preset.
    assert a.batch == 2 and a.lr == 2e-4


def test_apply_preset_is_a_noop_without_one_and_names_the_alternatives():
    from train import apply_preset, list_presets
    a = _args(preset=None)
    apply_preset(a)
    assert a.d == 512
    with pytest.raises(ValueError) as e:
        apply_preset(_args(preset="not-a-preset"))
    msg = str(e.value)
    assert "not-a-preset" in msg and "--list-presets" in msg
    # The suggestion must actually list a real preset.
    assert any(name in msg for name in list_presets())


def test_estimate_params_is_monotone_and_sane():
    from train import PRESETS, estimate_params
    small = estimate_params(vocab=256, d=64, layers=2, ffn_mult=2.0)
    big = estimate_params(vocab=65536, d=2048, layers=18, ffn_mult=2.0)
    assert 0 < small < big
    # Doubling every dimension must grow the estimate, not stay flat.
    assert estimate_params(512, 128, 4, 2.0) > small
    # The named presets should land near their advertised order of magnitude.
    for name, p in PRESETS.items():
        est = estimate_params(p["vocab"], p["d"], p["layers"], p.get("ffn_mult", 2.5))
        assert est > 0, name


def test_dataset_fingerprint_is_order_independent_and_content_sensitive(tmp_path):
    from train import _dataset_fingerprint
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("alpha")
    b.write_text("beta")
    base = _dataset_fingerprint([a, b])
    assert base == _dataset_fingerprint([b, a]), "argument order changed the digest"
    assert base != _dataset_fingerprint([a]), "dropping a file did not change the digest"
    a.write_text("alpha changed")
    assert base != _dataset_fingerprint([a, b]), "editing a file did not change the digest"
    # A missing file must not raise: the caller degrades to "" on IO errors.
    assert isinstance(_dataset_fingerprint([a, tmp_path / "missing.txt"]), str)


def test_sha_file_matches_hashlib(tmp_path):
    import hashlib
    from train import _sha_file
    p = tmp_path / "blob.bin"
    p.write_bytes(b"smaul" * 1000)
    assert _sha_file(p) == hashlib.sha256(b"smaul" * 1000).hexdigest()


def test_install_handlers_is_only_called_from_main():
    """Importing train must not hijack process signals (benchmark.py imports it)."""
    import train
    before = signal.getsignal(signal.SIGINT)
    importlib.reload(train)
    assert signal.getsignal(signal.SIGINT) is before, \
        "importing train installed a SIGINT handler"
    train.install_handlers()
    assert signal.getsignal(signal.SIGINT) is not before
    signal.signal(signal.SIGINT, before)   # leave the process as we found it


def test_dense_block_size_is_a_performance_knob_not_a_semantic_one():
    """_DENSE_OB must not change the update, only how many blocks it takes.

    Blocks are independent, so this is free to tune; a test that pins the
    result makes it safe to tune.
    """
    from train import SmaulOpt

    def run(dense_ob):
        torch.manual_seed(0)
        SmaulOpt._DENSE_OB = dense_ob
        cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=1, n_heads=2,
                           architecture="rawr", rawr_sparsity=0.9)
        m = SmaulLinear(cfg)
        m.train()
        opt = SmaulOpt(list(m.parameters()), lr=2e-4)
        torch.manual_seed(1)
        idx = torch.randint(0, 128, (2, 16))
        for _ in range(2):
            opt.zero_grad(m)
            _, l = m(idx, idx)
            l.backward()
            opt.narrow_grads_(m)
            opt.step(m)
        return l.item(), {k: v.detach().clone() for k, v in m.state_dict().items()}

    prev = SmaulOpt._DENSE_OB
    try:
        la, sa = run(SmaulOpt._OB)             # the FP8-sized block
        lb, sb = run(SmaulOpt._DENSE_OB)       # the dense-sized block
        assert la == lb
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.equal(sa[k], sb[k]), k
        # And the FP8 requant block size must NOT have drifted: it is a
        # requantization granularity, not a cache hint.
        assert SmaulOpt._OB == 64
    finally:
        SmaulOpt._DENSE_OB = prev


# ---------------------------------------------------------------------------
# main() itself: the primary entry point, and the only place the CLI defaults,
# the Rawr graph build, the model/optimizer construction and the step loop meet.
# It was entirely uncovered (train.py sat at 60% with main() at zero), which is
# how a real run could regress without any test noticing -- the recovery pass
# added a print_model_compute call in here and there was no test for it either.
#
# These drive main() for a couple of steps against a temporary corpus. It is
# affordable: a tokenizer builds in ~2 ms on a tiny corpus, and the model is a
# 1-layer micro preset. No ./datasets and no network.
# ---------------------------------------------------------------------------

def _corpus(tmp_path, reps=60, extra=""):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.txt").write_text(
        "hello world this is a small english corpus for a smoke test. " * reps
        + extra, encoding="utf-8")
    return d


def _train_argv(out, data, extra=()):
    return ["train.py", "--data", str(data), "--out", str(out),
            "--d", "32", "--layers", "1", "--heads", "2", "--ffn_mult", "2.0",
            "--ctx", "32", "--batch", "2", "--steps", "2", "--log_every", "1",
            "--save_every", "1000", "--rawr-max-docs", "1",
            "--rawr-max-tokens-per-doc", "64", "--tok_records", "2000",
            "--vocab", "64", "--threads", "1", *extra]


def _run_main(monkeypatch, argv):
    import train
    monkeypatch.setattr("sys.argv", argv)
    train.STOP = False                 # a prior test may have tripped the handler
    train.main()


def test_main_trains_and_reports(tmp_path, monkeypatch, capsys):
    """A real 2-step run: Rawr by default, both numbers printed and distinct."""
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path)))
    printed = capsys.readouterr().out
    # The step line, with the live-gradient figure the recovery pass added.
    assert "stored " in printed and "live grads" in printed
    assert "[done]" in printed
    # The graph figure is labelled as the graph's, and the model's real
    # arithmetic is printed next to it. They must not be conflated, and at
    # --rawr-sparsity 0.9 the dense attention is most of the MACs.
    assert "graph_sparsity:" in printed and "NOT the model" in printed
    assert "dense share of MAC:" in printed
    assert "model_sparsity:" in printed
    share = float(printed.split("dense share of MAC:")[1].split("%")[0])
    assert 0.0 < share < 100.0
    # And the run actually produced a checkpoint.
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()


@pytest.mark.parametrize("arch", ["rawr", "plain"])
def test_main_both_architectures(tmp_path, monkeypatch, capsys, arch):
    """--architecture plain must not take the Rawr-only branches."""
    out = tmp_path / f"run_{arch}"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path),
                                      ("--architecture", arch)))
    printed = capsys.readouterr().out
    assert "[done]" in printed
    if arch == "plain":
        # No graph is built, so neither the graph nor the model-compute block.
        assert "graph_sparsity:" not in printed
        assert "dense share of MAC:" not in printed
        assert '"architecture": "plain"' in (out / "config.json").read_text()
    else:
        assert "graph_sparsity:" in printed
        assert (out / "rawr_graph.json").exists()


@pytest.mark.parametrize("opt", ["lion", "smaul"])
def test_main_both_optimizers(tmp_path, monkeypatch, capsys, opt):
    out = tmp_path / f"run_{opt}"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path),
                                      ("--optimizer", opt)))
    assert "[done]" in capsys.readouterr().out
    assert (out / "model.safetensors").exists()
    # SmaulOpt checkpoints its state; Lion is deliberately resume-free.
    assert (out / "optimizer.json").exists()
    if opt == "smaul":
        assert (out / "optimizer_state.safetensors").exists()


def test_main_saves_periodically_and_prints_a_save_line(tmp_path, monkeypatch, capsys):
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path),
                                      ("--steps", "1", "--save_every", "1")))
    printed = capsys.readouterr().out
    assert "[save]" in printed
    assert (out / "model.safetensors").exists()


def test_main_respects_rawr_sparsity_in_what_it_reports(tmp_path, monkeypatch, capsys):
    """The reported dense share must fall as --rawr-sparsity rises.

    Sparse layers shrink with the knob; the dense attention does not, so the
    dense share of per-token MAC goes *up*. If this ever inverts, the model
    sparsity claim is being computed over the wrong set of layers.
    """
    shares = []
    for sp in ("0.5", "0.9", "0.99"):
        out = tmp_path / f"run_{sp}"
        _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path),
                                          ("--rawr-sparsity", sp)))
        printed = capsys.readouterr().out
        shares.append(float(printed.split("dense share of MAC:")[1].split("%")[0]))
    assert shares[0] < shares[1] < shares[2], shares


def test_main_rejects_a_missing_data_dir(tmp_path, monkeypatch):
    with pytest.raises((ValueError, RuntimeError)) as e:
        _run_main(monkeypatch, _train_argv(tmp_path / "run", tmp_path / "nope"))
    assert "data" in str(e.value).lower()


def test_main_rejects_an_empty_data_dir(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        _run_main(monkeypatch, _train_argv(tmp_path / "run", empty))


# main() rejects bad input in two different ways, and both are fail-fast.
# _validate_args and the checks after apply_preset raise ValueError naming the
# flag; argparse's own type/choices reject at parse time with SystemExit(2) and
# a usage message. The distinction is worth pinning -- a traceback and a usage
# message are different things for a user to debug -- and so is the coverage:
# every one of these 18 was a real check in the source with nothing asserting it.
@pytest.mark.parametrize("flag,value", [
    ("--steps", "0"), ("--steps", "-3"),
    ("--d", "0"), ("--d", "-8"),
    ("--layers", "0"),
    ("--heads", "0"), ("--heads", "-1"),
    ("--ctx", "0"), ("--ctx", "-4"),
    ("--batch", "0"),
    ("--lr", "0"), ("--lr", "-1.0"), ("--lr", "nan"),
    ("--wd", "-1"), ("--wd", "nan"),
    ("--grad_clip", "0"),
    ("--epsilon", "0"), ("--epsilon", "inf"),
    ("--beta-m", "-1"), ("--beta-v", "-1"), ("--beta-m", "1.0"),
    ("--tok_records", "-1"),
    ("--threads", "0"),
    ("--rawr-sparsity", "-0.1"), ("--rawr-sparsity", "1.5"),
    ("--rawr-min-degree", "0"),
    ("--vocab", "0"),
])
def test_main_validates_numeric_arguments(tmp_path, monkeypatch, flag, value):
    """The ValueError names the flag, so the user knows which one to fix."""
    data = _corpus(tmp_path)
    argv = _train_argv(tmp_path / "run", data) + [flag, value]
    with pytest.raises(ValueError) as e:
        _run_main(monkeypatch, argv)
    # --rawr-sparsity and friends are reported by their argparse dest name, so
    # normalise the flag the same way; the rest already match verbatim.
    # Each check names the offending option, though not consistently: most spell
    # it the way the user typed it, while --beta-m reports its argparse dest
    # (beta_m). Both are accepted here. The real exception is
    # --rawr-min-degree, which is validated inside rawr_graph.build_graph and so
    # reports "min_degree must be >= 1" -- the concept rather than the spelling.
    # Still fail-fast and still correct; recorded here so the difference is
    # deliberate rather than surprising, and so a change in wording is noticed.
    dest = flag.lstrip("-").replace("-", "_")
    accepted = {dest, flag} | ({"min_degree"} if flag == "--rawr-min-degree" else set())
    assert any(a in str(e.value) for a in accepted), (flag, value, str(e.value))


@pytest.mark.parametrize("flag,value", [
    ("--optimizer", "adam"),          # not a house optimizer
    ("--architecture", "sparse"),
    ("--embedding-storage", "disk"),
    ("--precision", "bf16"),
    ("--state-dtype", "fp64"),
])
def test_main_rejects_out_of_range_choices_at_parse_time(tmp_path, monkeypatch, flag, value):
    """argparse choices fail at parse time with a usage error, not a traceback."""
    data = _corpus(tmp_path)
    argv = _train_argv(tmp_path / "run", data) + [flag, value]
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, argv)
    assert e.value.code == 2


def test_main_does_not_leave_a_partial_checkpoint_after_a_validation_error(tmp_path, monkeypatch):
    """A rejected run must not leave a directory that looks trained."""
    out = tmp_path / "run"
    with pytest.raises(ValueError):
        _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path)) + ["--lr", "0"])
    assert not (out / "model.safetensors").exists()
    assert not (out / "config.json").exists()
    assert not (out / "optimizer.json").exists()


def test_main_uses_the_preset_when_given(tmp_path, monkeypatch, capsys):
    """--preset must override the explicit dims on the command line."""
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path), ("--preset", "2K")))
    import json
    cfg = json.loads((out / "config.json").read_text())
    from train import PRESETS
    p = PRESETS["2K"]
    assert (cfg["d_model"], cfg["n_layer"], cfg["n_heads"]) == (p["d"], p["layers"], p["heads"])


# ---------------------------------------------------------------------------
# The diverged-run guard. Every step whose loss or gradient norm is non-finite
# is skipped and counted, and 50 in a row aborts. That is the only thing standing
# between a diverging run and an infinite loop writing NaN checkpoints, and none
# of it was exercised.
#
# There is no --resume flag in train.py, so the resume machinery in
# dataset.PretrainStream is unreachable from here. docs/train.md says so.
# ---------------------------------------------------------------------------

def _corpus(tmp_path, reps=60, extra=""):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.txt").write_text(
        "hello world this is a small english corpus for a smoke test. " * reps
        + extra, encoding="utf-8")
    return d


def _train_argv(out, data, extra=()):
    return ["train.py", "--data", str(data), "--out", str(out),
            "--d", "32", "--layers", "1", "--heads", "2", "--ffn_mult", "2.0",
            "--ctx", "32", "--batch", "2", "--steps", "2", "--log_every", "1",
            "--save_every", "1000", "--rawr-max-docs", "1",
            "--rawr-max-tokens-per-doc", "64", "--tok_records", "2000",
            "--vocab", "64", "--threads", "1", *extra]


def _run_main(monkeypatch, argv):
    import train
    monkeypatch.setattr("sys.argv", argv)
    train.STOP = False                 # a prior test may have tripped the handler
    train.main()


class _NaNLoss(SmaulLinear):
    """Real model, real step, but the loss is not finite."""
    bad = True

    def forward(self, x, y, **kw):
        logits, loss = super().forward(x, y, **kw)
        if self.bad:
            return logits, torch.tensor(float("nan"), requires_grad=True)
        return logits, loss


def _patch_model(monkeypatch, cls):
    monkeypatch.setattr("train.SmaulLinear", cls)


def test_a_non_finite_loss_step_is_skipped_and_counted(tmp_path, monkeypatch, capsys):
    _patch_model(monkeypatch, _NaNLoss)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path), ("--steps", "3",)))
    printed = capsys.readouterr().out
    assert "non-finite loss, skip step 0 (1 consecutive)" in printed
    # A skipped step does not count towards the step budget.
    assert "no steps completed" in printed


def test_fifty_consecutive_bad_losses_abort(tmp_path, monkeypatch, capsys):
    _patch_model(monkeypatch, _NaNLoss)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path), ("--steps", "500",)))
    printed = capsys.readouterr().out
    assert "50 consecutive non-finite losses; stopping to avoid infinite loop" in printed
    # It stopped at 50, not after running all 500.
    assert printed.count("non-finite loss, skip step") == 50, printed.count(
        "non-finite loss, skip step")


def test_fifty_consecutive_bad_grads_abort(tmp_path, monkeypatch, capsys):
    """The gradient branch, via the real seam.

    Both optimizers compute the pre-clip global norm in _clip() from the
    module-level _grad_norm, so making that return inf makes opt.step report inf
    while the loss, the forward and the backward all stay real. The previous
    attempt corrupted the gradients from inside forward(), which called
    backward() a second time and raised.
    """
    monkeypatch.setattr("train._grad_norm", lambda grads: float("inf"))
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, _corpus(tmp_path), ("--steps", "500",)))
    printed = capsys.readouterr().out
    assert "50 consecutive non-finite grads; stopping" in printed
    assert printed.count("non-finite grads, skip step") == 50


def test_the_counter_resets_after_a_good_step(tmp_path, monkeypatch, capsys):
    """Otherwise 50 slow bad steps spread over a long run abort a healthy one."""
    _patch_model(monkeypatch, _NaNLoss)

    data = _corpus(tmp_path)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, data, ("--steps", "6",)))
    first = capsys.readouterr().out
    assert "consecutive)" in first
    assert "1 consecutive" in first and "2 consecutive" in first

    _NaNLoss.bad = False                 # the run recovers
    try:
        _run_main(monkeypatch, _train_argv(out, data, ("--steps", "2",)))
    finally:
        _NaNLoss.bad = True
    assert "[done] steps=2" in capsys.readouterr().out


def test_a_run_where_every_step_fails_does_not_overwrite_the_checkpoint(tmp_path, monkeypatch, capsys):
    """An existing good checkpoint must survive a run that never completes."""
    _patch_model(monkeypatch, _NaNLoss)
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    # First a healthy run, to leave a checkpoint behind.
    _NaNLoss.bad = False
    try:
        _run_main(monkeypatch, _train_argv(out, data, ("--steps", "2",)))
    finally:
        _NaNLoss.bad = True
    capsys.readouterr()
    good = (out / "model.safetensors").read_bytes()

    _run_main(monkeypatch, _train_argv(out, data, ("--steps", "2",)))
    assert "no steps completed; checkpoint not overwritten" in capsys.readouterr().out
    assert (out / "model.safetensors").read_bytes() == good, "the good checkpoint changed"


# --- the CLI surfaces main() reaches before the loop -----------------------

def test_list_presets_prints_every_preset(monkeypatch, capsys):
    import train
    monkeypatch.setattr("sys.argv", ["train.py", "--list-presets"])
    train.main()
    printed = capsys.readouterr().out
    for name in train.PRESETS:
        assert name in printed
    assert "params" in printed


def test_tokenizer_is_reused_when_it_matches(monkeypatch, capsys, tmp_path):
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, data))
    tok = out / "tokenizer.json"
    first = tok.read_bytes()
    capsys.readouterr()
    _run_main(monkeypatch, _train_argv(out, data))
    assert tok.read_bytes() == first, "a matching tokenizer was rebuilt"


def test_a_mismatched_tokenizer_is_rebuilt(tmp_path, monkeypatch, capsys):
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, data))
    capsys.readouterr()
    # A tokenizer for a different vocab must not be used as-is.
    _run_main(monkeypatch, _train_argv(out, data, ("--vocab", "48")))
    assert "rebuilding" in capsys.readouterr().out


def test_a_corrupt_tokenizer_is_reported_clearly(tmp_path, monkeypatch):
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    tok = out / "tokenizer.json"
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text("{not json", encoding="utf-8")
    with pytest.raises((RuntimeError, ValueError)) as e:
        _run_main(monkeypatch, _train_argv(out, data))
    assert "tokenizer" in str(e.value).lower()


def test_an_explicit_tokenizer_path_is_used(tmp_path, monkeypatch):
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    elsewhere = tmp_path / "mytok.json"
    _run_main(monkeypatch, _train_argv(out, data, ("--tokenizer", str(elsewhere),)))
    assert elsewhere.exists(), "--tokenizer was ignored"


def test_the_config_records_what_the_model_was_trained_on(tmp_path, monkeypatch):
    """The tokenizer hash and dataset fingerprint are how a checkpoint says what
    it saw. A resumed-from or compared-against run needs them to be present."""
    import json
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, data))
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["tokenizer_sha256"], "no tokenizer hash recorded"
    assert cfg["dataset_fingerprint"], "no dataset fingerprint recorded"
    assert cfg["architecture"] in ("rawr", "plain")
    assert cfg["embedding_storage"] == "ram"


def test_mmap_embedding_storage_works_end_to_end(tmp_path, monkeypatch, capsys):
    import json
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    _run_main(monkeypatch, _train_argv(out, data,
                                       ("--embedding-storage", "mmap",)))
    assert (out / "embeddings.dat").exists(), "the mmap table was not created"
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["embedding_storage"] == "mmap"
    assert "[done]" in capsys.readouterr().out


def test_rawr_dict_and_graph_out_are_honoured(tmp_path, monkeypatch, capsys):
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    words = tmp_path / "words.txt"
    words.write_text("hello\n\nworld\n", encoding="utf-8")
    graph = tmp_path / "graph.json"
    _run_main(monkeypatch, _train_argv(out, data,
                                       ("--rawr-dict", str(words),
                                        "--rawr-graph-out", str(graph))))
    assert graph.exists(), "--rawr-graph-out was ignored"
    assert "graph digest=" in capsys.readouterr().out


def test_main_survives_an_unreadable_dataset_for_fingerprinting(tmp_path, monkeypatch):
    """A fingerprint failure must not abort the run; it degrades to ""."""
    data = _corpus(tmp_path)
    out = tmp_path / "run"
    import train as train_mod
    original = train_mod._dataset_fingerprint

    def boom(*a, **k):
        raise OSError("stat failed")

    monkeypatch.setattr(train_mod, "_dataset_fingerprint", boom)
    _run_main(monkeypatch, _train_argv(out, data))
    assert (out / "config.json").exists()
    monkeypatch.setattr(train_mod, "_dataset_fingerprint", original)

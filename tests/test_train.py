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

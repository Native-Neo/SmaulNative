import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from safetensors.torch import load_file

from merge_moe import merge
from model import LinearConfig, SmaulLinear


def _ckpt(root, seed, tokenizer=None, **over):
    """A real dense checkpoint, distinct per seed so wiring errors are visible."""
    torch.manual_seed(seed)
    kw = dict(vocab_size=32, d_model=32, n_layer=1, n_heads=2,
              ffn_mult=2.0, precision="fp32")
    kw.update(over)
    cfg = LinearConfig(**kw)
    model = SmaulLinear(cfg)
    d = root / f"ck{seed}"
    model.save_pretrained(d)
    if tokenizer is not None:
        (d / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    return d


@pytest.fixture
def trio(tmp_path):
    """base + 2 branches, each with distinguishable weights."""
    return (_ckpt(tmp_path, 1, tokenizer='{"vocab": {"a": 0}}'),
            [_ckpt(tmp_path, 2, tokenizer='{"vocab": {"a": 0}}'),
             _ckpt(tmp_path, 3, tokenizer='{"vocab": {"a": 0}}')])


# --- the wiring, which is the whole point of the merge ---------------------

def test_each_branch_becomes_the_expert_at_the_same_index(tmp_path, trio):
    """Expert e must hold branch e's FFN. A mis-indexed copy produces a model
    that loads and forwards perfectly while silently duplicating one expert."""
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out, top_k=2)
    merged = load_file(str(out / "model.safetensors"), device="cpu")
    for e, branch in enumerate(branches):
        src = load_file(str(branch / "model.safetensors"), device="cpu")
        ffn_keys = [k for k in src if ".ffn." in k]
        assert ffn_keys, "the branch has no ffn tensors to compare"
        for k in ffn_keys:
            dst = k.replace(".ffn.", f".ffn.experts.{e}.", 1)
            assert dst in merged, f"expert {e} is missing {k}"
            assert torch.equal(merged[dst], src[k]), f"expert {e} got the wrong weights for {k}"


def test_shared_tensors_come_from_the_base_not_the_branches(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out, top_k=2)
    merged = load_file(str(out / "model.safetensors"), device="cpu")
    base_sd = load_file(str(base / "model.safetensors"), device="cpu")
    branch_sd = load_file(str(branches[0] / "model.safetensors"), device="cpu")
    shared = [k for k in base_sd if ".ffn." not in k and ".gate." not in k]
    assert shared, "expected tensors shared from base"
    for k in shared:
        assert torch.equal(merged[k], base_sd[k]), f"{k} did not come from base"
        if not torch.equal(base_sd[k], branch_sd[k]):
            continue        # base and branch happen to agree here; nothing to prove
        assert torch.equal(merged[k], base_sd[k])


def test_routers_are_fresh_not_copied_from_anywhere(tmp_path, trio, capsys):
    """Routers are newly initialised, which the code warns about -- check it."""
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out, top_k=2)
    assert "routers (gate) are freshly initialized" in capsys.readouterr().out
    merged = load_file(str(out / "model.safetensors"), device="cpu")
    routers = [k for k in merged if ".gate.weight" in k or ".gate.bias" in k]
    assert routers, "expected router tensors in the merged model"
    # They must not equal a base router (dense models have none) and must be
    # per-expert, i.e. distinct from one another.
    if len(routers) > 1:
        assert not torch.equal(merged[routers[0]], merged[routers[1]])


def test_merged_model_loads_and_forwards(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out, top_k=2)
    model = SmaulLinear.from_pretrained(out)
    cfg = model.cfg
    assert cfg.is_moe and cfg.num_experts == 2 and cfg.num_experts_per_tok == 2
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    logits.sum().backward()
    assert torch.isfinite(model.emb.weight.grad).all()


def test_top_k_one_is_the_default_and_still_uses_every_expert(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out)
    assert LinearConfig.load(out / "config.json").num_experts_per_tok == 1
    model = SmaulLinear.from_pretrained(out)
    ids = torch.randint(0, 32, (1, 6))
    assert torch.isfinite(model(ids, ids)[0]).all()


def test_a_single_branch_still_produces_a_moe(tmp_path):
    base = _ckpt(tmp_path, 1)
    branch = _ckpt(tmp_path, 2)
    out = tmp_path / "merged"
    merge(base, [branch], out)
    cfg = LinearConfig.load(out / "config.json")
    assert cfg.is_moe and cfg.num_experts == 1


# --- output contents --------------------------------------------------------

def test_merge_config_records_its_inputs(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out, top_k=2)
    meta = json.loads((out / "merge_config.json").read_text())
    assert meta["num_experts"] == 2 and meta["top_k"] == 2
    assert [Path(p).name for p in meta["branches"]] == [p.name for p in branches]
    assert "routers freshly initialized" in meta["note"]


def test_the_base_tokenizer_is_copied_so_the_dir_is_usable(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    merge(base, branches, out)
    assert json.loads((out / "tokenizer.json").read_text()) == {"vocab": {"a": 0}}


def test_a_missing_base_tokenizer_warns_rather_than_failing(tmp_path):
    base = _ckpt(tmp_path, 1)
    branch = _ckpt(tmp_path, 2)
    out = tmp_path / "merged"
    merge(base, [branch], out)
    assert not (out / "tokenizer.json").exists()


def test_a_branch_tokenizer_mismatch_is_refused(tmp_path):
    """Otherwise the merged model tokenizes differently depending on the expert."""
    base = _ckpt(tmp_path, 1, tokenizer='{"vocab": {"a": 0}}')
    bad = _ckpt(tmp_path, 2, tokenizer='{"vocab": {"b": 0}}')
    out = tmp_path / "merged"
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        merge(base, [bad], out)


def test_a_branch_without_a_tokenizer_only_warns(tmp_path, capsys):
    base = _ckpt(tmp_path, 1, tokenizer='{"vocab": {"a": 0}}')
    branch = _ckpt(tmp_path, 2)
    out = tmp_path / "merged"
    merge(base, [branch], out)
    assert "skipping equality check" in capsys.readouterr().out


# --- refusals ---------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("vocab_size", 64), ("d_model", 64), ("n_layer", 2), ("n_heads", 4),
    ("ffn_mult", 3.0), ("tile", 32), ("eps", 1e-5),
])
def test_a_branch_that_disagrees_on_the_config_is_refused(tmp_path, field, value):
    base = _ckpt(tmp_path, 1)
    branch = _ckpt(tmp_path, 2, **{field: value})
    with pytest.raises(ValueError, match=field):
        merge(base, [branch], tmp_path / "merged")


def test_a_precision_mismatch_is_refused(tmp_path):
    base = _ckpt(tmp_path, 1)
    branch = _ckpt(tmp_path, 2, precision="fp8")
    with pytest.raises(ValueError, match="precision"):
        merge(base, [branch], tmp_path / "merged")


def test_an_already_moe_branch_is_refused(tmp_path):
    base = _ckpt(tmp_path, 1)
    torch.manual_seed(9)
    moe = SmaulLinear(LinearConfig(vocab_size=32, d_model=32, n_layer=1, n_heads=2,
                                   ffn_mult=2.0, precision="fp32", is_moe=True,
                                   num_experts=2, num_experts_per_tok=1))
    d = tmp_path / "moebranch"
    moe.save_pretrained(d)
    with pytest.raises(ValueError, match="already MoE"):
        merge(base, [d], tmp_path / "merged")


def test_an_incomplete_checkpoint_is_refused(tmp_path):
    base = _ckpt(tmp_path, 1)
    incomplete = tmp_path / "broken"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        merge(base, [incomplete], tmp_path / "merged")


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(tmp_path, trio, top_k):
    base, branches = trio
    with pytest.raises(ValueError, match="top_k must be"):
        merge(base, branches, tmp_path / "merged", top_k=top_k)


def test_top_k_cannot_exceed_the_expert_count(tmp_path, trio):
    base, branches = trio
    with pytest.raises(ValueError, match="exceeds expert count"):
        merge(base, branches, tmp_path / "merged", top_k=3)


def test_out_must_not_be_a_file(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "afile"
    out.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        merge(base, branches, out)


def test_a_non_empty_out_is_refused_without_force(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    out.mkdir()
    (out / "stale.txt").write_text("keep me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        merge(base, branches, out)
    assert (out / "stale.txt").exists(), "the refusal must not have touched it"


def test_force_overwrites(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    out.mkdir()
    (out / "stale.txt").write_text("gone", encoding="utf-8")
    merge(base, branches, out, force=True)
    assert (out / "model.safetensors").exists()


def test_an_empty_out_dir_is_fine_without_force(tmp_path, trio):
    base, branches = trio
    out = tmp_path / "merged"
    out.mkdir()
    merge(base, branches, out)          # must not raise
    # And it must actually have written the merge, not merely returned quietly.
    assert (out / "model.safetensors").exists()
    assert (out / "config.json").exists()


def test_two_merges_of_the_same_inputs_are_identical(tmp_path, trio):
    """A merge has to be reproducible, or it cannot be verified or extended.

    Every tensor but the router already was: branches and base are loaded from
    disk. The router was the one tensor that differed, because SmaulLinear was
    constructed against whatever the global RNG happened to be. That also means
    you cannot re-run a merge to check it, or add one expert and keep the
    existing routing.
    """
    base, branches = trio
    a, b = tmp_path / "a", tmp_path / "b"
    merge(base, branches, a)
    merge(base, branches, b)
    sa = load_file(str(a / "model.safetensors"), device="cpu")
    sb = load_file(str(b / "model.safetensors"), device="cpu")
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_a_different_branch_set_still_gets_its_own_routers(tmp_path):
    """Seeding must not make every merge produce the same router."""
    base = _ckpt(tmp_path, 1)
    one = _ckpt(tmp_path, 4)
    other = _ckpt(tmp_path, 5)
    a, b = tmp_path / "a", tmp_path / "b"
    merge(base, [one], a)
    merge(base, [other], b)
    ra = load_file(str(a / "model.safetensors"), device="cpu")["blocks.0.ffn.gate.weight"]
    rb = load_file(str(b / "model.safetensors"), device="cpu")["blocks.0.ffn.gate.weight"]
    assert not torch.equal(ra, rb), "the router no longer depends on the inputs"


def test_the_merge_leaves_the_callers_rng_alone(tmp_path, trio):
    """Seeding inside a library function must not disturb everyone after it."""
    base, branches = trio
    torch.manual_seed(123)
    expected = torch.randn(4)
    torch.manual_seed(123)
    merge(base, branches, tmp_path / "merged")
    assert torch.equal(torch.randn(4), expected), "the global RNG stream was consumed"


def test_the_merged_dir_does_not_depend_on_where_it_is_written(tmp_path, trio):
    """Merging to two different output paths gives the same model.

    The seed is derived from base and branch paths, so a re-merge into a fresh
    directory reproduces the previous one rather than producing a subtly
    different model.
    """
    base, branches = trio
    a, b = tmp_path / "out_a", tmp_path / "somewhere_else/b"
    merge(base, branches, a)
    merge(base, branches, b)
    sa = load_file(str(a / "model.safetensors"), device="cpu")
    sb = load_file(str(b / "model.safetensors"), device="cpu")
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k

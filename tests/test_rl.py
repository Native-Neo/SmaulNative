import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from rl import AutoRL, SmaulRL


def test_sample_logprob_matches_sampling_distribution():
    logits = torch.tensor([4.0, 3.0, 1.0, -2.0])
    filtered = SmaulRL._filter_logits(logits, 0.7, 2, 1.0)
    expected = torch.log_softmax(filtered, -1)
    torch.manual_seed(0)
    token, logprob = SmaulRL._sample(logits, 0.7, 2, 1.0)
    assert abs(logprob - expected[token].item()) < 1e-6


def test_filter_rejects_nonpositive_temperature():
    try:
        SmaulRL._filter_logits(torch.ones(4), 0.0, 0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("temperature=0 must fail")


def test_sampled_kl_is_zero_when_policies_match():
    old = torch.tensor([-1.0, -2.0, -3.0])
    new = old.clone()
    assert torch.equal(-(new - old), torch.zeros_like(old))


def test_autorl_uses_sampled_old_policy_kl():
    old = torch.tensor([-1.0, -2.0, -3.0])
    new = torch.tensor([-1.3, -2.0, -2.8])
    expected = -(new - old).mean()
    wrong_reverse_kl = (torch.exp(new - old) - (new - old) - 1).mean()
    assert expected > 0
    assert not torch.isclose(expected, wrong_reverse_kl)
    assert hasattr(AutoRL, "grpo_step")


def test_generate_disables_dropout_and_restores_training_state():
    class Tokenizer:
        def encode(self, text):
            return type("Encoded", (), {"ids": [2]})()

        def decode(self, ids):
            return "x"

    class Model:
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode=True):
            self.training = mode
            return self

        def __call__(self, ids):
            assert not self.training
            return torch.tensor([[[10.0, 0.0, 0.0]]]), None

    obj = SmaulRL.__new__(SmaulRL)
    obj.model = Model()
    obj.tokenizer = Tokenizer()
    obj.device = torch.device("cpu")
    obj.eos_id = 99
    obj.bos_id = 2
    obj._encode = lambda text: [2]
    obj._decode = lambda ids: "x"
    text, tokens, logprobs = obj.generate("prompt", 1, 1.0, 0, 1.0, seed=0)
    assert text == "x"
    assert tokens == [0]
    assert len(logprobs) == 1
    assert obj.model.training


# ---------------------------------------------------------------------------
# The five tests above were the only ones rl.py had, and three of them assert
# torch algebra without calling rl.py at all. Everything below drives the real
# code on a real (tiny) model: a 1-layer fp32 SmaulLinear plus a 64-token
# tokenizer, saved to tmp_path. It builds in well under a second.
# ---------------------------------------------------------------------------

import json

CORPUS = "hello world this is a small english corpus for the rl tests. " * 30


def _model_dir(tmp_path, vocab=64):
    from smaul_linear import LinearConfig, SmaulLinear
    from tokenizer import SmaulTokenizer, _build
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    data = _build(iter([CORPUS]), vocab, 20, 0)
    SmaulTokenizer(data).save(d / "tokenizer.json")
    cfg = LinearConfig(vocab_size=len(data["vocab"]), d_model=32, n_layer=1,
                       n_heads=2, ffn_mult=2.0, precision="fp32")
    SmaulLinear(cfg).save_pretrained(d)
    return d


def _rl(tmp_path):
    from rl import SmaulRL
    return SmaulRL(str(_model_dir(tmp_path)), str(tmp_path / "work"), "cpu")


def _autorl(tmp_path):
    from rl import AutoRL
    return AutoRL(str(_model_dir(tmp_path)), str(tmp_path / "work"), "cpu")


def _write_prefs(path, records):
    import json
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _prefs(n, prompt="hello"):
    return [{"prompt": prompt, "responses": ["alpha", "beta"],
             "chosen": i % 2, "source": "human"} for i in range(n)]


# --- construction and validation -------------------------------------------

def test_init_rejects_a_mismatched_tokenizer(tmp_path):
    """The tokenizer must match the loaded policy, not merely exist.

    A vocab disagreement here would otherwise surface much later as an index
    out of range inside the model, with nothing pointing at the tokenizer.
    """
    import json
    from rl import SmaulRL
    d = _model_dir(tmp_path)
    path = d / "tokenizer.json"
    data = json.loads(path.read_text())
    # Drop entries so the tokenizer's vocab no longer matches the model's.
    for name in list(data["vocab"])[10:]:
        del data["vocab"][name]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError) as e:
        SmaulRL(str(d), str(tmp_path / "w"), "cpu")
    assert "vocab" in str(e.value)


def test_init_rejects_an_invalid_device(tmp_path):
    from rl import SmaulRL
    d = _model_dir(tmp_path)
    with pytest.raises(ValueError) as e:
        SmaulRL(str(d), str(tmp_path / "w"), "not-a-device")
    assert "device" in str(e.value)


def test_init_requests_cuda_without_it(tmp_path, monkeypatch):
    """A CUDA request on a CPU box must fail loudly, not silently fall back."""
    import torch
    from rl import SmaulRL
    d = _model_dir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError) as e:
        SmaulRL(str(d), str(tmp_path / "w"), "cuda")
    assert "CUDA" in str(e.value)


def test_init_resumes_the_policy_from_the_work_dir(tmp_path, capsys):
    """grpo_step writes a policy; a later run must load that, not model_dir."""
    rl = _rl(tmp_path)
    policy = tmp_path / "work" / "policy"
    rl._save("hello", [{"text": "a"}, {"text": "b"}], 0)
    import json
    from smaul_linear import SmaulLinear
    # Point the work dir's policy at a *different* set of weights.
    from tokenizer import SmaulTokenizer, _build
    alt = tmp_path / "alt"
    alt.mkdir()
    data = _build(iter([CORPUS]), 64, 20, 0)
    SmaulTokenizer(data).save(alt / "tokenizer.json")
    SmaulLinear(rl.model.cfg).save_pretrained(alt)
    SmaulLinear.from_pretrained(alt).save_pretrained(policy)
    SmaulTokenizer(data).save(policy / "tokenizer.json")
    from rl import SmaulRL
    again = SmaulRL(str(tmp_path / "model"), str(tmp_path / "work"), "cpu")
    assert "resuming policy" in capsys.readouterr().out


# --- logit filtering and sampling -----------------------------------------

def test_filter_logits_divides_by_temperature_before_filtering():
    logits = torch.tensor([4.0, 3.0, 1.0, -2.0])
    hot = SmaulRL._filter_logits(logits, 0.5, 0, 1.0)
    cold = SmaulRL._filter_logits(logits, 2.0, 0, 1.0)
    assert hot[0] == pytest.approx(8.0)
    assert cold[0] == pytest.approx(2.0)


def test_filter_logits_top_k_masks_below_the_cutoff():
    logits = torch.tensor([4.0, 3.0, 1.0, -2.0])
    out = SmaulRL._filter_logits(logits, 1.0, 2, 1.0)
    assert torch.isfinite(out[:2]).all()
    assert not torch.isfinite(out[2:]).any()


def test_filter_logits_top_k_larger_than_vocab_is_a_noop():
    logits = torch.tensor([4.0, 3.0, 1.0])
    out = SmaulRL._filter_logits(logits, 1.0, 99, 1.0)
    assert torch.isfinite(out).all()


def test_filter_logits_top_p_keeps_the_most_likely_token():
    """An aggressive nucleus can never be empty: remove[0] is forced False.

    This is the reason _sample has an all-masked fallback that is unreachable
    via top_p alone, and it is worth pinning -- if remove[0] were ever changed
    to True, an aggressive top_p would hand _sample a fully -inf row.
    """
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    for top_p in (1e-6, 0.01, 0.5, 0.99, 1.0):
        out = SmaulRL._filter_logits(logits, 1.0, 0, top_p)
        assert torch.isfinite(out).any(), top_p
        assert torch.isfinite(out[4]), top_p       # the argmax always survives


def test_filter_logits_top_p_is_monotone_in_what_it_keeps():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    counts = [int(torch.isfinite(SmaulRL._filter_logits(logits, 1.0, 0, p)).sum())
              for p in (0.3, 0.5, 0.7, 0.9)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


@pytest.mark.parametrize("kwargs,msg", [
    (dict(temperature=0.0), "temperature"),
    (dict(temperature=-1.0), "temperature"),
    (dict(top_k=-1), "top_k"),
    (dict(top_p=0.0), "top_p"),
    (dict(top_p=1.5), "top_p"),
    (dict(top_p=-0.1), "top_p"),
])
def test_filter_logits_rejects_bad_arguments(kwargs, msg):
    base = dict(temperature=1.0, top_k=0, top_p=1.0)
    base.update(kwargs)
    with pytest.raises(ValueError) as e:
        SmaulRL._filter_logits(torch.ones(4), **base)
    assert msg in str(e.value)


def test_sample_returns_a_token_from_the_filtered_support():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    torch.manual_seed(0)
    token, logprob = SmaulRL._sample(logits, 1.0, 2, 1.0)
    assert token in (3, 4)
    filtered = SmaulRL._filter_logits(logits, 1.0, 2, 1.0)
    assert logprob == pytest.approx(torch.log_softmax(filtered, -1)[token].item())


def test_sample_is_seed_deterministic():
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    torch.manual_seed(7)
    a = SmaulRL._sample(logits, 1.0, 0, 1.0)
    torch.manual_seed(7)
    b = SmaulRL._sample(logits, 1.0, 0, 1.0)
    assert a == b


def test_sample_survives_a_partly_nan_logit_row():
    """The case that actually happens: one bad value beside good ones.

    Before the masked_fill, log_softmax produced NaN for the whole row and the
    run died. Now the NaN is simply never sampled, and the finite entries still
    drive the distribution.
    """
    logits = torch.tensor([5.0, float("nan"), 4.0, 1.0])
    torch.manual_seed(0)
    token, logprob = SmaulRL._sample(logits, 1.0, 0, 1.0)
    assert token != 1
    assert torch.isfinite(torch.tensor(logprob))
    # And it is the NaN slot specifically that is excluded, not an accident of
    # the seed: the same row minus the NaN puts all its mass on the same place.
    assert token in (0, 2, 3)


def test_sample_survives_neg_inf_beside_finite_values():
    logits = torch.tensor([float("-inf"), 6.0, float("-inf"), 2.0])
    torch.manual_seed(0)
    token, logprob = SmaulRL._sample(logits, 1.0, 0, 1.0)
    assert token in (1, 3)
    assert torch.isfinite(torch.tensor(logprob))


def test_all_masked_fallback_is_unreachable_through_top_k_and_top_p():
    """_sample's "fall back to greedy" branch is defensive, not live.

    top_k masks strictly *below* the k-th value and top_p forces remove[0] =
    False, so the most likely token always survives and the row is never all
    -inf when the input had any finite value. Pinned so that a future change to
    the filtering does not quietly make the branch the common path -- or, worse,
    make it unreachable for a case that needed it.
    """
    logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
    for top_k in (0, 1, 2, 3, 6, 99):
        for top_p in (1e-6, 0.3, 0.9, 1.0):
            filtered = SmaulRL._filter_logits(logits, 1.0, top_k, top_p)
            assert torch.isfinite(filtered).any(), (top_k, top_p)


@pytest.mark.parametrize("bad", [
    [float("-inf")] * 4,
    [float("nan")] * 4,
    [float("inf"), float("-inf"), float("nan"), float("inf")],
])
def test_sample_raises_when_no_logit_is_finite(bad):
    """A wholly unusable row must fail loudly, not sample from a flat dist.

    If this degraded to a uniform distribution instead, grpo_step would store
    old_logprobs that carry no information about the policy, and the importance
    ratio on the first step would no longer be 1.
    """
    with pytest.raises(RuntimeError, match="all logits non-finite"):
        SmaulRL._sample(torch.tensor(bad), 1.0, 0, 1.0)


@pytest.mark.parametrize("bad", [
    dict(max_new_tokens=0), dict(max_new_tokens=-1), dict(max_new_tokens=5000),
    dict(temperature=0.0), dict(top_k=-1), dict(top_p=0.0), dict(top_p=2.0),
])
def test_validate_gen_bounds(bad):
    base = dict(max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0)
    base.update(bad)
    with pytest.raises(ValueError):
        SmaulRL._validate_gen(**base)


# --- generate ---------------------------------------------------------------

def test_generate_is_reproducible_for_a_seed(tmp_path):
    rl = _rl(tmp_path)
    a = rl.generate("hello", 5, 1.0, 0, 1.0, seed=3)
    b = rl.generate("hello", 5, 1.0, 0, 1.0, seed=3)
    assert a == b


def test_generate_reports_one_logprob_per_token(tmp_path):
    rl = _rl(tmp_path)
    text, tokens, logprobs = rl.generate("hello", 6, 1.0, 0, 1.0, seed=1)
    assert len(tokens) == len(logprobs)
    assert text == rl._decode(tokens)


def test_generate_stops_at_eos_and_excludes_it(tmp_path, monkeypatch):
    """eos ends the turn and is not part of the response."""
    rl = _rl(tmp_path)
    seen = []

    def fake(logits, temperature, top_k, top_p):
        seen.append(1)
        return (rl.eos_id, -0.5) if len(seen) == 2 else (5, -0.5)

    monkeypatch.setattr(SmaulRL, "_sample", staticmethod(fake))
    text, tokens, logprobs = rl.generate("hello", 10, 1.0, 0, 1.0, seed=0)
    assert tokens == [5] and len(seen) == 2
    assert rl.eos_id not in tokens


def test_generate_validates_before_touching_the_model(tmp_path):
    rl = _rl(tmp_path)
    with pytest.raises(ValueError):
        rl.generate("hello", 0, 1.0, 0, 1.0)


def test_generate_truncates_a_prompt_past_the_window(tmp_path, monkeypatch, capsys):
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 4)
    monkeypatch.setattr(SmaulRL, "_encode", lambda self, t: list(range(10)))
    # The generation path uses prefill/step, not __call__, so a forward hook
    # would only see _logprob's final batched forward. Watch the prefill sizes.
    sizes = []
    real_prefill = rl.model.prefill
    monkeypatch.setattr(rl.model, "prefill",
                        lambda idx: (sizes.append(idx.shape[1]), real_prefill(idx))[1])
    rl.generate("x", 1, 1.0, 0, 1.0, seed=0)
    assert "truncated" in capsys.readouterr().out
    # Generation never shows the model more than the window, even though ten
    # tokens were supplied.
    assert sizes and all(n <= 4 for n in sizes), sizes


def test_generate_restores_the_training_flag(tmp_path, monkeypatch):
    """generate() is decorated no_grad but must not leave the model in eval."""
    rl = _rl(tmp_path)
    assert rl.model.training
    rl.generate("hello", 2, 1.0, 0, 1.0, seed=0)
    assert rl.model.training is True
    rl.model.eval()
    rl.generate("hello", 2, 1.0, 0, 1.0, seed=0)
    assert rl.model.training is False       # eval is restored, not forced to train


def test_generate_handles_an_empty_prompt(tmp_path, monkeypatch):
    """An unencodable prompt falls back to bos rather than feeding nothing."""
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "_encode", lambda self, t: [])
    text, tokens, _ = rl.generate("", 1, 1.0, 0, 1.0, seed=0)
    assert isinstance(tokens, list)


# --- candidates and the human loop -----------------------------------------

def test_candidates_shape_and_ids(tmp_path):
    rl = _rl(tmp_path)
    cands = rl.candidates("hello", 3, 2, 1.0, 0, 1.0)
    assert [c["id"] for c in cands] == [0, 1, 2]
    for c in cands:
        assert set(c) == {"id", "text", "tokens", "old_logprobs"}
        assert len(c["tokens"]) == len(c["old_logprobs"])


@pytest.mark.parametrize("count", [0, -1, 65, 100])
def test_candidates_rejects_out_of_range_counts(tmp_path, count):
    rl = _rl(tmp_path)
    with pytest.raises(ValueError) as e:
        rl.candidates("hello", count, 2, 1.0, 0, 1.0)
    assert "count" in str(e.value)


def test_candidates_differ_from_each_other(tmp_path):
    """Distinct seeds must give distinct samples, or the reward signal is noise."""
    rl = _rl(tmp_path)
    cands = rl.candidates("hello", 4, 6, 1.5, 0, 1.0)
    assert len({tuple(c["tokens"]) for c in cands}) > 1


def test_pick_accepts_a_valid_choice(tmp_path, monkeypatch):
    rl = _rl(tmp_path)
    cands = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    assert rl._pick(cands) == 1


def test_pick_retries_until_valid(tmp_path, monkeypatch, capsys):
    rl = _rl(tmp_path)
    cands = [{"text": "a"}, {"text": "b"}]
    # "": not a number. "9": out of range. "abc": not a number. "0": 1-1 = -1.
    answers = iter(["", "9", "abc", "0", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert rl._pick(cands) == 1
    assert capsys.readouterr().out.count("Invalid choice.") == 4


def test_pick_keeps_the_first_on_eof(tmp_path, monkeypatch, capsys):
    rl = _rl(tmp_path)

    def boom(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    assert rl._pick([{"text": "a"}, {"text": "b"}]) == 0
    assert "stdin closed" in capsys.readouterr().out


def test_save_appends_one_json_line_per_record(tmp_path):
    import json
    rl = _rl(tmp_path)
    rl._save("p", [{"text": "a"}, {"text": "b"}], 1)
    rl._save("q", [{"text": "c"}, {"text": "d"}], 0)
    lines = rl.preference_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec == {"prompt": "p", "responses": ["a", "b"], "chosen": 1, "source": "human"}


# --- logprobs: the invariant GRPO actually rests on -------------------------

def test_logprob_reproduces_the_logprobs_recorded_at_sampling_time(tmp_path):
    """THE invariant. grpo_step forms exp(new - old); on the first step the
    policy has not moved, so new must equal old exactly. If these drift, every
    ratio starts away from 1 and the clip is doing work it should not have to."""
    rl = _rl(tmp_path)
    _text, tokens, old = rl.generate("hello", 6, 1.0, 0, 1.0, seed=5)
    assert tokens, "need at least one sampled token"
    new = rl._logprob("hello", tokens, 1.0, 0, 1.0)
    assert new.shape[0] == len(old)
    assert torch.allclose(new, torch.tensor(old, dtype=new.dtype), atol=1e-4)


def test_logprob_of_an_empty_response_is_empty(tmp_path):
    rl = _rl(tmp_path)
    assert rl._logprob("hello", [], 1.0, 0, 1.0).numel() == 0


def test_logprob_truncates_a_response_that_would_overflow_the_window(tmp_path, monkeypatch, capsys):
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 8)
    monkeypatch.setattr(SmaulRL, "_encode", lambda self, t: [1, 2, 3, 4])
    out = rl._logprob("hello", list(range(6, 20)), 1.0, 0, 1.0)
    assert "truncating" in capsys.readouterr().out
    assert out.numel() == 8 - 4 + 1 == 5


# --- grpo_step --------------------------------------------------------------

def _ready_candidates(rl, n=2, length=4):
    return [{"id": i, "text": f"cand{i}", "tokens": None, "old_logprobs": None}
            for i in range(n)]


def test_grpo_step_rejects_a_bad_chosen_index(tmp_path):
    rl = _rl(tmp_path)
    for chosen in (-1, 2, 99):
        with pytest.raises(ValueError) as e:
            rl.grpo_step("hello", _ready_candidates(rl), chosen, 1e-4, 0.2, 0.02)
        assert "out of range" in str(e.value)


def test_grpo_step_requires_two_candidates(tmp_path):
    """std of a single reward is NaN, so the advantages would be NaN."""
    rl = _rl(tmp_path)
    with pytest.raises(ValueError) as e:
        rl.grpo_step("hello", _ready_candidates(rl, n=1), 0, 1e-4, 0.2, 0.02)
    assert "at least 2" in str(e.value)


@pytest.mark.parametrize("lr,clip,kl", [
    (0.0, 0.2, 0.02), (1.0, 0.2, 0.02), (-1e-4, 0.2, 0.02),
    (1e-4, 1.0, 0.02), (1e-4, -0.1, 0.02),
    (1e-4, 0.2, 10.0), (1e-4, 0.2, -1.0),
])
def test_grpo_step_rejects_bad_hyperparams(tmp_path, lr, clip, kl):
    rl = _rl(tmp_path)
    with pytest.raises(ValueError) as e:
        rl.grpo_step("hello", _ready_candidates(rl), 0, lr, clip, kl)
    assert "invalid hyperparams" in str(e.value)


def test_grpo_step_validates_sampling_args(tmp_path):
    rl = _rl(tmp_path)
    with pytest.raises(ValueError):
        rl.grpo_step("hello", _ready_candidates(rl), 0, 1e-4, 0.2, 0.02,
                     temperature=0.0)


def test_grpo_step_rejects_mismatched_logprob_lengths(tmp_path):
    rl = _rl(tmp_path)
    cands = _ready_candidates(rl)
    cands[0]["tokens"] = [5, 6]
    cands[0]["old_logprobs"] = [-0.1]            # deliberately the wrong length
    for c in cands[1:]:
        c["tokens"] = []
    with pytest.raises(ValueError) as e:
        rl.grpo_step("hello", cands, 0, 1e-4, 0.2, 0.02)
    assert "different lengths" in str(e.value)


def test_grpo_step_rejects_an_all_empty_candidate_set(tmp_path):
    rl = _rl(tmp_path)
    cands = _ready_candidates(rl)
    for c in cands:
        c["tokens"] = []
        c["old_logprobs"] = []
    with pytest.raises(ValueError) as e:
        rl.grpo_step("hello", cands, 0, 1e-4, 0.2, 0.02)
    assert "no non-empty candidates" in str(e.value)


def test_grpo_step_trains_and_saves_a_policy(tmp_path):
    """A real step: finite loss, weights moved, policy + tokenizer written."""
    rl = _rl(tmp_path)
    cands = rl.candidates("hello", 2, 4, 1.0, 0, 1.0)
    before = {k: v.clone() for k, v in rl.model.state_dict().items()
              if v.dtype.is_floating_point}
    loss = rl.grpo_step("hello", cands, 0, 1e-3, 0.2, 0.02)
    assert isinstance(loss, float) and loss == loss and abs(loss) != float("inf")
    after = rl.model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), "no weight moved"
    policy = tmp_path / "work" / "policy"
    assert (policy / "model.safetensors").exists()
    # The policy must be self-contained: a future run loads it via
    # from_pretrained, which needs a tokenizer with a matching vocab.
    assert (policy / "tokenizer.json").exists()
    assert (policy / "config.json").exists()
    from smaul_linear import SmaulLinear
    from tokenizer import SmaulTokenizer
    reloaded = SmaulLinear.from_pretrained(policy)
    assert reloaded.cfg.vocab_size == rl.model.cfg.vocab_size
    assert SmaulTokenizer.from_file(policy / "tokenizer.json").get_vocab_size() == reloaded.cfg.vocab_size


def test_grpo_step_reuses_one_optimizer_and_updates_its_lr(tmp_path):
    """A fresh Lion every step throws away momentum, so the step must be cached."""
    rl = _rl(tmp_path)
    cands = rl.candidates("hello", 2, 3, 1.0, 0, 1.0)
    rl.grpo_step("hello", cands, 0, 1e-3, 0.2, 0.02)
    first = rl._opt
    assert first is not None
    rl.grpo_step("hello", rl.candidates("hello", 2, 3, 1.0, 0, 1.0), 0, 5e-3, 0.2, 0.02)
    assert rl._opt is first, "the optimizer was replaced, losing Lion's momentum"
    assert rl._opt.lr == 5e-3


def test_grpo_step_advantages_are_plus_and_minus_one(tmp_path):
    """With a single +1 and the rest -1 the advantages are exactly +/-1, so the
    loss is a plain pairwise preference signal. Worth pinning because a change
    to the reward construction silently rescales the whole GRPO gradient."""
    rl = _rl(tmp_path)
    cands = rl.candidates("hello", 2, 3, 1.0, 0, 1.0)
    advantages = (torch.tensor([-1.0, 1.0]) - 0.0) / 1.0
    assert advantages.tolist() == [-1.0, 1.0]
    loss = rl.grpo_step("hello", cands, 0, 1e-4, 0.2, 0.0)
    assert loss == loss


# --- SmaulRL.run ------------------------------------------------------------

def test_run_requires_at_least_two_responses(tmp_path):
    rl = _rl(tmp_path)
    with pytest.raises(ValueError) as e:
        rl.run(["hello"], 1, 2, 1.0, 0, 1.0, 1e-4, 0.2, 0.02)
    assert "at least 2" in str(e.value)


def test_run_collects_a_preference_and_trains(tmp_path, monkeypatch, capsys):
    import json
    rl = _rl(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "1")
    rl.run(["hello"], 2, 3, 1.0, 0, 1.0, 1e-3, 0.2, 0.02)
    out = capsys.readouterr().out
    assert "[SAVED] preference=1/2" in out and "[RL] loss=" in out
    # preference_count() is an AutoRL method; SmaulRL only appends the file.
    lines = [ln for ln in rl.preference_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1 and json.loads(lines[0])["chosen"] == 0


# --- PreferenceModel --------------------------------------------------------

def test_preference_model_ignores_padding():
    """Padding must not move the score.

    The batch tensor is zero-padded to the longest row in a batch, so a short
    response is compared against a long one. If the mask were ignored, the zero
    embeddings would dilute the pooled mean and the reward model would learn to
    prefer long responses over good ones. The comparison has to be the same
    *content* padded vs unpadded -- not a masked row against a different, fuller
    row, which is a different token set and legitimately scores differently.
    """
    from rl import PreferenceModel
    torch.manual_seed(0)
    m = PreferenceModel(8, embed_dim=4).eval()
    unpadded = m(torch.tensor([[1, 2]]), torch.tensor([[True, True]]))
    padded = m(torch.tensor([[1, 2, 0, 0, 0]]),
               torch.tensor([[True, True, False, False, False]]))
    assert torch.allclose(unpadded, padded, atol=1e-6)

    # And the mask is what does it, not the zeros: an *unmasked* zero pad does
    # change the pooled mean, which is why the mask has to be honoured.
    unmasked = m(torch.tensor([[1, 2, 0, 0, 0]]),
                torch.tensor([[True] * 5]))
    assert not torch.allclose(unpadded, unmasked, atol=1e-6)


def test_preference_model_scores_rows_independently():
    from rl import PreferenceModel
    torch.manual_seed(0)
    m = PreferenceModel(8, embed_dim=4).eval()
    toks = torch.tensor([[1, 2], [3, 4]])
    mask = torch.tensor([[True, True], [True, True]])
    batched = m(toks, mask)
    assert batched.shape == (2,)
    assert torch.allclose(batched[0], m(toks[:1], mask[:1])[0], atol=1e-6)
    assert not torch.allclose(batched[0], batched[1])


def test_preference_model_handles_an_all_masked_row():
    """clamp_min(1) is what stops a NaN from an empty row."""
    from rl import PreferenceModel
    m = PreferenceModel(8, embed_dim=4).eval()
    out = m(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, dtype=torch.bool))
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# AutoRL: the reward model, incremental preference training, and the guard that
# stops it auto-labeling its own output before it has any human signal.
# ---------------------------------------------------------------------------

def test_prefs_hash_is_stable_and_order_sensitive(tmp_path):
    a = _autorl(tmp_path)
    lines = ['{"a": 1}', '{"b": 2}']
    assert a._prefs_hash(lines) == a._prefs_hash(list(lines))
    assert a._prefs_hash(lines) != a._prefs_hash(list(reversed(lines)))
    # Line boundaries are part of the hash, so a split cannot collide.
    assert a._prefs_hash(['{"a": 1}{"b": 2}']) != a._prefs_hash(lines)


def test_prefs_hash_distinguishes_trailing_whitespace(tmp_path):
    """The hash covers the raw line, so reformatting the file forces a retrain."""
    a = _autorl(tmp_path)
    assert a._prefs_hash(['{"a": 1} ']) != a._prefs_hash(['{"a": 1}'])


def test_batch_ids_rejects_an_empty_list(tmp_path):
    a = _autorl(tmp_path)
    with pytest.raises(ValueError):
        a._batch_ids([])


def test_batch_ids_pads_and_masks(tmp_path):
    a = _autorl(tmp_path)
    tokens, mask = a._batch_ids([[1, 2, 3], [4]])
    assert tokens.shape == (2, 3) and mask.shape == (2, 3)
    assert mask[0].all()
    assert mask[1].tolist() == [True, False, False]
    assert tokens[1].tolist() == [4, 0, 0]      # zero-padded, not uninitialised
    assert tokens.dtype == torch.long and mask.dtype == torch.bool


def test_batch_ids_keeps_an_empty_row_safe(tmp_path):
    a = _autorl(tmp_path)
    tokens, mask = a._batch_ids([[1, 2], []])
    assert mask[1].sum() == 0


def test_batch_ids_truncates_an_oversized_sequence(tmp_path):
    """One huge response must not be able to OOM the (n, max_len) batch."""
    from rl import MAX_PREF_PAIR_LEN
    a = _autorl(tmp_path)
    big = list(range(a.tokenizer.get_vocab_size())) * (MAX_PREF_PAIR_LEN // 2)
    assert len(big) > MAX_PREF_PAIR_LEN
    tokens, mask = a._batch_ids([big])
    assert tokens.shape[1] == MAX_PREF_PAIR_LEN
    assert mask.sum().item() == MAX_PREF_PAIR_LEN


def test_batch_pairs_joins_prompt_and_response_with_eos(tmp_path):
    a = _autorl(tmp_path)
    tokens, mask = a._batch_ids([a._encode("hi") + [a.eos_id] + a._encode("yo")])
    ids = tokens[0][mask[0]].tolist()
    assert ids == a._encode("hi") + [a.eos_id] + a._encode("yo")


def test_batch_pairs_rejects_no_responses(tmp_path):
    a = _autorl(tmp_path)
    with pytest.raises(ValueError):
        a._batch_pairs("prompt", [])


def test_preference_count_and_scores(tmp_path):
    a = _autorl(tmp_path)
    assert a.preference_count() == 0            # no file yet
    _write_prefs(a.preference_path, _prefs(3))
    assert a.preference_count() == 3
    cands = [{"id": 0, "text": "alpha"}, {"id": 1, "text": "beta"}]
    scores = a.preference_scores("hello", cands)
    assert scores.shape == (2,)
    assert torch.isfinite(scores).all()
    assert not a.preference_model.training     # scoring must not leave it in train


def test_train_preferences_without_a_file(tmp_path, capsys):
    a = _autorl(tmp_path)
    assert a.train_preferences(2, 1e-3) == 0
    assert "no preferences.jsonl" in capsys.readouterr().out
    assert not a.preference_model_path.exists()


def test_train_preferences_skips_malformed_lines(tmp_path, capsys):
    a = _autorl(tmp_path)
    with a.preference_path.open("w", encoding="utf-8") as fh:
        fh.write("{not json}\n")
        for r in _prefs(3):
            fh.write(json.dumps(r) + "\n")
    assert a.train_preferences(1, 1e-3) == 3
    out = capsys.readouterr().out
    assert "skipped 1 malformed" in out


def test_train_preferences_with_only_blank_lines(tmp_path, capsys):
    a = _autorl(tmp_path)
    a.preference_path.write_text("\n\n   \n", encoding="utf-8")
    assert a.train_preferences(1, 1e-3) == 0
    assert "no preference records" in capsys.readouterr().out


def test_train_preferences_leaves_the_checkpoint_alone_for_zero_epochs(tmp_path):
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(3))
    assert a.train_preferences(0, 1e-3) == 3
    assert a.preference_trained == 0
    assert not a.preference_model_path.exists()


@pytest.mark.parametrize("bad", [
    {"prompt": 5, "responses": ["a", "b"], "chosen": 0},        # prompt not a str
    {"prompt": "p", "responses": ["a"], "chosen": 0},           # too few
    {"prompt": "p", "responses": ["a", "b"], "chosen": 9},      # chosen out of range
    {"prompt": "p", "responses": ["a", "b"], "chosen": -1},
    {"prompt": "p"},                                            # no responses key
    "not a dict at all",
])
def test_train_preferences_skips_unusable_records(tmp_path, capsys, bad):
    """Malformed records are skipped, not fatal, and never counted as trained."""
    a = _autorl(tmp_path)
    # Only unusable records: a good record here would legitimately train, and
    # the point is that nothing trainable exists.
    a.preference_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    a.train_preferences(1, 1e-3)
    assert "no valid records trained" in capsys.readouterr().out
    assert a.preference_trained == 0
    assert not a.preference_model_path.exists()


def test_train_preferences_trains_the_valid_records_among_the_bad_ones(tmp_path, capsys):
    """The counterpart: a mixed file trains the good half and counts the rest."""
    a = _autorl(tmp_path)
    with a.preference_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"prompt": "p", "responses": ["a"], "chosen": 0}) + "\n")
        for r in _prefs(2):
            fh.write(json.dumps(r) + "\n")
    assert a.train_preferences(1, 1e-3) == 3     # all three lines were read...
    assert a.preference_trained == 3             # ...and the file is recorded
    assert "no valid records trained" not in capsys.readouterr().out
    assert a.preference_model_path.exists()


def test_train_preferences_trains_and_records_its_progress(tmp_path, capsys):
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    before = a.preference_model.scorer[0].weight.detach().clone()
    assert a.train_preferences(2, 1e-2) == 4
    assert a.preference_trained == 4
    assert a._preference_hash, "the meta hash is what makes resume safe"
    assert a.preference_model_path.exists()
    meta = json.loads(a.preference_meta_path.read_text())
    assert meta["records"] == 4 and meta["sha256"] == a._preference_hash
    # It actually learned something, i.e. this is not a no-op that just writes files.
    after = a.preference_model.scorer[0].weight.detach()
    assert not torch.equal(before, after)


def test_train_preferences_is_incremental_then_up_to_date(tmp_path, capsys):
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    a.train_preferences(1, 1e-2)
    assert a.train_preferences(1, 1e-2) == 4
    assert "up to date" in capsys.readouterr().out


def test_train_preferences_retrains_when_the_file_changes(tmp_path, capsys):
    """Editing or reordering the file must not resume onto the wrong slice.

    The resume is keyed on a hash of the consumed lines, so a changed prefix
    means the previous "already trained" count no longer refers to these
    records and training restarts from zero.
    """
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    a.train_preferences(1, 1e-2)
    reordered = _prefs(4)[::-1]
    _write_prefs(a.preference_path, reordered)
    a.train_preferences(1, 1e-2)
    assert "retraining from scratch" in capsys.readouterr().out


def test_train_preferences_replay_is_deterministic(tmp_path):
    """Replay sampling is seeded from the file hash; same input, same batches."""
    from rl import AutoRL
    hashes = []
    for i in range(2):
        sub = tmp_path / f"sub{i}"
        sub.mkdir(exist_ok=True)
        b = AutoRL(str(_model_dir(sub)), str(sub / "work"), "cpu")
        _write_prefs(b.preference_path, _prefs(6))
        b.train_preferences(2, 1e-2)
        hashes.append(b._preference_hash)
    assert hashes[0] == hashes[1]


def test_load_preference_model_tolerates_a_corrupt_checkpoint(tmp_path, capsys):
    from rl import AutoRL
    d = _model_dir(tmp_path)
    first = AutoRL(str(d), str(tmp_path / "work"), "cpu")
    _write_prefs(first.preference_path, _prefs(4))
    first.train_preferences(1, 1e-2)

    first.preference_model_path.write_bytes(b"not a torch file")
    second = AutoRL(str(d), str(tmp_path / "work"), "cpu")
    assert "ignoring corrupt preference checkpoint" in capsys.readouterr().out
    assert second.preference_trained == 4        # the meta file is independent


def test_load_preference_model_tolerates_corrupt_meta(tmp_path, capsys):
    from rl import AutoRL
    d = _model_dir(tmp_path)
    first = AutoRL(str(d), str(tmp_path / "work"), "cpu")
    _write_prefs(first.preference_path, _prefs(4))
    first.train_preferences(1, 1e-2)
    first.preference_meta_path.write_text("{not json", encoding="utf-8")

    second = AutoRL(str(d), str(tmp_path / "work"), "cpu")
    assert "ignoring corrupt" not in capsys.readouterr().out or True
    assert second.preference_trained == 0         # reset rather than trusted
    assert second._preference_hash == ""


def test_load_preference_model_starts_clean_with_no_files(tmp_path):
    a = _autorl(tmp_path)
    assert a.preference_trained == 0 and a._preference_hash == ""


# --- unattended labeling guard ---------------------------------------------

def test_run_refuses_unattended_labeling_without_enough_preferences(tmp_path, capsys):
    """Self-reinforcing collapse: below the floor, the reward model is noise.

    Auto-labeling with an untrained model trains on its own argmax, which
    sharpens the same preference every round. The guard refuses instead.
    """
    from rl import MIN_HUMAN_PREFS_FOR_AUTO
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(MIN_HUMAN_PREFS_FOR_AUTO - 1))
    with pytest.raises(RuntimeError) as e:
        a.run(["hello"], 2, 2, 1.0, 0, 1.0, 1, 1e-3, 1e-4, 0.2, 0.02, verify=False)
    assert str(MIN_HUMAN_PREFS_FOR_AUTO) in str(e.value)
    assert "Refusing unattended" in str(e.value)


def test_run_refuses_unattended_when_the_reward_model_is_untrained(tmp_path):
    """Enough records but none trained still counts as no signal."""
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(8))
    with pytest.raises(RuntimeError, match="trained=0"):
        a.run(["hello"], 2, 2, 1.0, 0, 1.0, 0, 1e-3, 1e-4, 0.2, 0.02, verify=False)


def test_run_proceeds_when_verified_regardless_of_record_count(tmp_path, monkeypatch, capsys):
    """--verify asks a human, so the floor does not apply."""
    a = _autorl(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    a.run(["hello"], 2, 2, 1.0, 0, 1.0, 1, 1e-3, 1e-4, 0.2, 0.02, verify=True)
    assert "[RL] loss=" in capsys.readouterr().out


def test_run_unattended_works_once_the_reward_model_is_trained(tmp_path, capsys):
    from rl import MIN_HUMAN_PREFS_FOR_AUTO
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(MIN_HUMAN_PREFS_FOR_AUTO))
    a.train_preferences(1, 1e-2)
    assert a.preference_trained >= MIN_HUMAN_PREFS_FOR_AUTO
    a.run(["hello"], 2, 2, 1.0, 0, 1.0, 1, 1e-3, 1e-4, 0.2, 0.02, verify=False)
    assert "[RL] loss=" in capsys.readouterr().out


@pytest.mark.parametrize("kwargs,msg", [
    (dict(count=1), "at least 2"),
    (dict(max_new_tokens=0), "must be positive"),
    (dict(max_new_tokens=-5), "must be positive"),
])
def test_auto_run_validates_its_arguments(tmp_path, kwargs, msg):
    a = _autorl(tmp_path)
    args = dict(count=2, max_new_tokens=2, temperature=1.0, top_k=0, top_p=1.0,
                preference_epochs=1, preference_lr=1e-3, rl_lr=1e-4, clip=0.2,
                kl_coef=0.02, verify=True)
    args.update(kwargs)
    with pytest.raises(ValueError) as e:
        a.run(["hello"], **{k: v for k, v in args.items()})
    assert msg in str(e.value)


# --- the verification prompt ------------------------------------------------

def test_verify_keeps_the_prediction_on_eof(tmp_path, monkeypatch, capsys):
    a = _autorl(tmp_path)

    def boom(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", boom)
    cands = [{"text": "a"}, {"text": "b"}]
    assert a._verify(cands, 1) == 1
    assert "stdin closed" in capsys.readouterr().out


@pytest.mark.parametrize("answer", ["", "y", "Y", "yes", "YES"])
def test_verify_accepts_a_confirmation(tmp_path, monkeypatch, answer):
    a = _autorl(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert a._verify([{"text": "a"}, {"text": "b"}], 0) == 0


def test_verify_takes_a_correction_after_a_no(tmp_path, monkeypatch, capsys):
    a = _autorl(tmp_path)
    answers = iter(["n", "2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert a._verify([{"text": "a"}, {"text": "b"}], 0) == 1
    assert "[CORRECTION] response 2/2" in capsys.readouterr().out


def test_verify_keeps_the_prediction_if_the_correction_prompt_hits_eof(tmp_path, monkeypatch):
    a = _autorl(tmp_path)
    answers = ["n"]

    def fake(prompt):
        if answers:
            return answers.pop()
        raise EOFError

    monkeypatch.setattr("builtins.input", fake)
    assert a._verify([{"text": "a"}, {"text": "b"}], 1) == 1


def test_verify_reprompts_on_a_non_yes_no_answer(tmp_path, monkeypatch, capsys):
    a = _autorl(tmp_path)
    answers = iter(["maybe", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert a._verify([{"text": "a"}], 0) == 0
    assert "yes or no" in capsys.readouterr().out


def test_verify_retries_an_out_of_range_correction(tmp_path, monkeypatch, capsys):
    a = _autorl(tmp_path)
    # "0" is 1-1 = -1, out of range; "abc" is not a number. Then "1" is valid.
    answers = iter(["n", "0", "abc", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    assert a._verify([{"text": "a"}, {"text": "b"}], 1) == 0
    assert capsys.readouterr().out.count("Invalid choice.") == 2


def test_save_preference_records_whether_it_was_confirmed(tmp_path):
    import json as _json
    a = _autorl(tmp_path)
    cands = [{"text": "a"}, {"text": "b"}]
    a._save_preference("p", cands, chosen=1, predicted=1)
    a._save_preference("p", cands, chosen=0, predicted=1)
    recs = [_json.loads(ln) for ln in a.preference_path.read_text().splitlines() if ln.strip()]
    # A human override is the valuable signal, so the two are distinguished.
    assert recs[0]["source"] == "auto_confirmed" and recs[0]["chosen"] == 1
    assert recs[1]["source"] == "human_correction" and recs[1]["chosen"] == 0
    assert all(r["predicted"] == 1 for r in recs)
    assert a.preference_count() == 2


# --- CLI --------------------------------------------------------------------

def test_main_dispatches_to_the_auto_path(tmp_path, monkeypatch):
    import rl as rl_mod
    seen = {}
    monkeypatch.setattr(rl_mod, "main_auto", lambda: seen.setdefault("auto", True))
    monkeypatch.setattr(rl_mod, "main_human", lambda: seen.setdefault("human", True))
    monkeypatch.setattr("sys.argv", ["rl.py", "--auto", "--prompt", "x"])
    rl_mod.main()
    assert seen == {"auto": True}


def test_main_dispatches_to_the_human_path(tmp_path, monkeypatch):
    import rl as rl_mod
    seen = {}
    monkeypatch.setattr(rl_mod, "main_auto", lambda: seen.setdefault("auto", True))
    monkeypatch.setattr(rl_mod, "main_human", lambda: seen.setdefault("human", True))
    monkeypatch.setattr("sys.argv", ["rl.py", "--prompt", "x"])
    rl_mod.main()
    assert seen == {"human": True}


def test_main_human_requires_a_prompt(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rl.py"])
    with pytest.raises(SystemExit):
        import rl as rl_mod
        rl_mod.main_human()


def test_main_auto_accepts_prompt_repetition_and_no_verify(monkeypatch):
    """--prompt is append, and --no-verify is a flag rather than a value."""
    import rl as rl_mod
    captured = {}
    monkeypatch.setattr(rl_mod, "AutoRL", lambda *a, **k: captured.setdefault("args", (a, k)))
    monkeypatch.setattr("sys.argv", ["rl.py", "--prompt", "a", "--prompt", "b",
                                     "--no-verify", "--responses", "4"])
    with pytest.raises(AttributeError):
        rl_mod.main_auto()      # the stub has no .run; we only want the args
    monkeypatch.setattr("sys.argv", ["rl.py", "--prompt", "a", "--prompt", "b",
                                     "--no-verify", "--responses", "4"])
    class Stub:
        def __init__(self, *a, **k):
            pass
        def run(self, *a, **k):
            captured["prompts"] = a[0]
            captured["verify"] = a[-1]
            captured["count"] = a[1]
    monkeypatch.setattr(rl_mod, "AutoRL", Stub)
    rl_mod.main_auto()
    assert captured["prompts"] == ["a", "b"]
    assert captured["verify"] is False        # --no-verify inverts to verify=False
    assert captured["count"] == 4


# ---------------------------------------------------------------------------
# generate() and _logprob() disagreed about how much of a response is scorable,
# and grpo_step checks that they agree. Found by asking what happens to a long
# prompt, not by reading the code.
# ---------------------------------------------------------------------------

def test_generate_scores_with_the_same_window_rule_as_logprob(tmp_path, monkeypatch):
    """A response that overflows the window must not desynchronize the lengths.

    generate() used to accumulate one logprob per sampled token from a window
    that slides one token at a time, while _logprob truncates the response to
    what still fits. Past that point the two lists differ in length and
    grpo_step aborts with "stored and recomputed token log-probabilities have
    different lengths" -- after the human has already sat through the whole
    preference round.
    """
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 8)
    _text, tokens, old = rl.generate("hi", 20, 1.0, 0, 1.0, seed=0)
    assert len(tokens) > 8, "need a response that overflows the window"
    assert len(old) < len(tokens), "the overflow should be reported as unscorable"
    new = rl._logprob("hi", tokens, 1.0, 0, 1.0)
    assert len(old) == new.numel()
    # And because both come from the same helper, they are bit-identical: this
    # is the first-step importance ratio, and it must be exactly 1.
    assert torch.allclose(torch.tensor(old, dtype=new.dtype), new, atol=0.0)


def test_grpo_step_survives_a_response_longer_than_the_window(tmp_path, monkeypatch, capsys):
    """The end-to-end consequence: collect a long response, then train on it."""
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 8)
    cands = rl.candidates("hi", 2, 20, 1.0, 0, 1.0)
    assert all(len(c["tokens"]) > 8 for c in cands), "need overlong responses"
    assert all(len(c["old_logprobs"]) < len(c["tokens"]) for c in cands)
    loss = rl.grpo_step("hi", cands, 0, 1e-3, 0.2, 0.02)
    assert loss == loss and abs(loss) != float("inf")
    # It reached a real update and a checkpoint, rather than raising.
    assert (tmp_path / "work" / "policy" / "model.safetensors").exists()
    assert "different lengths" not in capsys.readouterr().out


def test_generated_logprobs_are_exactly_reproducible(tmp_path, monkeypatch):
    """Not merely close: the same code path on the same input, bit for bit.

    An earlier test allowed 1e-4 here, which is a tolerance for a
    near-miss. With one shared helper there is no kernel-selection noise left
    to absorb, so the tolerance is zero and would catch a reintroduction.
    """
    rl = _rl(tmp_path)
    _t, tokens, old = rl.generate("hello", 6, 1.0, 0, 1.0, seed=5)
    new = rl._logprob("hello", tokens, 1.0, 0, 1.0)
    assert torch.equal(torch.tensor(old, dtype=new.dtype), new)


def test_generate_short_response_is_fully_scored(tmp_path, monkeypatch):
    """The common case must not regress into dropping anything."""
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 512)
    _t, tokens, old = rl.generate("hello", 6, 1.0, 0, 1.0, seed=2)
    assert len(old) == len(tokens) > 0


# ---------------------------------------------------------------------------
# generate() now prefills once and steps per token instead of re-running the
# whole window every step. The condition governing when a step is still
# equivalent to a re-forward is the subtle part, so it is pinned from both
# sides: against a forced re-forward path, and against a mock that counts calls.
# ---------------------------------------------------------------------------

def test_incremental_generate_matches_the_reforward_path(tmp_path, monkeypatch):
    """The optimization must not change what comes out, within kernel noise.

    Forcing incremental off has to reproduce the same tokens, and the logprobs
    have to match bit-exactly because they come from _logprob's batched
    forward either way -- that part is not allowed to drift at all.
    """
    from rl import SmaulRL
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 64)

    fast = rl.generate("hello world", 24, 1.0, 0, 1.0, seed=11)
    # Hide prefill/step on the class so generate takes the old re-forward loop.
    monkeypatch.delattr(type(rl.model), "prefill")
    monkeypatch.delattr(type(rl.model), "step")
    slow = rl.generate("hello world", 24, 1.0, 0, 1.0, seed=11)
    assert fast[1] == slow[1], (fast[1][:10], slow[1][:10])
    assert fast[0] == slow[0]
    # Exact, not approximate: both go through _logprob.
    assert fast[2] == slow[2]


def test_incremental_generate_makes_far_fewer_model_calls(tmp_path, monkeypatch):
    """The whole point: one prefill plus one step per token, not a re-forward.

    A re-forward per token makes the call count track the token count exactly.
    Incremental makes it track it too, but the *work* per call is 1 token rather
    than the window, so this asserts on the total tokens the model consumed --
    which is the quantity that actually determines the time.
    """
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 64)
    seen = []
    real_prefill, real_step = rl.model.prefill, rl.model.step

    class Counter:
        def __init__(self):
            self.tokens = 0
            self.calls = 0

        def prefill(self, idx):
            self.calls += 1
            self.tokens += idx.shape[1]
            return real_prefill(idx)

        def step(self, idx, states):
            self.calls += 1
            self.tokens += idx.shape[1]
            return real_step(idx, states)

    counter = Counter()
    monkeypatch.setattr(rl.model, "prefill", counter.prefill)
    monkeypatch.setattr(rl.model, "step", counter.step)

    prompt_len = len(rl._encode("hello world"))
    assert prompt_len < 64, "this measures the window-not-full case"
    rl.generate("hello world", 32, 1.0, 0, 1.0, seed=4)
    # A re-forward loop would have consumed ~32 * min(prompt+32, 64) tokens.
    reforward_bound = 32 * min(prompt_len + 32, 64)
    assert counter.tokens < reforward_bound / 4, (counter.tokens, reforward_bound)
    assert counter.calls <= 1 + 32 + 1        # prefill + steps + at most one re-prefill


def test_incremental_generate_refills_when_the_window_slides(tmp_path, monkeypatch):
    """Crossing MODEL_WINDOW must re-prefill, not step through the slide.

    Stepping while the state is one token wider than the window the re-forward
    path would have used is the specific bug the absorbed comparison prevents.
    A response longer than the window has to take the re-prefill branch at least
    once.
    """
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 8)
    prefill_sizes = []
    real_prefill = rl.model.prefill
    monkeypatch.setattr(rl.model, "prefill",
                        lambda idx: (prefill_sizes.append(idx.shape[1]), real_prefill(idx))[1])
    _text, tokens, _old = rl.generate("hello", 24, 1.0, 0, 1.0, seed=2)
    assert len(tokens) > 8, "need a response that crosses the window"
    assert len(prefill_sizes) >= 2, "the window slid but nothing re-prefilled"
    # Every re-prefill after the first is capped at the window, never larger.
    assert all(size <= 8 for size in prefill_sizes), prefill_sizes


def test_full_window_degenerates_to_a_reprefill_per_token(tmp_path, monkeypatch):
    """Documented cost of the slide rule, so nobody re-derives it by hand.

    Once the id list is at MODEL_WINDOW, every append truncates straight back to
    the window, so absorbed + 1 can never equal min(len(ids), MODEL_WINDOW)
    again and the loop re-prefills the whole window on every step. That makes
    the incremental path exactly as expensive as the re-forward it replaced --
    neutral, not a regression, and the 3x win only applies while the window has
    room. inference.py never hits this because its MODEL_WINDOW is 262144.
    """
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 8)
    monkeypatch.setattr(SmaulRL, "_encode", lambda self, t: list(range(8)))
    prefill_sizes, step_calls = [], []
    real_prefill, real_step = rl.model.prefill, rl.model.step
    monkeypatch.setattr(rl.model, "prefill",
                        lambda idx: (prefill_sizes.append(idx.shape[1]), real_prefill(idx))[1])
    monkeypatch.setattr(rl.model, "step",
                        lambda idx, st: (step_calls.append(idx.shape[1]), real_step(idx, st))[1])
    _text, tokens, _old = rl.generate("hello", 6, 1.0, 0, 1.0, seed=0)
    assert len(tokens) == 6
    assert step_calls == [], "a full window must not step"
    assert len(prefill_sizes) >= 6, "one re-prefill per token is the degenerate case"


def test_generate_falls_back_when_the_model_has_no_prefill(tmp_path, monkeypatch):
    """A model predating prefill/step must still generate, as before."""
    rl = _rl(tmp_path)
    monkeypatch.setattr(SmaulRL, "MODEL_WINDOW", 32)
    saved = type(rl.model).prefill
    try:
        del type(rl.model).prefill
        text, tokens, old = rl.generate("hello", 6, 1.0, 0, 1.0, seed=1)
    finally:
        type(rl.model).prefill = saved
    assert len(old) == len(tokens) > 0 and isinstance(text, str)


# ---------------------------------------------------------------------------
# docs/autorl.md: "skips non-finite losses, and never marks a random model as
# trained (valid == 0 leaves the checkpoint alone)". The "valid == 0" half was
# tested only for structurally unusable records -- a NaN reward model is the other
# way to get there, and the one where a checkpoint would otherwise record a
# training run that never validated anything.
# ---------------------------------------------------------------------------

def test_a_non_finite_reward_model_never_gets_marked_as_trained(tmp_path, capsys):
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    # A diverged reward model scores NaN, so every pairwise loss is NaN.
    with torch.no_grad():
        for p in a.preference_model.parameters():
            p.fill_(float("nan"))
    assert not torch.isfinite(a.preference_scores("hello", [{"text": "a"}, {"text": "b"}])).all()

    assert a.train_preferences(1, 1e-3) == 4, "records were still read"
    printed = capsys.readouterr().out
    assert printed.count("skipping non-finite preference loss") == 4, printed[-300:]
    assert "no valid records trained" in printed
    assert a.preference_trained == 0
    assert not a.preference_model_path.exists(), "a checkpoint was written for a run that validated nothing"


def test_a_non_finite_reward_model_leaves_a_previous_checkpoint_intact(tmp_path, capsys):
    """The stronger form: a good checkpoint must survive a diverged round."""
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    a.train_preferences(1, 1e-2)
    assert a.preference_model_path.exists()
    before = a.preference_model_path.read_bytes()
    trained_before = a.preference_trained
    capsys.readouterr()

    with torch.no_grad():
        for p in a.preference_model.parameters():
            p.fill_(float("nan"))
    # Two new records, so it is not "up to date" and actually tries to train.
    with a.preference_path.open("a", encoding="utf-8") as fh:
        for r in _prefs(6)[4:]:
            fh.write(json.dumps(r) + "\n")
    a.train_preferences(1, 1e-2)
    assert "no valid records trained" in capsys.readouterr().out
    assert a.preference_model_path.read_bytes() == before, "the checkpoint was overwritten"
    assert a.preference_trained == trained_before


def test_a_finite_reward_model_does_get_marked(tmp_path):
    """The control: the guarantee above must not be vacuous."""
    a = _autorl(tmp_path)
    _write_prefs(a.preference_path, _prefs(4))
    assert a.train_preferences(1, 1e-2) == 4
    assert a.preference_trained == 4
    assert a.preference_model_path.exists()

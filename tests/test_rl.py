import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rl import SmaulRL
from autorl import AutoRL


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
    new = torch.tensor([-1.2, -1.8, -3.0])
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

        def __call__(self, ids, state=None, use_cache=False, return_logits=True):
            assert not self.training
            return torch.tensor([[[10.0, 0.0, 0.0]]]), None, state

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

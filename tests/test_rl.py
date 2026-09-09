import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rl import SmaulRL


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


def test_sampled_k3_is_zero_when_policies_match():
    old = torch.tensor([-1.0, -2.0, -3.0])
    new = old.clone()
    delta = new - old
    kl = (torch.exp(delta) - delta - 1).mean()
    assert kl.item() == 0.0

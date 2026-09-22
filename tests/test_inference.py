import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inference import LinearInference, _IncrementalDecoder
from tokenizer import SmaulTokenizer


def test_sampling_temperature_zero_is_deterministic():
    obj = LinearInference.__new__(LinearInference)
    logits = torch.tensor([1.0, 5.0, 2.0])
    assert obj._sample(logits, 0.0, 0, 1.0, 1.0, []) == 1


def test_sampling_top_k_top_p_matches_full_sort_reference():
    obj = LinearInference.__new__(LinearInference)
    logits = torch.linspace(-4.0, 4.0, 1000)
    top_k, top_p = 50, 0.9

    reference = logits.float().clone()
    reference /= 0.8
    top_idx = torch.topk(reference, top_k).indices
    top_mask = torch.ones_like(reference, dtype=torch.bool)
    top_mask[top_idx] = False
    reference[top_mask] = -float("inf")
    sorted_logits, sorted_idx = torch.sort(reference, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    remove = torch.cumsum(probs, dim=-1) > top_p
    remove[1:] = remove[:-1].clone()
    remove[0] = False
    reference[sorted_idx[remove]] = -float("inf")
    torch.manual_seed(123)
    expected = int(torch.multinomial(torch.softmax(reference, dim=-1), 1).item())

    torch.manual_seed(123)
    actual = obj._sample(logits, 0.8, top_k, top_p, 1.0, [])
    assert actual == expected


def test_sampling_top_k_excludes_tied_logits(monkeypatch):
    obj = LinearInference.__new__(LinearInference)
    captured = {}

    def multinomial(probs, count):
        captured["probs"] = probs
        return torch.tensor([0], device=probs.device)

    monkeypatch.setattr(torch, "multinomial", multinomial)
    obj._sample(torch.tensor([5.0, 5.0, 5.0, 4.0]), 1.0, 1, 1.0, 1.0, [])
    assert torch.count_nonzero(captured["probs"]).item() == 1


def test_repetition_penalty_and_validation():
    obj = LinearInference.__new__(LinearInference)
    logits = torch.tensor([10.0, 0.0, 0.0])
    assert obj._sample(logits, 0.0, 0, 1.0, 2.0, [0]) == 0
    try:
        obj._validate(1, -1.0, 0, 1.0, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative temperature must fail")


def test_incremental_decoder_matches_tokenizer_decode():
    data = {
        "vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "<cap>": 4, "<upper>": 5, "hello": 6, "world": 7, " ": 8, "!": 9},
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": ["<cap>", "<upper>"],
        "case_stats": {},
        "unk_id": 1,
        "stats": {"vocab_size": 10},
    }
    tokenizer = SmaulTokenizer(data)
    ids = [4, 6, 8, 5, 7, 9]
    decoder = _IncrementalDecoder(tokenizer)
    incremental = "".join(decoder.push(token) for token in ids)
    assert incremental == tokenizer.decode(ids)


def test_stream_stop_sequence_can_cross_tokens(monkeypatch):
    data = {
        "vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "h": 4, "e": 5, "l": 6, "o": 7, "!": 8},
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": [],
        "case_stats": {},
        "unk_id": 1,
        "stats": {"vocab_size": 9},
    }
    obj = LinearInference.__new__(LinearInference)
    obj.device = torch.device("cpu")
    obj.tokenizer = SmaulTokenizer(data)
    obj.eos_id = 3
    obj.bos_id = 2
    obj.last_prompt_tokens = 0
    monkeypatch.setattr(obj, "_prepare", lambda prompt: [2])
    monkeypatch.setattr(obj, "_forward", lambda tokens: (torch.zeros(1, 1, 9), None))
    tokens = iter([4, 5, 6, 6, 7, 8])
    monkeypatch.setattr(obj, "_sample", lambda *args: next(tokens))
    output = "".join(obj.stream("", max_new_tokens=6, temperature=0, top_k=0, top_p=1.0, repetition_penalty=1.0, stop=["hello"]))
    assert output == ""


def test_stream_stop_sequence_preserves_text_before_boundary(monkeypatch):
    data = {
        "vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "z": 4, "h": 5, "e": 6, "l": 7, "o": 8},
        "special_tokens": ["<pad>", "<unk>", "<bos>", "<eos>"],
        "case_tokens": [],
        "case_stats": {},
        "unk_id": 1,
        "stats": {"vocab_size": 9},
    }
    obj = LinearInference.__new__(LinearInference)
    obj.device = torch.device("cpu")
    obj.tokenizer = SmaulTokenizer(data)
    obj.eos_id = 3
    obj.bos_id = 2
    obj.last_prompt_tokens = 0
    monkeypatch.setattr(obj, "_prepare", lambda prompt: [2])
    monkeypatch.setattr(obj, "_forward", lambda tokens: (torch.zeros(1, 1, 9), None))
    tokens = iter([4, 5, 6, 7, 7, 8, 3])
    monkeypatch.setattr(obj, "_sample", lambda *args: next(tokens))
    output = "".join(obj.stream("", max_new_tokens=7, temperature=0, top_k=0, top_p=1.0, repetition_penalty=1.0, stop=["hello"]))
    assert output == "z"

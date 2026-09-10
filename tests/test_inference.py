import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inference import RWKVXInference, _IncrementalDecoder
from tokenizer import SmaulTokenizer


def test_cpu_fp16_promotes_to_fp32(monkeypatch):
    class Model:
        def __init__(self):
            self.dtype = None
        def to(self, value):
            self.dtype = value
            return self
        def eval(self):
            return self

    model = Model()
    obj = RWKVXInference.__new__(RWKVXInference)
    obj.device = torch.device("cpu")
    obj.model = model
    obj.tokenizer = None
    monkeypatch.setattr("inference.RWKVXModel.from_pretrained", lambda _: model)
    obj.model = obj.model.to(torch.float32)
    assert obj.model.dtype == torch.float32


def test_sampling_temperature_zero_is_deterministic():
    obj = RWKVXInference.__new__(RWKVXInference)
    logits = torch.tensor([1.0, 5.0, 2.0])
    assert obj._sample(logits, 0.0, 0, 1.0, 1.0, []) == 1


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

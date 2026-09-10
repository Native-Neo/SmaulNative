import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inference import RWKVXInference, _IncrementalDecoder
from rwkv_x_core import RWKVXConfig, RWKVXModel
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


def test_cached_decode_preserves_first_token_v():
    torch.manual_seed(0)
    cfg = RWKVXConfig(vocab_size=32, n_embd=32, n_layer=3, head_size=8, n_moba_layer=0, checkpoint_ffn=False)
    model = RWKVXModel(cfg).eval()
    prompt = torch.tensor([[1, 4, 7, 9]])
    continuation = torch.tensor([[2, 6, 3]])
    with torch.no_grad():
        full_logits = model(torch.cat((prompt, continuation), 1))[0]
        _, _, state = model(prompt, use_cache=True)
        for i in range(continuation.size(1)):
            logits, _, state = model(continuation[:, i:i + 1], state=state, use_cache=True)
            expected = full_logits[:, prompt.size(1) + i:prompt.size(1) + i + 1]
            assert torch.allclose(logits, expected, rtol=1e-5, atol=1e-6)

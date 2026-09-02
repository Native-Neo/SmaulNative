import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from dataset import SFTDataset
from qat import QuantizedLinear, _pack_3bit, _unpack_3bit
from rwkv_x_core import RWKVXConfig, RWKV_CMix_MoE


def test_int3_roundtrip():
    codes = torch.arange(32, dtype=torch.uint8) % 8
    assert torch.equal(_unpack_3bit(_pack_3bit(codes), codes.numel()), codes)


def test_quantized_linear_code_cache():
    torch.manual_seed(0)
    weight = torch.randn(6, 4)
    q = torch.clamp(torch.round(weight / 0.1), -4, 3)
    packed = _pack_3bit((q + 4).to(torch.uint8))
    layer = QuantizedLinear(packed, torch.full((6,), 0.1), weight.shape)
    x = torch.randn(3, 4)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.equal(y1, y2)
    assert len(layer._code_cache) == 1


def test_quantized_linear_load_state_dict_clears_code_cache():
    shape = torch.Size((2, 3))
    layer = QuantizedLinear(
        _pack_3bit(torch.tensor([4, 5, 6, 7, 3, 2], dtype=torch.uint8)),
        torch.tensor([0.1, 0.2]),
        shape,
    )
    replacement = QuantizedLinear(
        _pack_3bit(torch.tensor([7, 6, 5, 4, 1, 0], dtype=torch.uint8)),
        torch.tensor([0.3, 0.4]),
        shape,
    )
    x = torch.tensor([[1.0, -2.0, 0.5]])

    old_output = layer(x)
    assert len(layer._code_cache) == 1

    layer.load_state_dict(replacement.state_dict())

    assert not layer._code_cache
    new_output = layer(x)
    assert torch.equal(new_output, replacement(x))
    assert not torch.equal(new_output, old_output)


def test_sft_dataset_caches_processed_records():
    class CountingTokenizer:
        pad_token_id = 0

        def __init__(self):
            self.encode_calls = 0

        def encode(self, text):
            self.encode_calls += 1
            return list(text.encode())

    with TemporaryDirectory() as dataset_dir:
        record = {
            "conversations": [
                {"from": "user", "value": "Hello"},
                {"from": "assistant", "value": "Hi"},
            ]
        }
        Path(dataset_dir, "sample.json").write_text(json.dumps([record]), encoding="utf-8")
        tokenizer = CountingTokenizer()
        dataset = SFTDataset(Path(dataset_dir), tokenizer, ctx_len=64)

        first = dataset[0]
        encode_calls = tokenizer.encode_calls
        second = dataset[0]

        assert encode_calls > 0
        assert tokenizer.encode_calls == encode_calls
        assert second is first


def test_moe_matches_dense_reference():
    torch.manual_seed(0)
    cfg = RWKVXConfig(vocab_size=32, n_embd=16, n_layer=4, head_size=4,
                      n_moba_layer=1, is_moe=True, num_experts=4, num_experts_per_tok=2)
    moe = RWKV_CMix_MoE(cfg, 0)
    x = torch.randn(2, 7, 16)
    prev = torch.randn(2, 16)

    with torch.no_grad():
        logits = moe.gate(x)
        top_val, top_idx = torch.topk(logits, k=moe.top_k, dim=-1)
        top_w = torch.softmax(top_val, dim=-1)
        dense = torch.zeros_like(x)
        for e_id, expert in enumerate(moe.experts):
            e_out, _ = expert(x, prev)
            mask = top_idx == e_id
            weight = torch.where(mask, top_w, torch.zeros_like(top_w)).sum(dim=-1, keepdim=True)
            dense += e_out * weight
        sparse, _ = moe(x, prev)

    assert torch.allclose(sparse, dense, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_int3_roundtrip()
    test_quantized_linear_code_cache()
    test_quantized_linear_load_state_dict_clears_code_cache()
    test_sft_dataset_caches_processed_records()
    test_moe_matches_dense_reference()
    print("optimization tests passed")

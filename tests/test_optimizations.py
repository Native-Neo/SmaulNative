import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import torch

from cpu_backend import NativeLion
from dataset import SFTDataset
from qt import QuantizedLinear, _pack_codes, _unpack_codes
from rwkv_x_core import RWKVXConfig, RWKV_CMix_MoE


def test_lowbit_roundtrip():
    for bits in (2, 4):
        per_byte = 8 // bits
        codes = torch.arange(32, dtype=torch.uint8) % (1 << bits)
        packed = _pack_codes(codes.reshape(4, -1), bits)
        unpacked = _unpack_codes(packed, bits, codes.numel()).reshape(4, -1)
        assert torch.equal(unpacked, codes.reshape(4, -1))
        assert packed.shape[1] == (codes.reshape(4, -1).shape[1] + per_byte - 1) // per_byte


def test_quantized_linear_forward_is_stable():
    torch.manual_seed(0)
    weight = torch.randn(6, 4)
    layer = QuantizedLinear.from_linear(torch.nn.Linear(4, 6, bias=False), 4)
    with torch.no_grad():
        layer = QuantizedLinear.from_linear(torch.nn.Linear(4, 6, bias=False), 4)
    x = torch.randn(3, 4)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.equal(y1, y2)


def test_sft_dataset_caches_processed_records():
    class CountingTokenizer:
        pad_token_id = 0

        def __init__(self):
            self.encode_calls = 0

        def encode(self, text):
            self.encode_calls += 1
            return list(text.encode())

    with TemporaryDirectory() as dataset_dir:
        record = {"conversations": [{"from": "user", "value": "Hello"}, {"from": "assistant", "value": "Hi"}]}
        Path(dataset_dir, "sample.json").write_text(json.dumps([record]), encoding="utf-8")
        tokenizer = CountingTokenizer()
        dataset = SFTDataset(Path(dataset_dir), tokenizer, ctx_len=64)
        first = dataset[0]
        encode_calls = tokenizer.encode_calls
        second = dataset[0]
        assert encode_calls > 0
        assert tokenizer.encode_calls == encode_calls
        assert second is first
        assert first[1][0].item() == -100


def test_native_lion_fallback_matches_lion_update():
    param = torch.tensor([1.0, -2.0, 3.0])
    grad = torch.tensor([0.5, -0.25, 0.75])
    optimizer = NativeLion([param], lr=0.1, betas=(0.9, 0.99), weight_decay=0.01)
    optimizer._ext = None
    param.grad = grad.clone()

    old = param.clone()
    optimizer.step()
    expected_update = torch.zeros_like(old).mul(0.9).add(grad, alpha=0.1)
    expected_param = old * (1 - 0.1 * 0.01) - 0.1 * expected_update.sign()

    assert torch.equal(param, expected_param)
    assert torch.equal(optimizer.state[param]["exp_avg"], grad * 0.01)


def test_moe_matches_dense_reference():
    torch.manual_seed(0)
    cfg = RWKVXConfig(vocab_size=32, n_embd=16, n_layer=4, head_size=4, n_moba_layer=1, is_moe=True,
                      num_experts=4, num_experts_per_tok=2)
    moe = RWKV_CMix_MoE(cfg, 0)
    x = torch.randn(2, 7, 16)
    prev = torch.randn(2, 16)

    with torch.no_grad():
        probs = torch.softmax(moe.gate(x), -1)
        topv, topi = torch.topk(probs, k=moe.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)
        dense = torch.zeros_like(x)
        prev_seq = torch.cat([prev.unsqueeze(1), x[:, :-1]], 1)
        for e_id, expert in enumerate(moe.experts):
            weight = torch.where(topi == e_id, topv, torch.zeros_like(topv)).sum(-1, keepdim=True)
            dense += expert.forward_selected(x, prev_seq) * weight
        sparse, _ = moe(x, prev)

    assert torch.allclose(sparse, dense, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    test_lowbit_roundtrip()
    test_quantized_linear_forward_is_stable()
    test_sft_dataset_caches_processed_records()
    test_native_lion_fallback_matches_lion_update()
    test_moe_matches_dense_reference()
    print("optimization tests passed")

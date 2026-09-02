import torch

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
    test_moe_matches_dense_reference()
    print("optimization tests passed")

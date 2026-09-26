import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from kernel.fp8_tile import fp8_modules
from smaul_linear import LinearConfig, SmaulLinear
from train import Lion


def test_lion_step_keeps_fp8_storage_and_finite_loss():
    torch.manual_seed(0)
    cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=1, n_heads=2, tile=32)
    model = SmaulLinear(cfg)
    opt = Lion(list(model.parameters()), lr=1e-4)
    idx = torch.randint(0, 128, (2, 16))
    opt.zero_grad(model)
    _, loss = model(idx, idx)
    assert torch.isfinite(loss)
    loss.backward()
    opt.step(model)
    for _, m in fp8_modules(model):
        assert m.w8.dtype == torch.uint8
    _, loss2 = model(idx, idx)
    assert torch.isfinite(loss2)


def test_checkpoint_round_trip_uses_same_names(tmp_path):
    torch.manual_seed(1)
    cfg = LinearConfig(vocab_size=128, d_model=64, n_layer=1, n_heads=2, tile=32)
    model = SmaulLinear(cfg)
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()
    restored = SmaulLinear.from_pretrained(tmp_path)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), restored.state_dict().items()):
        assert k1 == k2 and torch.equal(v1.cpu(), v2.cpu())

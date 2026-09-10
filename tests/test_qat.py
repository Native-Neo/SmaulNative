import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import qat
from rwkv_x_core import RWKVXConfig, RWKVXModel


def test_qat_checkpoint_restores_observers(tmp_path):
    cfg = RWKVXConfig(vocab_size=32, n_embd=32, n_layer=2, head_size=8, n_moba_layer=1)
    model = RWKVXModel(cfg)
    assert qat.prepare_qat(model) == 4

    with torch.no_grad():
        for module in qat._iter_cmix_modules(model):
            module.key.weight.add_(torch.randn_like(module.key.weight) * 0.01)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    model.eval()
    with torch.no_grad():
        expected = model(x)[0]

    model.save_pretrained(tmp_path, include_upstream=False)
    restored = RWKVXModel.from_pretrained(tmp_path)

    qat_modules = list(qat._iter_cmix_modules(restored))
    assert all(isinstance(module.key, qat.QATLinear) for module in qat_modules)
    assert all(isinstance(module.value, qat.QATLinear) for module in qat_modules)
    for original, loaded in zip(qat._iter_cmix_modules(model), qat._iter_cmix_modules(restored)):
        assert torch.equal(original.key.weight_fq.scale, loaded.key.weight_fq.scale)
        assert torch.equal(original.key.weight_fq.zero_point, loaded.key.weight_fq.zero_point)
        assert torch.equal(original.key.act_fq.scale, loaded.key.act_fq.scale)
        assert torch.equal(original.key.act_fq.zero_point, loaded.key.act_fq.zero_point)
        assert torch.equal(original.value.act_fq.scale, loaded.value.act_fq.scale)
        assert torch.equal(original.value.act_fq.zero_point, loaded.value.act_fq.zero_point)

    with torch.no_grad():
        actual = restored(x)[0]
    assert torch.equal(expected, actual)

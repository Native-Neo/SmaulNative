import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import qat
from rwkv_x_core import RWKVXConfig, RWKVXModel


def test_qat_checkpoint_restores_bit_width(tmp_path):
    cfg = RWKVXConfig(vocab_size=32, n_embd=32, n_layer=2, head_size=8, n_moba_layer=1, qat_bits=4)
    model = RWKVXModel(cfg)
    assert qat.prepare_qat(model, 4) == 4

    x = torch.randint(0, cfg.vocab_size, (1, 4))
    model.eval()
    with torch.no_grad():
        expected = model(x)[0]

    model.save_pretrained(tmp_path, include_upstream=False)
    restored = RWKVXModel.from_pretrained(tmp_path)

    qat_modules = list(qat._iter_cmix_modules(restored))
    assert all(isinstance(module.key, qat.FloatQATLinear) for module in qat_modules)
    assert all(module.key.bits == 4 and module.value.bits == 4 for module in qat_modules)

    with torch.no_grad():
        actual = restored(x)[0]
    assert torch.equal(expected, actual)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import qat
from rwkv_x_core import RWKVXConfig, RWKVXModel


def test_qat_checkpoint_loads_as_normal_model():
    cfg = RWKVXConfig(vocab_size=32, n_embd=32, n_layer=2, head_size=8, n_moba_layer=1)
    model = RWKVXModel(cfg)
    assert qat.prepare_qat(model) == 2 * 2

    state = model.state_dict()
    assert all("weight_fq" not in key and "act_fq" not in key for key in state)

    restored = RWKVXModel(cfg)
    restored.load_state_dict(state, strict=True)
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        model.eval()
        restored.eval()
        expected = model(x)[0]
        actual = restored(x)[0]
    assert torch.equal(expected, actual)

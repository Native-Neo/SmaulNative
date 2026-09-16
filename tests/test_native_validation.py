import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import cpu


def test_native_wkv_accepts_noncontiguous_inputs():
    state = torch.zeros(1, 1, 4, 4).transpose(2, 3)
    x = torch.zeros(1, 1, 2, 4).transpose(1, 2)
    out_state, y = cpu._native_wkv(state, x, x, x, x, x, x)
    assert out_state.shape == state.shape
    assert y.shape == x.shape


def test_native_wkv_rejects_invalid_head_dimension():
    state = torch.zeros(1, 1, 129, 129)
    x = torch.zeros(1, 1, 1, 129)
    with pytest.raises((ValueError, RuntimeError)):
        cpu._native_wkv(state, x, x, x, x, x, x)

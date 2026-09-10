import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import cpu


def test_native_wkv_rejects_noncontiguous_inputs():
    state = torch.zeros(1, 1, 4, 4)
    k = torch.zeros(1, 2, 1, 4).transpose(1, 2)
    with pytest.raises((ValueError, RuntimeError)):
        cpu._native_wkv(state, k, k, k, k, k, k)


def test_native_wkv_rejects_invalid_head_dimension():
    state = torch.zeros(1, 1, 3, 3)
    x = torch.zeros(1, 2, 1, 3)
    with pytest.raises((ValueError, RuntimeError)):
        cpu._native_wkv(state, x, x, x, x, x, x)

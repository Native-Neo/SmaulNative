import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import rwkv_x_core

from cpu import configure


def test_native_wkv_forward_backward():
    configure(2)
    original = rwkv_x_core._wkv_run_chunk
    import cpu
    rwkv_x_core._wkv_run_chunk = cpu._native_wkv

    torch.manual_seed(0)
    B, T, H, N = 1, 9, 2, 4
    args = [torch.randn(B, H, N, N, dtype=torch.float32, requires_grad=True)]
    args += [torch.randn(B, T, H, N, dtype=torch.float32, requires_grad=True) for _ in range(6)]
    native = rwkv_x_core._wkv_run_chunk(*args)

    ref_args = [x.detach().clone().requires_grad_(True) for x in args]
    ref = original(*ref_args)
    assert torch.allclose(native[0], ref[0], rtol=1e-5, atol=1e-5)
    assert torch.allclose(native[1], ref[1], rtol=1e-5, atol=1e-5)

    loss_n = native[0].square().mean() + native[1].square().mean()
    loss_r = ref[0].square().mean() + ref[1].square().mean()
    gn = torch.autograd.grad(loss_n, args)
    gr = torch.autograd.grad(loss_r, ref_args)
    for a, b in zip(gn, gr):
        assert torch.allclose(a, b, rtol=2e-4, atol=2e-5)

    rwkv_x_core._wkv_run_chunk = original


if __name__ == "__main__":
    test_native_wkv_forward_backward()
    print("native WKV test passed")

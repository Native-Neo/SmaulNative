import pytest
import torch

from kernel.compute import get_backend


def _native_backend():
    be = get_backend()
    if not be.has_blas_native:
        pytest.skip("AVX1 BLAS extension could not be built on this machine")
    return be


@pytest.mark.parametrize("m,k,n", [
    (1, 1, 1),
    (7, 13, 9),
    (17, 3, 67),
    (32, 64, 128),
    (128, 257, 73),
])
def test_sgemm_matches_torch(m, k, n):
    be = _native_backend()
    torch.manual_seed(1234)
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(k, n, dtype=torch.float32)
    got = be.sgemm(a, b)
    ref = a @ b
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("m,k,n", [
    (1, 1, 1),
    (7, 13, 9),
    (17, 3, 67),
    (32, 64, 128),
    (128, 257, 73),
])
def test_sgemm_bt_matches_torch(m, k, n):
    be = _native_backend()
    torch.manual_seed(5678)
    a = torch.randn(m, k, dtype=torch.float32)
    b = torch.randn(n, k, dtype=torch.float32)
    got = be.sgemm_bt(a, b)
    ref = a @ b.t()
    torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


def test_sgemm_zero_k():
    be = _native_backend()
    a = torch.empty(4, 0, dtype=torch.float32)
    b = torch.empty(0, 7, dtype=torch.float32)
    got = be.sgemm(a, b)
    assert got.shape == (4, 7)
    assert torch.count_nonzero(got) == 0


def test_dense_linear_gradients():
    _native_backend()
    from model import _DenseLinear

    torch.manual_seed(42)
    layer = _DenseLinear(13, 9, bias=True)
    ref = torch.nn.Linear(13, 9, bias=True)
    with torch.no_grad():
        ref.weight.copy_(layer.lin.weight)
        ref.bias.copy_(layer.lin.bias)

    x = torch.randn(4, 13, dtype=torch.float32, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    y = layer(x)
    yr = ref(xr)
    torch.testing.assert_close(y, yr, rtol=2e-5, atol=2e-5)

    gy = torch.randn_like(y)
    y.backward(gy)
    yr.backward(gy)
    torch.testing.assert_close(x.grad, xr.grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(layer.lin.weight.grad, ref.weight.grad, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(layer.lin.bias.grad, ref.bias.grad, rtol=2e-5, atol=2e-5)

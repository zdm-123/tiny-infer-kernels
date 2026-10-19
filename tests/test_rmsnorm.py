import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.rmsnorm import rmsnorm, rmsnorm_ref
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,cols", [(128, 1024), (64, 4096), (32, 768)])
def test_rmsnorm_matches_reference(dtype, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    y = rmsnorm(x, w, eps=1e-6)
    ref = rmsnorm_ref(x, w, eps=1e-6)
    atol = 2e-3 if dtype == torch.float16 else 1e-5
    rtol = 2e-3 if dtype == torch.float16 else 1e-5
    assert_close(y, ref, atol=atol, rtol=rtol, name="rmsnorm")


def test_rmsnorm_batched_shape():
    x = torch.randn(2, 16, 1024, device="cuda", dtype=torch.float16)
    w = torch.ones(1024, device="cuda", dtype=torch.float16)
    y = rmsnorm(x, w)
    assert y.shape == x.shape
    assert_close(y, rmsnorm_ref(x, w), atol=2e-3, rtol=2e-3, name="rmsnorm_bsh")

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.softmax import softmax, softmax_ref
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows,cols", [(128, 512), (64, 1024), (32, 2048)])
def test_softmax_matches_reference(dtype, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    y = softmax(x)
    ref = softmax_ref(x)
    atol = 2e-3 if dtype == torch.float16 else 1e-5
    rtol = 2e-3 if dtype == torch.float16 else 1e-5
    assert_close(y, ref, atol=atol, rtol=rtol, name="softmax")
    sums = y.float().sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-3)

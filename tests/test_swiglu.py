import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.swiglu import swiglu, swiglu_ref
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(128, 1024), (8, 16, 4096), (65536,)])
def test_swiglu_matches_reference(dtype, shape):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)
    atol = 2e-3 if dtype == torch.float16 else 1e-5
    rtol = 2e-3 if dtype == torch.float16 else 1e-5
    y = swiglu(gate, up)
    assert y.shape == gate.shape
    assert_close(y, swiglu_ref(gate, up), atol=atol, rtol=rtol, name="swiglu")


def test_swiglu_rejects_shape_mismatch():
    gate = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    up = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="shape mismatch"):
        swiglu(gate, up)

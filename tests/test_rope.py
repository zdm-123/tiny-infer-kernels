import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.rope import rope, rope_ref
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize(
    "shape",
    [(128, 128), (32, 64), (4, 8, 32, 128), (2, 16, 256)],
)
def test_rope_matches_reference(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    cos = torch.randn(*shape[:-1], shape[-1] // 2, device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    atol = 2e-3 if dtype == torch.float16 else 1e-5
    rtol = 2e-3 if dtype == torch.float16 else 1e-5
    assert_close(rope(x, cos, sin), rope_ref(x, cos, sin), atol=atol, rtol=rtol, name="rope")


def test_rope_broadcasts_over_batch_and_heads():
    torch.manual_seed(1)
    x = torch.randn(2, 4, 16, 128, device="cuda", dtype=torch.float16)
    cos = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    assert_close(rope(x, cos, sin), rope_ref(x, cos, sin), atol=2e-3, rtol=2e-3, name="rope_bcast")


def test_rope_rejects_odd_dim():
    x = torch.randn(4, 7, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 3, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="even"):
        rope(x, cos, torch.randn_like(cos))

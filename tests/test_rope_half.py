import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.rope import rope_rotate_half, rope_rotate_half_ref
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("dim", [64, 128])
def test_rotate_half_matches_reference(dim):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(16, dim, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    assert_close(
        rope_rotate_half(x, cos, sin),
        rope_rotate_half_ref(x, cos, sin),
        atol=1e-2,
        rtol=1e-2,
        name="rope_half",
    )

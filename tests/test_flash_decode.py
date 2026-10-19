import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.flash_attn import flash_attn_math
from tiny_infer_kernels.kernels.flash_decode import flash_decode
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("seq_k", [17, 128, 256])
@pytest.mark.parametrize("dim", [64, 128])
def test_flash_decode_matches_math(splits, seq_k, dim):
    torch.manual_seed(0)
    q = torch.randn(2, 3, 1, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 3, seq_k, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 3, seq_k, dim, device="cuda", dtype=torch.float16)
    got = flash_decode(q, k, v, num_splits=splits)
    ref = flash_attn_math(q, k, v, causal=False)
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name=f"decode_{splits}_{seq_k}_{dim}")


def test_flash_decode_rejects_prefill_query():
    q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    with pytest.raises(ValueError, match="q_len"):
        flash_decode(q, k, v)


@pytest.mark.parametrize("seq_k", [64, 2048])
def test_flash_decode_gqa_matches_sdpa(seq_k):
    torch.manual_seed(0)
    q = torch.randn(1, 14, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, seq_k, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, seq_k, 64, device="cuda", dtype=torch.float16)
    got = flash_decode(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name=f"decode_gqa_{seq_k}")

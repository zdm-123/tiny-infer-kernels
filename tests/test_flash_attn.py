import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.flash_attn import flash_attn, flash_attn_math, flash_attn_ref
from tiny_infer_kernels.utils.check import assert_close


def _case(batch, heads, seq, dim, causal, dtype=torch.float16, seq_k=None):
    torch.manual_seed(0)
    seq_k = seq if seq_k is None else seq_k
    q = torch.randn(batch, heads, seq, dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, heads, seq_k, dim, device="cuda", dtype=dtype)
    v = torch.randn(batch, heads, seq_k, dim, device="cuda", dtype=dtype)
    return q, k, v, causal


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("seq", [17, 32, 128])
def test_flash_attn_matches_math(causal, dim, seq):
    q, k, v, causal = _case(2, 3, seq, dim, causal)
    got = flash_attn(q, k, v, causal=causal)
    ref = flash_attn_math(q, k, v, causal=causal)
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name=f"fa2_{causal}_{dim}_{seq}")


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attn_matches_sdpa(causal):
    q, k, v, causal = _case(1, 4, 64, 64, causal)
    got = flash_attn(q, k, v, causal=causal)
    ref = flash_attn_ref(q, k, v, causal=causal)
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name="fa2_sdpa")


def test_flash_attn_gqa_matches_sdpa():
    torch.manual_seed(0)
    q = torch.randn(1, 8, 32, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    got = flash_attn(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name="fa2_gqa")


def test_flash_attn_rejects_indivisible_gqa():
    q = torch.randn(1, 3, 16, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 2, 16, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="multiple"):
        flash_attn(q, k, v)

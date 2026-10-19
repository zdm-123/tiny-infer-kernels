import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.kernels.w8a16_gemm import (
    dequant_weight_fp16,
    quantize_weight_w8a16,
    w8a16_gemm,
    w8a16_gemm_ref,
)
from tiny_infer_kernels.utils.check import assert_close


@pytest.mark.parametrize("m,n,k", [(1, 64, 128), (7, 96, 80), (32, 128, 256)])
def test_w8a16_matches_fp16_dequant_gemm(m, n, k):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
    w_i8, scale = quantize_weight_w8a16(weight)
    got = w8a16_gemm(a, w_i8, scale)
    ref = a @ dequant_weight_fp16(w_i8, scale).t()
    assert_close(got, ref, atol=2e-2, rtol=2e-2, name="w8a16")


def test_quantize_matches_fp32_reference_scale():
    torch.manual_seed(1)
    weight = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    a = torch.randn(4, 32, device="cuda", dtype=torch.float16)
    w_i8, scale = quantize_weight_w8a16(weight)
    got = w8a16_gemm(a, w_i8, scale)
    ref = w8a16_gemm_ref(a, w_i8, scale)
    # Kernel dequantizes in fp16; the fp32 reference is a looser bound.
    assert_close(got, ref, atol=5e-2, rtol=5e-2, name="w8a16_fp32ref")

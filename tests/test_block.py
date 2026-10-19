import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.layers.block import DecoderWeights, decoder_block
from tiny_infer_kernels.utils.check import assert_close


def _weights():
    return DecoderWeights(hidden=128, heads=2, intermediate=256, dtype=torch.float16, device="cuda")


def test_prefill_block_matches_eager():
    torch.manual_seed(0)
    weights = _weights()
    x = torch.randn(1, 16, 128, device="cuda", dtype=torch.float16)
    cos = torch.randn(16, 32, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    got = decoder_block(x, cos, sin, weights, triton_kernels=True, quantized=False)
    ref = decoder_block(x, cos, sin, weights, triton_kernels=False, quantized=False)
    assert_close(got, ref, atol=3e-2, rtol=3e-2, name="prefill_block")


def test_decode_block_matches_eager():
    torch.manual_seed(0)
    weights = _weights()
    x = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    cache_k = torch.randn(1, 2, 32, 64, device="cuda", dtype=torch.float16)
    cache_v = torch.randn_like(cache_k)
    cos = torch.randn(1, 32, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    got = decoder_block(
        x, cos, sin, weights, k_cache=cache_k, v_cache=cache_v, triton_kernels=True, quantized=False
    )
    ref = decoder_block(
        x, cos, sin, weights, k_cache=cache_k, v_cache=cache_v, triton_kernels=False, quantized=False
    )
    assert_close(got, ref, atol=3e-2, rtol=3e-2, name="decode_block")

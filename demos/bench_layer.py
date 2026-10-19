"""Time one decoder block: eager fp16, Triton fp16, Triton + W8A16.

Weights are random. This measures kernel integration, not a published checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.layers.block import DecoderWeights, decoder_block
from tiny_infer_kernels.utils.bench import bench_ms
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


def _time(fn) -> float:
    return bench_ms(fn, warmup=10, rep=30)


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    weights = DecoderWeights(hidden=1024, heads=8, intermediate=2816, dtype=torch.float16, device="cuda")
    x = torch.randn(1, 256, 1024, device="cuda", dtype=torch.float16)
    cos = torch.randn(256, 64, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)
    eager = decoder_block(x, cos, sin, weights, triton_kernels=False)
    tri = decoder_block(x, cos, sin, weights, triton_kernels=True)
    print(f"prefill max_abs_err eager vs triton: {max_abs_err(tri, eager):.4e}")
    print(f"prefill eager   {_time(lambda: decoder_block(x, cos, sin, weights, triton_kernels=False)):8.3f} ms")
    print(f"prefill triton  {_time(lambda: decoder_block(x, cos, sin, weights, triton_kernels=True)):8.3f} ms")
    print(f"prefill w8a16   {_time(lambda: decoder_block(x, cos, sin, weights, triton_kernels=True, quantized=True)):8.3f} ms")

    x1 = torch.randn(1, 1, 1024, device="cuda", dtype=torch.float16)
    cache_k = torch.randn(1, 8, 256, 128, device="cuda", dtype=torch.float16)
    cache_v = torch.randn_like(cache_k)
    cos1 = torch.randn(1, 64, device="cuda", dtype=torch.float16)
    sin1 = torch.randn_like(cos1)
    kwargs = dict(k_cache=cache_k, v_cache=cache_v)
    eager_d = decoder_block(x1, cos1, sin1, weights, triton_kernels=False, **kwargs)
    tri_d = decoder_block(x1, cos1, sin1, weights, triton_kernels=True, **kwargs)
    print(f"decode  max_abs_err eager vs triton: {max_abs_err(tri_d, eager_d):.4e}")
    print(f"decode  eager   {_time(lambda: decoder_block(x1, cos1, sin1, weights, triton_kernels=False, **kwargs)):8.3f} ms")
    print(f"decode  triton  {_time(lambda: decoder_block(x1, cos1, sin1, weights, triton_kernels=True, **kwargs)):8.3f} ms")
    print(
        "decode  w8a16   "
        f"{_time(lambda: decoder_block(x1, cos1, sin1, weights, triton_kernels=True, quantized=True, **kwargs)):8.3f} ms"
    )


if __name__ == "__main__":
    main()

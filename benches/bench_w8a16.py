"""W8A16 GEMM vs an fp16 weight that is already in HBM."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.w8a16_gemm import (
    dequant_weight_fp16,
    quantize_weight_w8a16,
    w8a16_gemm,
    w8a16_weight_bytes,
)
from tiny_infer_kernels.utils.bench import bench_ms, tflops
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


CASES = [
    (1, 4096, 4096),
    (8, 4096, 4096),
    (64, 4096, 4096),
    (256, 4096, 4096),
]


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")
    print(f"{'shape':>18}  {'impl':<12}  {'ms':>8}  {'TFLOPS':>8}  {'err':>10}")
    for m, n, k in CASES:
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.float16)
        w_i8, scale = quantize_weight_w8a16(weight)
        w_fp16 = dequant_weight_fp16(w_i8, scale)
        fp16_bytes, int8_bytes = w8a16_weight_bytes(n, k)
        flops = 2.0 * m * n * k
        err = max_abs_err(w8a16_gemm(a, w_i8, scale), a @ w_fp16.t())
        label = f"M{m} N{n} K{k}"
        for name, fn in [
            ("fp16_gemm", lambda: a @ w_fp16.t()),
            ("w8a16", lambda: w8a16_gemm(a, w_i8, scale)),
        ]:
            ms = bench_ms(fn)
            print(
                f"{label:>18}  {name:<12}  {ms:8.3f}  {tflops(flops, ms):8.2f}  {err:10.2e}"
                f"  W_bytes {fp16_bytes if name == 'fp16_gemm' else int8_bytes}"
            )
        print()


if __name__ == "__main__":
    main()

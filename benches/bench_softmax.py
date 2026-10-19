"""Fused softmax latency / effective bandwidth vs PyTorch softmax."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.softmax import softmax, softmax_bytes, softmax_ref
from tiny_infer_kernels.utils.bench import bench_ms, effective_bandwidth_gbps
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


SHAPES = [
    (4096, 512),
    (4096, 1024),
    (4096, 2048),
    (4096, 4096),
]


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    print(f"{'shape':>16}  {'impl':<18}  {'ms':>8}  {'GB/s':>8}  {'err':>10}")
    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        nbytes = softmax_bytes(rows, cols, x.dtype)

        eager = lambda: softmax_ref(x)
        tri = lambda: softmax(x)
        err = max_abs_err(tri(), eager())

        for name, fn, extra_err in [
            ("pytorch_softmax", eager, 0.0),
            ("triton_fused", tri, err),
        ]:
            ms = bench_ms(fn)
            gbps = effective_bandwidth_gbps(nbytes, ms)
            print(f"{rows}x{cols:>4}  {name:<18}  {ms:8.3f}  {gbps:8.1f}  {extra_err:10.2e}")
        print()


if __name__ == "__main__":
    main()

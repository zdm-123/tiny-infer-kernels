"""FlashAttention-2 prefill vs PyTorch SDPA, reported in TFLOPS."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.flash_attn import (
    flash_attn,
    flash_attn_flops,
    flash_attn_math,
    flash_attn_ref,
)
from tiny_infer_kernels.utils.bench import bench_ms, tflops
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


# batch, heads, seq, dim. Kept small enough for a 12 GB card.
SHAPES = [
    (1, 8, 1024, 128),
    (1, 8, 2048, 128),
    (2, 8, 2048, 128),
    (1, 8, 4096, 128),
]


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    print(f"{'shape':>22}  {'impl':<12}  {'ms':>8}  {'TFLOPS':>8}  {'err':>10}")
    for batch, heads, seq, dim in SHAPES:
        q = torch.randn(batch, heads, seq, dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        flops = flash_attn_flops(batch, heads, seq, seq, dim, causal=True)
        label = f"B{batch} H{heads} S{seq} D{dim}"

        # Correctness spot-check on a short prefix so the math reference fits in memory.
        n = min(128, seq)
        err = max_abs_err(
            flash_attn(q[:, :, :n], k[:, :, :n], v[:, :, :n], causal=True),
            flash_attn_math(q[:, :, :n], k[:, :, :n], v[:, :, :n], causal=True),
        )

        for name, fn in [
            ("sdpa", lambda: flash_attn_ref(q, k, v, causal=True)),
            ("triton", lambda: flash_attn(q, k, v, causal=True)),
        ]:
            ms = bench_ms(fn)
            print(f"{label:>22}  {name:<12}  {ms:8.3f}  {tflops(flops, ms):8.2f}  {err:10.2e}")
        print()


if __name__ == "__main__":
    main()

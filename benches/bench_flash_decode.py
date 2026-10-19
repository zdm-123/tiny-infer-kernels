"""Decode attention: Flash-Decoding vs prefill FA2 and SDPA at q_len=1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.flash_attn import flash_attn, flash_attn_ref
from tiny_infer_kernels.kernels.flash_decode import choose_num_splits, flash_decode
from tiny_infer_kernels.utils.bench import bench_ms
from tiny_infer_kernels.utils.device import gpu_summary


SHAPES = [
    (1, 8, 4096, 128),
    (1, 8, 8192, 128),
    (1, 32, 4096, 128),
    (4, 8, 4096, 128),
]


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")
    sms = info["multi_processor_count"]
    print(f"{'shape':>22}  {'splits':>6}  {'impl':<14}  {'ms':>8}")
    for batch, heads, seq_k, dim in SHAPES:
        q = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq_k, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        splits = choose_num_splits(batch, heads, seq_k, sms)
        label = f"B{batch} H{heads} Sk{seq_k}"
        for name, fn in [
            ("sdpa", lambda: flash_attn_ref(q, k, v, causal=False)),
            ("fa2_prefill", lambda: flash_attn(q, k, v, causal=False)),
            ("flash_decode", lambda: flash_decode(q, k, v, num_splits=splits)),
        ]:
            ms = bench_ms(fn)
            print(f"{label:>22}  {splits:6d}  {name:<14}  {ms:8.3f}")
        print()


if __name__ == "__main__":
    main()

"""Native GQA decode versus expanding KV to the query head count.

Qwen2.5-0.5B has 14 query heads and 2 KV heads. Expanding the cache
multiplies KV traffic by 7. The fused kernel loads each K/V tile once.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tiny_infer_kernels.kernels.flash_decode import flash_decode
from tiny_infer_kernels.models.qwen2 import repeat_kv
from tiny_infer_kernels.utils.bench import bench_ms
from tiny_infer_kernels.utils.device import gpu_summary


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    batch, q_heads, kv_heads, dim = 1, 14, 2, 64
    print(f"{'seq':>8}  {'sdpa':>8}  {'repeat':>8}  {'native':>8}")
    for seq_k in (128, 1024, 4096, 8192):
        q = torch.randn(batch, q_heads, 1, dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, kv_heads, seq_k, dim, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)
        k_rep = repeat_kv(k, q_heads // kv_heads).contiguous()
        v_rep = repeat_kv(v, q_heads // kv_heads).contiguous()
        sdpa = bench_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True))
        repeated = bench_ms(lambda: flash_decode(q, k_rep, v_rep))
        native = bench_ms(lambda: flash_decode(q, k, v))
        print(f"{seq_k:8d}  {sdpa:8.3f}  {repeated:8.3f}  {native:8.3f}")


if __name__ == "__main__":
    main()

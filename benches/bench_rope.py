"""RoPE latency / effective bandwidth vs PyTorch eager and torch.compile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.rope import rope, rope_bytes, rope_ref
from tiny_infer_kernels.utils.bench import bench_ms, effective_bandwidth_gbps
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


# (rows, dim). dim is a head size; rows stand in for batch * heads * seq.
SHAPES = [
    (8192, 64),
    (8192, 128),
    (16384, 128),
    (32768, 128),
]


def _maybe_compiled(fn):
    try:
        return torch.compile(fn, mode="reduce-overhead")
    except Exception:
        return None


def main() -> None:
    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    compiled = _maybe_compiled(lambda x, cos, sin: rope_ref(x, cos, sin))
    print(f"{'shape':>16}  {'impl':<18}  {'ms':>8}  {'GB/s':>8}  {'err':>10}")

    for rows, dim in SHAPES:
        x = torch.randn(rows, dim, device="cuda", dtype=torch.float16)
        cos = torch.randn(rows, dim // 2, device="cuda", dtype=torch.float16)
        sin = torch.randn_like(cos)
        nbytes = rope_bytes(x.numel(), x.dtype)

        eager = lambda: rope_ref(x, cos, sin)
        tri = lambda: rope(x, cos, sin)
        err = max_abs_err(tri(), eager())

        for name, fn, extra_err in [
            ("pytorch_eager", eager, 0.0),
            ("triton", tri, err),
        ]:
            ms = bench_ms(fn)
            gbps = effective_bandwidth_gbps(nbytes, ms)
            print(f"{rows}x{dim:>4}  {name:<18}  {ms:8.3f}  {gbps:8.1f}  {extra_err:10.2e}")

        if compiled is not None:
            _ = compiled(x, cos, sin)
            torch.cuda.synchronize()
            ms = bench_ms(lambda: compiled(x, cos, sin))
            gbps = effective_bandwidth_gbps(nbytes, ms)
            print(f"{rows}x{dim:>4}  {'torch.compile':<18}  {ms:8.3f}  {gbps:8.1f}  {'n/a':>10}")
        print()


if __name__ == "__main__":
    main()

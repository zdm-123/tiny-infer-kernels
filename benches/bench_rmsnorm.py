"""RMSNorm latency / effective bandwidth vs PyTorch eager and torch.compile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.kernels.rmsnorm import rmsnorm, rmsnorm_bytes, rmsnorm_ref
from tiny_infer_kernels.utils.bench import bench_ms, effective_bandwidth_gbps
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


SHAPES = [
    (4096, 1024),
    (4096, 2048),
    (4096, 4096),
    (8192, 4096),
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

    compiled = _maybe_compiled(lambda x, w: rmsnorm_ref(x, w))
    print(f"{'shape':>16}  {'impl':<18}  {'ms':>8}  {'GB/s':>8}  {'err':>10}")

    for rows, cols in SHAPES:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float16)
        w = torch.randn(cols, device="cuda", dtype=torch.float16)
        nbytes = rmsnorm_bytes(rows, cols, x.dtype)

        eager = lambda: rmsnorm_ref(x, w)
        tri = lambda: rmsnorm(x, w)

        ref = eager()
        got = tri()
        err = max_abs_err(got, ref)

        for name, fn, extra_err in [
            ("pytorch_eager", eager, 0.0),
            ("triton", tri, err),
        ]:
            ms = bench_ms(fn)
            gbps = effective_bandwidth_gbps(nbytes, ms)
            print(f"{rows}x{cols:>4}  {name:<18}  {ms:8.3f}  {gbps:8.1f}  {extra_err:10.2e}")

        if compiled is not None:
            # Warm compile once, then bench.
            _ = compiled(x, w)
            torch.cuda.synchronize()
            ms = bench_ms(lambda: compiled(x, w))
            gbps = effective_bandwidth_gbps(nbytes, ms)
            print(f"{rows}x{cols:>4}  {'torch.compile':<18}  {ms:8.3f}  {gbps:8.1f}  {'n/a':>10}")
        print()


if __name__ == "__main__":
    main()

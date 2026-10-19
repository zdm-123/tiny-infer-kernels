"""Benchmark helpers.

Interview talking point: never quote a single kernel time. Report
latency, effective bandwidth (or TFLOPS), and the baseline it beat.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import triton


def bench_ms(fn: Callable[[], object], warmup: int = 25, rep: int = 100) -> float:
    """Median kernel time in milliseconds via Triton's CUDA-event timer."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to benchmark kernels")
    torch.cuda.synchronize()
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def effective_bandwidth_gbps(nbytes: int, ms: float) -> float:
    """nbytes / time, in GB/s. Use this for memory-bound ops (RMSNorm, softmax)."""
    if ms <= 0:
        return 0.0
    return nbytes * 1e-9 / (ms * 1e-3)


def tflops(flop_count: float, ms: float) -> float:
    """Floating-point ops / time, in TFLOPS. Use this for GEMM / attention."""
    if ms <= 0:
        return 0.0
    return flop_count * 1e-12 / (ms * 1e-3)

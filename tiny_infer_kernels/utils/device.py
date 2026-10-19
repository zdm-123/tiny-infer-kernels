"""GPU / Roofline helpers.

Roofline in one sentence: achieved performance is min(peak FLOPS,
arithmetic_intensity * peak_bandwidth). If an op is memory-bound,
fusion that cuts HBM traffic is the main win — not extra math.
"""

from __future__ import annotations

from typing import Any

import torch


def cuda_available() -> bool:
    return torch.cuda.is_available()


def gpu_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "memory_gb": round(props.total_memory / (1024**3), 2),
        "multi_processor_count": props.multi_processor_count,
        "warp_size": getattr(props, "warp_size", 32),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

"""Environment self-check. Run inside the project venv:

    source /home/zdm/project1/pytorch/bin/activate
    python scripts/check_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")

    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        return 1

    print(f"torch:  {torch.__version__}")
    print(f"cuda:   {torch.version.cuda}  available={torch.cuda.is_available()}")

    try:
        import triton
    except ImportError:
        print("triton: NOT INSTALLED")
        return 1

    print(f"triton: {triton.__version__}")

    from tiny_infer_kernels.utils.device import gpu_summary

    info = gpu_summary()
    if not info.get("cuda"):
        print("GPU:    no CUDA device visible from this process")
        print("Hint:   in WSL, `nvidia-smi` must work; install NVIDIA WSL driver on Windows.")
        return 1

    print(f"gpu:    {info['name']}  sm{info['capability']}  {info['memory_gb']} GB  SMs={info['multi_processor_count']}")

    # Tiny compile + launch smoke test.
    x = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    w = torch.ones(8, device="cuda", dtype=torch.float16)
    from tiny_infer_kernels.kernels.rmsnorm import rmsnorm, rmsnorm_ref

    y = rmsnorm(x, w)
    ref = rmsnorm_ref(x, w)
    err = (y.float() - ref.float()).abs().max().item()
    print(f"smoke:  rmsnorm max_abs_err={err:.4e}")
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

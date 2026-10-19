"""Fused SwiGLU elementwise in Triton.

    y = silu(gate) * up
    silu(g) = g * sigmoid(g)

The two GEMMs that produce ``gate`` and ``up`` stay separate (Module 4 can
fuse dequant into those GEMMs). This kernel only removes the extra HBM
round-trip of the activation:

    eager:  read gate, write silu(gate), read silu, read up, write y
    fused:  read gate, read up, write y
            bytes = 3 * numel * element_size
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask, other=0.0)
    u = tl.load(up_ptr + offs, mask=mask, other=0.0)
    gf = g.to(tl.float32)
    uf = u.to(tl.float32)
    silu = gf * tl.sigmoid(gf)
    y = silu * uf
    tl.store(y_ptr + offs, y.to(g.dtype), mask=mask)


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up``. ``gate`` and ``up`` must have the same shape."""
    if not gate.is_cuda:
        raise ValueError("swiglu expects CUDA tensors")
    if gate.shape != up.shape:
        raise ValueError(f"shape mismatch {tuple(gate.shape)} vs {tuple(up.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"dtype mismatch {gate.dtype} vs {up.dtype}")

    gate_f = gate.contiguous().view(-1)
    up_f = up.contiguous().view(-1)
    y = torch.empty_like(gate_f)
    n = gate_f.numel()
    block = 1024
    _swiglu_kernel[(triton.cdiv(n, block),)](
        gate_f,
        up_f,
        y,
        n,
        BLOCK=block,
        num_warps=4,
    )
    return y.view_as(gate)


def swiglu_bytes(numel: int, dtype: torch.dtype) -> int:
    """Fused traffic: read gate, read up, write y."""
    es = torch.empty((), dtype=dtype).element_size()
    return numel * es * 3

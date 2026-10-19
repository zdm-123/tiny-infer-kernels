"""Fused row-wise softmax in Triton.

Naive softmax traffic for an M x N fp32 matrix (official Triton tutorial):
    read x, write x, read x, write row-max, read x, write exp, read exp,
    write row-sum, read exp, write y
    ≈ 8MN + 4M bytes  vs  fused  2MN bytes   →  theoretical ~4x

Limitation of this kernel: each row must fit in on-chip memory
(BLOCK_SIZE = next_power_of_2(N)). Long sequence softmax belongs in
FlashAttention (online softmax), not here. That is an interview talking point.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    y_ptr,
    x_ptr,
    x_stride,
    y_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=-float("inf"))
    x_f = x.to(tl.float32)
    x_f = x_f - tl.max(x_f, axis=0)
    num = tl.exp(x_f)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(y_ptr + row * y_stride + cols, y.to(x.dtype), mask=mask)


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Fused softmax over the last dimension. Rows must fit in SRAM."""
    if not x.is_cuda:
        raise ValueError("softmax expects a CUDA tensor")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    y_2d = torch.empty_like(x_2d)
    n_rows, n_cols = x_2d.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 16384:
        raise ValueError(
            f"n_cols={n_cols} is too large for the one-tile softmax kernel. "
            "Use online softmax / FlashAttention for long sequences."
        )

    # Same occupancy heuristic as the official fused-softmax tutorial.
    num_warps = 4
    if block_size >= 2048:
        num_warps = 8
    if block_size >= 4096:
        num_warps = 16

    _softmax_kernel[(n_rows,)](
        y_2d,
        x_2d,
        x_2d.stride(0),
        y_2d.stride(0),
        n_cols,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return y_2d.view_as(x)


def softmax_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    """Fused kernel traffic: read x once, write y once."""
    es = torch.empty((), dtype=dtype).element_size()
    return rows * cols * es * 2

"""Fused RMSNorm in Triton.

Definition (same as Llama / Qwen):

    y = x * rsqrt(mean(x^2) + eps) * weight

Memory accounting (fp16, rows=M, cols=N):
    naive PyTorch: read x, write x^2, read x^2, write var, read x, read w, write y
    fused kernel:  read x once, read w once, write y once
        bytes ≈ 2*M*N*2 + N*2  (input + output + weight)

This is a memory-bound kernel. The win is fewer HBM round-trips, not extra FLOPs.
The implementation tiles along N so long hidden sizes (4096 / 8192) still fit.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    n_cols,
    eps,
    x_stride,
    y_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_stride
    y_row = y_ptr + row * y_stride

    # Pass 1: sum of squares in fp32 for numerical stability.
    sum_sq = 0.0
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rstd = tl.rsqrt(sum_sq / n_cols + eps)

    # Pass 2: normalize, scale by weight, write.
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x.to(tl.float32) * rstd * w.to(tl.float32)
        tl.store(y_row + cols, y.to(x.dtype), mask=mask)


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference used for correctness. Keep this tiny and obvious."""
    orig_dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    y = x_f * torch.rsqrt(var + eps) * weight.float()
    return y.to(orig_dtype)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused RMSNorm over the last dimension. `x` is reshaped to (M, N)."""
    if not x.is_cuda:
        raise ValueError("rmsnorm expects a CUDA tensor")
    if weight.ndim != 1 or weight.numel() != x.shape[-1]:
        raise ValueError(f"weight must be 1-D of length {x.shape[-1]}")

    x_2d = x.contiguous().view(-1, x.shape[-1])
    weight = weight.contiguous()
    y_2d = torch.empty_like(x_2d)

    n_rows, n_cols = x_2d.shape
    _rmsnorm_kernel[(n_rows,)](
        x_2d,
        weight,
        y_2d,
        n_cols,
        eps,
        x_2d.stride(0),
        y_2d.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4 if n_cols <= 2048 else 8,
    )
    return y_2d.view_as(x)


def rmsnorm_bytes(rows: int, cols: int, dtype: torch.dtype) -> int:
    """HBM traffic of the fused kernel: read x, read w, write y."""
    es = torch.empty((), dtype=dtype).element_size()
    return rows * cols * es * 2 + cols * es

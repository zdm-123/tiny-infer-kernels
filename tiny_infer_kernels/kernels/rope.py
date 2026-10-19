"""Interleaved RoPE in Triton.

Pair layout (even, odd), matching ``rope_ref``:

    y_even = x_even * cos - x_odd * sin
    y_odd  = x_even * sin + x_odd * cos

Llama / Qwen use a different pairing (``rotate_half``: first half vs second
half, with cos/sin duplicated to the full head dim). The arithmetic is the
same rotation; only the index of the partner element changes. This kernel
follows the even/odd form already used by the reference.

Memory (one token row, dim D, element size E):
    fused: read x + read cos + read sin + write y
         = 2*D*E + 2*(D/2)*E = 3*D*E
    eager: slice, several muls/adds, stack — multiple HBM round-trips.

Head dim is small (64–256), so one program per token keeps the row on chip.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    n_pairs,
    x_stride,
    cos_stride,
    sin_stride,
    y_stride,
    BLOCK_PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * x_stride
    y_row = y_ptr + row * y_stride
    cos_row = cos_ptr + row * cos_stride
    sin_row = sin_ptr + row * sin_stride

    for off in range(0, n_pairs, BLOCK_PAIRS):
        pairs = off + tl.arange(0, BLOCK_PAIRS)
        mask = pairs < n_pairs
        x1 = tl.load(x_row + pairs * 2, mask=mask, other=0.0)
        x2 = tl.load(x_row + pairs * 2 + 1, mask=mask, other=0.0)
        c = tl.load(cos_row + pairs, mask=mask, other=0.0)
        s = tl.load(sin_row + pairs, mask=mask, other=0.0)

        x1f = x1.to(tl.float32)
        x2f = x2.to(tl.float32)
        cf = c.to(tl.float32)
        sf = s.to(tl.float32)
        y1 = x1f * cf - x2f * sf
        y2 = x1f * sf + x2f * cf
        tl.store(y_row + pairs * 2, y1.to(x1.dtype), mask=mask)
        tl.store(y_row + pairs * 2 + 1, y2.to(x2.dtype), mask=mask)


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., dim) even dim. cos/sin broadcastable to (..., dim/2)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y.to(x.dtype)


def _as_pair_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {x.shape[-1]}")
    x = x.contiguous()
    pair_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    try:
        cos_b = torch.broadcast_to(cos, pair_shape)
        sin_b = torch.broadcast_to(sin, pair_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"cos/sin shape {tuple(cos.shape)} / {tuple(sin.shape)} "
            f"cannot broadcast to {tuple(pair_shape)}"
        ) from exc
    return x, cos_b.contiguous(), sin_b.contiguous()


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved RoPE. ``cos``/``sin`` broadcast like ``rope_ref``."""
    if not x.is_cuda:
        raise ValueError("rope expects a CUDA tensor")
    x, cos_b, sin_b = _as_pair_tables(x, cos, sin)
    dim = x.shape[-1]
    rows = x.numel() // dim
    x_2d = x.view(rows, dim)
    cos_2d = cos_b.view(rows, dim // 2)
    sin_2d = sin_b.view(rows, dim // 2)
    y_2d = torch.empty_like(x_2d)

    n_pairs = dim // 2
    block_pairs = min(128, triton.next_power_of_2(n_pairs))
    _rope_kernel[(rows,)](
        x_2d,
        cos_2d,
        sin_2d,
        y_2d,
        n_pairs,
        x_2d.stride(0),
        cos_2d.stride(0),
        sin_2d.stride(0),
        y_2d.stride(0),
        BLOCK_PAIRS=block_pairs,
        num_warps=4,
    )
    return y_2d.view_as(x)


def rope_bytes(numel: int, dtype: torch.dtype) -> int:
    """Fused traffic: read x, cos, sin and write y. cos+sin together equal one x."""
    es = torch.empty((), dtype=dtype).element_size()
    return numel * es * 3


@triton.jit
def _rope_half_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    y_ptr,
    half,
    x_stride,
    cos_stride,
    sin_stride,
    y_stride,
    BLOCK: tl.constexpr,
):
    """Llama/Qwen layout: first half pairs with second half, not even/odd."""
    row = tl.program_id(0)
    x_row = x_ptr + row * x_stride
    y_row = y_ptr + row * y_stride
    cos_row = cos_ptr + row * cos_stride
    sin_row = sin_ptr + row * sin_stride
    for off in range(0, half, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < half
        x1 = tl.load(x_row + idx, mask=mask, other=0.0)
        x2 = tl.load(x_row + half + idx, mask=mask, other=0.0)
        c1 = tl.load(cos_row + idx, mask=mask, other=0.0)
        c2 = tl.load(cos_row + half + idx, mask=mask, other=0.0)
        s1 = tl.load(sin_row + idx, mask=mask, other=0.0)
        s2 = tl.load(sin_row + half + idx, mask=mask, other=0.0)
        x1f = x1.to(tl.float32)
        x2f = x2.to(tl.float32)
        y1 = x1f * c1.to(tl.float32) - x2f * s1.to(tl.float32)
        y2 = x2f * c2.to(tl.float32) + x1f * s2.to(tl.float32)
        tl.store(y_row + idx, y1.to(x1.dtype), mask=mask)
        tl.store(y_row + half + idx, y2.to(x2.dtype), mask=mask)


def rope_rotate_half_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``y = x * cos + rotate_half(x) * sin``. cos/sin broadcast to ``x``."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    y1 = x1 * cos[..., :half] - x2 * sin[..., :half]
    y2 = x2 * cos[..., half:] + x1 * sin[..., half:]
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _as_full_tables(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {x.shape[-1]}")
    x = x.contiguous()
    try:
        cos_b = torch.broadcast_to(cos, x.shape)
        sin_b = torch.broadcast_to(sin, x.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"cos/sin shape {tuple(cos.shape)} / {tuple(sin.shape)} "
            f"cannot broadcast to {tuple(x.shape)}"
        ) from exc
    return x, cos_b.contiguous(), sin_b.contiguous()


def rope_rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Qwen/Llama RoPE. ``cos`` and ``sin`` have the same last dim as ``x``."""
    if not x.is_cuda:
        raise ValueError("rope_rotate_half expects a CUDA tensor")
    x, cos_b, sin_b = _as_full_tables(x, cos, sin)
    dim = x.shape[-1]
    rows = x.numel() // dim
    x_2d = x.view(rows, dim)
    cos_2d = cos_b.view(rows, dim)
    sin_2d = sin_b.view(rows, dim)
    y_2d = torch.empty_like(x_2d)
    half = dim // 2
    block = min(128, triton.next_power_of_2(half))
    _rope_half_kernel[(rows,)](
        x_2d,
        cos_2d,
        sin_2d,
        y_2d,
        half,
        x_2d.stride(0),
        cos_2d.stride(0),
        sin_2d.stride(0),
        y_2d.stride(0),
        BLOCK=block,
        num_warps=4,
    )
    return y_2d.view_as(x)


def build_rope_cache(
    seq: int,
    head_dim: int,
    theta: float,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicated-frequency cos/sin used by Qwen and Llama. Shape ``(seq, head_dim)``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

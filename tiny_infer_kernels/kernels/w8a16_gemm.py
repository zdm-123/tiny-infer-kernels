"""W8A16 weight-only GEMM.

``weight`` is stored as int8 with one scale per output channel. The kernel
loads that int8 tile, dequantizes it in registers, and feeds ``tl.dot``.
The fp16 weight tensor is never materialized.

    y[m, n] = sum_k a[m, k] * (w_int8[n, k] * scale[n])

Traffic for the weight, which dominates decode GEMMs (small M):
    fp16 weight: 2 * N * K bytes
    int8 weight: 1 * N * K bytes
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _configs():
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def _w8a16_kernel(
    a_ptr,
    w_ptr,
    s_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    OUT_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    scale = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float16)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left),
            other=0.0,
        )
        w = tl.load(
            w_ptrs,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_left),
            other=0,
        )
        w_f = w.to(tl.float16) * scale[:, None]
        acc += tl.dot(a, tl.trans(w_f))
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    if OUT_FP16:
        c = acc.to(tl.float16)
    else:
        c = acc.to(tl.bfloat16)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def quantize_weight_w8a16(weight_fp16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8 quant. weight: (N, K)."""
    if weight_fp16.ndim != 2:
        raise ValueError("weight must be (N, K)")
    w = weight_fp16.float()
    scale = w.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale.to(weight_fp16.dtype)


def w8a16_gemm_ref(
    a: torch.Tensor,
    w_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """a: (M, K) fp16, w_int8: (N, K) int8, scale: (N,) -> (M, N)."""
    w = w_int8.float() * scale[:, None].float()
    return (a.float() @ w.t()).to(a.dtype)


def dequant_weight_fp16(w_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Materialize the fp16 weight the kernel dequantizes on the fly."""
    return (w_int8.float() * scale.float()[:, None]).to(dtype=scale.dtype)


def w8a16_gemm(
    a: torch.Tensor,
    w_int8: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Fused int8 dequant + GEMM. ``a`` is (M, K), ``w_int8`` is (N, K)."""
    if not a.is_cuda:
        raise ValueError("w8a16_gemm expects CUDA tensors")
    if a.ndim != 2 or w_int8.ndim != 2:
        raise ValueError("a must be (M, K) and w_int8 must be (N, K)")
    if a.shape[1] != w_int8.shape[1]:
        raise ValueError(f"K mismatch {a.shape[1]} vs {w_int8.shape[1]}")
    if w_int8.dtype != torch.int8:
        raise ValueError("w_int8 must be int8")
    if scale.numel() != w_int8.shape[0]:
        raise ValueError("scale must have one value per output channel")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("a must be float16 or bfloat16")

    a = a.contiguous()
    w_int8 = w_int8.contiguous()
    scale = scale.to(dtype=a.dtype).contiguous()
    m, k = a.shape
    n = w_int8.shape[0]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),)
    _w8a16_kernel[grid](
        a, w_int8, scale, c,
        m, n, k,
        a.stride(0), a.stride(1),
        w_int8.stride(0), w_int8.stride(1),
        c.stride(0), c.stride(1),
        OUT_FP16=a.dtype == torch.float16,
    )
    return c


def w8a16_weight_bytes(n: int, k: int) -> tuple[int, int]:
    """(fp16 weight bytes, int8 weight bytes) ignoring the tiny scale vector."""
    return n * k * 2, n * k

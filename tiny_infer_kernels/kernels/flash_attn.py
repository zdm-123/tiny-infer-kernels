"""FlashAttention-2 prefill forward in Triton.

Online softmax over K/V tiles so the S x S score matrix is never written
to HBM. One program owns one query tile of one (batch, head).

    m_new = max(m, rowmax(Q K^T * scale))
    p     = exp(scores - m_new)
    l     = l * exp(m - m_new) + rowsum(p)
    O     = O * exp(m - m_new) + p @ V

Scores stay in fp32. Q/K/V dots use the input dtype (fp16 tensor cores).
Causal masking matches PyTorch SDPA: bottom-right aligned, so when
``seq_q == seq_k`` a query only sees keys at positions ``<=`` its own.
Fully masked tiles before the diagonal are not visited.

Head dim must be 32, 64, 128, or 256. Grouped-query attention reads the
KV head ``q_head // (Hq / Hkv)`` directly, so the cache is not expanded.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)


def _attn_configs():
    return [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_attn_configs(), key=["seq_q", "seq_k", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    seq_q,
    seq_k,
    causal_shift,
    H,
    KV_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    off_kv = off_h // KV_GROUP

    q_base = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_base = off_b.to(tl.int64) * stride_kb + off_kv.to(tl.int64) * stride_kh
    v_base = off_b.to(tl.int64) * stride_vb + off_kv.to(tl.int64) * stride_vh
    o_base = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(
        q_ptr + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=offs_m[:, None] < seq_q,
        other=0.0,
    )

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if CAUSAL:
        k_limit = tl.minimum(seq_k, (pid_m + 1) * BLOCK_M + causal_shift)
    else:
        k_limit = seq_k

    for start_n in range(0, k_limit, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(
            k_ptr + k_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn,
            mask=offs_n[None, :] < seq_k,
            other=0.0,
        )
        qk = tl.dot(q, k).to(tl.float32) * sm_scale
        keep = offs_n[None, :] < seq_k
        keep = keep & (offs_m[:, None] < seq_q)
        if CAUSAL:
            keep = keep & (offs_n[None, :] <= offs_m[:, None] + causal_shift)
        qk = tl.where(keep, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.exp(qk - m_ij_safe[:, None])
        p = tl.where(m_ij[:, None] == float("-inf"), 0.0, p)
        alpha = tl.exp(m_i - m_ij_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=offs_n[:, None] < seq_k,
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_ij

    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    tl.store(
        o_ptr + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        out.to(q.dtype),
        mask=offs_m[:, None] < seq_q,
    )


def flash_attn_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Exact attention in fp32. Materializes the score matrix; tests only."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * sm_scale
    if causal:
        seq_q = q.shape[-2]
        seq_k = k.shape[-2]
        q_pos = torch.arange(seq_q, device=q.device)[:, None]
        k_pos = torch.arange(seq_k, device=q.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos + (seq_k - seq_q), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return torch.matmul(probs, v.float()).to(dtype=q.dtype)


def flash_attn_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=sm_scale)


def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Self-attention prefill. ``q, k, v`` are ``(batch, heads, seq, dim)``."""
    if not q.is_cuda:
        raise ValueError("flash_attn expects CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 (batch, heads, seq, dim)")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("batch mismatch")
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if kv_heads != v.shape[1]:
        raise ValueError(f"k/v head mismatch {kv_heads} vs {v.shape[1]}")
    if q_heads % kv_heads != 0:
        raise ValueError(f"query heads {q_heads} must be a multiple of kv heads {kv_heads}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v dtypes must match")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_attn supports float16 and bfloat16")
    head_dim = q.shape[-1]
    if head_dim != k.shape[-1] or head_dim != v.shape[-1]:
        raise ValueError("head dim mismatch")
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(f"head dim {head_dim} not in {_SUPPORTED_HEAD_DIMS}")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v sequence length mismatch")

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, q_heads, seq_q, _ = q.shape
    seq_k = k.shape[-2]
    kv_group = q_heads // kv_heads
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(seq_q, meta["BLOCK_M"]), batch * q_heads)
    _attn_fwd[grid](
        q,
        k,
        v,
        out,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        seq_q,
        seq_k,
        seq_k - seq_q,
        q_heads,
        KV_GROUP=kv_group,
        HEAD_DIM=head_dim,
        CAUSAL=causal,
    )
    return out


def flash_attn_flops(batch: int, heads: int, seq_q: int, seq_k: int, dim: int, causal: bool) -> float:
    """QK^T and PV matmuls. Causal with equal lengths counts half the scores."""
    flops = 4.0 * batch * heads * seq_q * seq_k * dim
    if causal and seq_q == seq_k:
        flops *= 0.5
    return flops

"""Flash-Decoding: split-K attention for q_len = 1.

Prefill FlashAttention launches one program per query tile. With one query
token that is only ``batch * heads`` programs, so most SMs sit idle while
a long KV cache is streamed.

Flash-Decoding cuts the KV sequence into splits. Each program owns one
``(batch, head, split)`` and writes a partial output plus the online-softmax
statistics ``(m, l)``. A second kernel merges them:

    m* = max_s m_s
    O  = sum_s exp(m_s - m*) * acc_s / sum_s exp(m_s - m*) * l_s

``acc_s`` is the unnormalized tile accumulator (weights ``exp(score - m_s)``),
not the already-divided partial output. The single query attends to the whole
KV cache (it is the newest token). Use ``flash_attn`` when q_len > 1.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)
_BLOCK_N = 64


@triton.jit
def _decode_split(
    q_ptr,
    k_ptr,
    v_ptr,
    po_ptr,
    pm_ptr,
    pl_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_pls,
    seq_k,
    split_size,
    H,
    KV_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H
    off_kv = off_h // KV_GROUP
    kv_start = pid_s * split_size
    kv_end = tl.minimum(kv_start + split_size, seq_k)

    offs_d = tl.arange(0, HEAD_DIM)
    q_base = off_b.to(tl.int64) * stride_qb + off_h.to(tl.int64) * stride_qh
    k_base = off_b.to(tl.int64) * stride_kb + off_kv.to(tl.int64) * stride_kh
    v_base = off_b.to(tl.int64) * stride_vb + off_kv.to(tl.int64) * stride_vh
    q = tl.load(q_ptr + q_base + offs_d * stride_qd, mask=offs_d < HEAD_DIM, other=0.0)

    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for start_n in range(kv_start, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < kv_end
        k = tl.load(
            k_ptr + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask[:, None],
            other=0.0,
        )
        score = tl.sum(k.to(tl.float32) * q.to(tl.float32)[None, :], axis=1) * sm_scale
        score = tl.where(mask, score, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(score, axis=0))
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p = tl.exp(score - m_ij_safe)
        p = tl.where(mask & (m_ij != float("-inf")), p, 0.0)
        alpha = tl.exp(m_i - m_ij_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_ij

    po_base = off_b.to(tl.int64) * stride_pob + off_h.to(tl.int64) * stride_poh + pid_s.to(tl.int64) * stride_pos
    stat_base = off_b.to(tl.int64) * stride_pmb + off_h.to(tl.int64) * stride_pmh + pid_s.to(tl.int64) * stride_pms
    lstat_base = off_b.to(tl.int64) * stride_plb + off_h.to(tl.int64) * stride_plh + pid_s.to(tl.int64) * stride_pls
    tl.store(po_ptr + po_base + offs_d * stride_pod, acc)
    tl.store(pm_ptr + stat_base, m_i)
    tl.store(pl_ptr + lstat_base, l_i)


@triton.jit
def _decode_split_gqa(
    q_ptr,
    k_ptr,
    v_ptr,
    po_ptr,
    pm_ptr,
    pl_ptr,
    sm_scale,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_pls,
    seq_k,
    split_size,
    HKV,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KV_GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
):
    """One program owns one KV head and reuses each K/V tile across the query group."""
    pid_bkv = tl.program_id(0)
    pid_s = tl.program_id(1)
    off_b = pid_bkv // HKV
    off_kv = pid_bkv % HKV
    kv_start = pid_s * split_size
    kv_end = tl.minimum(kv_start + split_size, seq_k)

    offs_g = tl.arange(0, GROUP_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    q_heads = off_kv * KV_GROUP + offs_g
    q = tl.load(
        q_ptr
        + off_b.to(tl.int64) * stride_qb
        + q_heads[:, None].to(tl.int64) * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=(offs_g[:, None] < KV_GROUP) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    k_base = off_b.to(tl.int64) * stride_kb + off_kv.to(tl.int64) * stride_kh
    v_base = off_b.to(tl.int64) * stride_vb + off_kv.to(tl.int64) * stride_vh

    m_i = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GROUP_PAD], dtype=tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM], dtype=tl.float32)

    for start_n in range(kv_start, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask = offs_n < kv_end
        k = tl.load(
            k_ptr + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
            mask=mask[:, None],
            other=0.0,
        )
        v = tl.load(
            v_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=mask[:, None],
            other=0.0,
        )
        k_f = k.to(tl.float32)
        v_f = v.to(tl.float32)
        for h in range(KV_GROUP):
            qh = tl.sum(tl.where(offs_g[:, None] == h, q.to(tl.float32), 0.0), axis=0)
            score = tl.sum(k_f * qh[None, :], axis=1) * sm_scale
            score = tl.where(mask, score, float("-inf"))
            m_h = tl.sum(tl.where(offs_g == h, m_i, 0.0), axis=0)
            l_h = tl.sum(tl.where(offs_g == h, l_i, 0.0), axis=0)
            acc_h = tl.sum(tl.where(offs_g[:, None] == h, acc, 0.0), axis=0)
            m_ij = tl.maximum(m_h, tl.max(score, axis=0))
            m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
            p = tl.exp(score - m_ij_safe)
            p = tl.where(mask & (m_ij != float("-inf")), p, 0.0)
            alpha = tl.exp(m_h - m_ij_safe)
            alpha = tl.where(m_h == float("-inf"), 0.0, alpha)
            l_h = l_h * alpha + tl.sum(p, axis=0)
            acc_h = acc_h * alpha + tl.sum(p[:, None] * v_f, axis=0)
            m_i = tl.where(offs_g == h, m_ij, m_i)
            l_i = tl.where(offs_g == h, l_h, l_i)
            acc = tl.where(offs_g[:, None] == h, acc_h[None, :], acc)

    for h in range(KV_GROUP):
        q_head = off_kv * KV_GROUP + h
        acc_h = tl.sum(tl.where(offs_g[:, None] == h, acc, 0.0), axis=0)
        m_h = tl.sum(tl.where(offs_g == h, m_i, 0.0), axis=0)
        l_h = tl.sum(tl.where(offs_g == h, l_i, 0.0), axis=0)
        po_base = (
            off_b.to(tl.int64) * stride_pob
            + q_head.to(tl.int64) * stride_poh
            + pid_s.to(tl.int64) * stride_pos
        )
        tl.store(po_ptr + po_base + offs_d * stride_pod, acc_h)
        tl.store(
            pm_ptr + off_b.to(tl.int64) * stride_pmb + q_head.to(tl.int64) * stride_pmh + pid_s.to(tl.int64) * stride_pms,
            m_h,
        )
        tl.store(
            pl_ptr + off_b.to(tl.int64) * stride_plb + q_head.to(tl.int64) * stride_plh + pid_s.to(tl.int64) * stride_pls,
            l_h,
        )


@triton.jit
def _decode_combine(
    po_ptr,
    pm_ptr,
    pl_ptr,
    o_ptr,
    stride_pob,
    stride_poh,
    stride_pos,
    stride_pod,
    stride_pmb,
    stride_pmh,
    stride_pms,
    stride_plb,
    stride_plh,
    stride_pls,
    stride_ob,
    stride_oh,
    stride_od,
    num_splits,
    H,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    off_b = pid_bh // H
    off_h = pid_bh % H
    offs_d = tl.arange(0, HEAD_DIM)

    m_global = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(num_splits):
        stat = off_b.to(tl.int64) * stride_pmb + off_h.to(tl.int64) * stride_pmh + s.to(tl.int64) * stride_pms
        m_global = tl.maximum(m_global, tl.load(pm_ptr + stat))
    m_safe = tl.where(m_global == float("-inf"), 0.0, m_global)

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_global = tl.zeros((), dtype=tl.float32)
    for s in range(num_splits):
        s64 = s.to(tl.int64)
        ms = tl.load(pm_ptr + off_b.to(tl.int64) * stride_pmb + off_h.to(tl.int64) * stride_pmh + s64 * stride_pms)
        ls = tl.load(pl_ptr + off_b.to(tl.int64) * stride_plb + off_h.to(tl.int64) * stride_plh + s64 * stride_pls)
        w = tl.exp(ms - m_safe)
        w = tl.where(m_global == float("-inf"), 0.0, w)
        partial = tl.load(
            po_ptr
            + off_b.to(tl.int64) * stride_pob
            + off_h.to(tl.int64) * stride_poh
            + s64 * stride_pos
            + offs_d * stride_pod
        )
        # partial is the unnormalized accumulator, already scaled by exp(score - m_s).
        acc += w * partial
        l_global += w * ls

    out = tl.where(l_global > 0, acc / l_global, 0.0)
    o_base = off_b.to(tl.int64) * stride_ob + off_h.to(tl.int64) * stride_oh
    tl.store(o_ptr + o_base + offs_d * stride_od, out)


def choose_num_splits(batch: int, heads: int, seq_k: int, sm_count: int) -> int:
    """Enough splits to fill the GPU, without making each split tiny."""
    cap = max(1, seq_k // 128)
    need = max(1, (sm_count * 2) // max(1, batch * heads))
    return max(1, min(cap, need, 32))


def _pack_splits(seq_k: int, num_splits: int) -> tuple[int, int]:
    split_size = triton.cdiv(seq_k, num_splits)
    split_size = triton.cdiv(split_size, _BLOCK_N) * _BLOCK_N
    return split_size, triton.cdiv(seq_k, split_size)


def flash_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sm_scale: float | None = None,
    num_splits: int | None = None,
) -> torch.Tensor:
    """Decode attention. ``q`` is ``(batch, heads, 1, dim)``; ``k, v`` are the cache."""
    if not q.is_cuda:
        raise ValueError("flash_decode expects CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 (batch, heads, seq, dim)")
    if q.shape[-2] != 1:
        raise ValueError("flash_decode expects q_len == 1; use flash_attn for prefill")
    if q.shape[0] != k.shape[0]:
        raise ValueError("batch mismatch")
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if k.shape[0] != v.shape[0] or k.shape[-2:] != v.shape[-2:]:
        raise ValueError("k/v shape mismatch")
    if q_heads % kv_heads != 0:
        raise ValueError(f"query heads {q_heads} must be a multiple of kv heads {kv_heads}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v dtypes must match")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("flash_decode supports float16 and bfloat16")
    head_dim = q.shape[-1]
    if head_dim not in _SUPPORTED_HEAD_DIMS or k.shape[-1] != head_dim:
        raise ValueError(f"head dim must be one of {_SUPPORTED_HEAD_DIMS}")

    if sm_scale is None:
        sm_scale = head_dim ** -0.5

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch = q.shape[0]
    seq_k = k.shape[-2]
    kv_group = q_heads // kv_heads
    # Long caches are bandwidth-bound: one program per KV head reuses the tile
    # across the query group. Short caches keep one program per query head so
    # the GPU still has enough blocks.
    fused = kv_group > 1 and seq_k >= 2048
    parallel_heads = kv_heads if fused else q_heads
    if num_splits is None:
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        num_splits = choose_num_splits(batch, parallel_heads, seq_k, sm_count)
    num_splits = max(1, min(int(num_splits), seq_k))
    split_size, num_splits = _pack_splits(seq_k, num_splits)

    partial_o = torch.empty(batch, q_heads, num_splits, head_dim, device=q.device, dtype=torch.float32)
    partial_m = torch.empty(batch, q_heads, num_splits, device=q.device, dtype=torch.float32)
    partial_l = torch.empty(batch, q_heads, num_splits, device=q.device, dtype=torch.float32)
    out = torch.empty(batch, q_heads, 1, head_dim, device=q.device, dtype=torch.float32)

    if fused:
        _decode_split_gqa[(batch * kv_heads, num_splits)](
            q, k, v, partial_o, partial_m, partial_l, sm_scale,
            q.stride(0), q.stride(1), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            seq_k, split_size, kv_heads,
            BLOCK_N=_BLOCK_N, HEAD_DIM=head_dim, KV_GROUP=kv_group,
            GROUP_PAD=triton.next_power_of_2(kv_group), num_warps=4,
        )
    else:
        _decode_split[(batch * q_heads, num_splits)](
            q, k, v, partial_o, partial_m, partial_l, sm_scale,
            q.stride(0), q.stride(1), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
            partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
            partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
            seq_k, split_size, q_heads,
            KV_GROUP=kv_group, BLOCK_N=_BLOCK_N, HEAD_DIM=head_dim, num_warps=4,
        )
    _decode_combine[(batch * q_heads,)](
        partial_o, partial_m, partial_l, out,
        partial_o.stride(0), partial_o.stride(1), partial_o.stride(2), partial_o.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        partial_l.stride(0), partial_l.stride(1), partial_l.stride(2),
        out.stride(0), out.stride(1), out.stride(3),
        num_splits, q_heads, HEAD_DIM=head_dim, num_warps=4,
    )
    return out.to(dtype=q.dtype)

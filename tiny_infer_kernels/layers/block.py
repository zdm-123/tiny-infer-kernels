"""One decoder block wired from the Triton kernels.

This is not a downloaded checkpoint. Weights are random fp16 tensors with
the same shapes as a small Llama/Qwen block, so the demo runs offline on a
12 GB card and still exercises the inference path:

    RMSNorm -> QKV -> RoPE -> attention -> out-proj
            -> RMSNorm -> SwiGLU -> down-proj

``attn="prefill"`` uses FlashAttention-2. ``attn="decode"`` uses Flash-Decoding
against a KV cache. ``quantized=True`` replaces the four linears with W8A16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tiny_infer_kernels.kernels.flash_attn import flash_attn, flash_attn_ref
from tiny_infer_kernels.kernels.flash_decode import flash_decode
from tiny_infer_kernels.kernels.rmsnorm import rmsnorm, rmsnorm_ref
from tiny_infer_kernels.kernels.rope import rope, rope_ref
from tiny_infer_kernels.kernels.swiglu import swiglu, swiglu_ref
from tiny_infer_kernels.kernels.w8a16_gemm import quantize_weight_w8a16, w8a16_gemm


class DecoderWeights:
    def __init__(self, hidden: int, heads: int, intermediate: int, dtype: torch.dtype, device: str):
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.intermediate = intermediate
        self.dtype = dtype
        g = torch.Generator(device="cpu")
        g.manual_seed(0)

        def param(*shape):
            return (torch.randn(*shape, generator=g) * 0.02).to(device=device, dtype=dtype)

        self.norm1 = param(hidden)
        self.norm2 = param(hidden)
        self.w_qkv = param(3 * hidden, hidden)
        self.w_o = param(hidden, hidden)
        self.w_gate = param(intermediate, hidden)
        self.w_up = param(intermediate, hidden)
        self.w_down = param(hidden, intermediate)
        self.qkv_i8, self.qkv_scale = quantize_weight_w8a16(self.w_qkv)
        self.o_i8, self.o_scale = quantize_weight_w8a16(self.w_o)
        self.gate_i8, self.gate_scale = quantize_weight_w8a16(self.w_gate)
        self.up_i8, self.up_scale = quantize_weight_w8a16(self.w_up)
        self.down_i8, self.down_scale = quantize_weight_w8a16(self.w_down)

    def linears_fp16(self):
        return self.w_qkv, self.w_o, self.w_gate, self.w_up, self.w_down


def _split_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
    b, s, _ = t.shape
    return t.view(b, s, heads, -1).transpose(1, 2).contiguous()


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    b, h, s, d = t.shape
    return t.transpose(1, 2).contiguous().view(b, s, h * d)


def _linear(x: torch.Tensor, weight: torch.Tensor, quantized: bool, packed) -> torch.Tensor:
    flat = x.reshape(-1, x.shape[-1])
    if quantized:
        w_i8, scale = packed
        y = w8a16_gemm(flat, w_i8, scale)
    else:
        y = F.linear(flat, weight)
    return y.view(*x.shape[:-1], weight.shape[0])


def decoder_block(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weights: DecoderWeights,
    *,
    k_cache: torch.Tensor | None = None,
    v_cache: torch.Tensor | None = None,
    triton_kernels: bool = True,
    quantized: bool = False,
) -> torch.Tensor:
    """Run one block. ``x`` is (batch, seq, hidden). Decode passes a KV cache and seq=1."""
    eps = 1e-6
    norm = rmsnorm if triton_kernels else rmsnorm_ref
    apply_rope = rope if triton_kernels else rope_ref
    act = swiglu if triton_kernels else swiglu_ref

    h = norm(x, weights.norm1, eps)
    qkv = _linear(h, weights.w_qkv, quantized, (weights.qkv_i8, weights.qkv_scale))
    q, k, v = qkv.split(weights.hidden, dim=-1)
    q = _split_heads(q, weights.heads)
    k = _split_heads(k, weights.heads)
    v = _split_heads(v, weights.heads)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    if k_cache is not None:
        k = torch.cat([k_cache, k], dim=2)
        v = torch.cat([v_cache, v], dim=2)
    decode = q.shape[-2] == 1 and k.shape[-2] > 1
    if triton_kernels and decode:
        attn = flash_decode(q, k, v)
    elif triton_kernels:
        attn = flash_attn(q, k, v, causal=True)
    else:
        attn = flash_attn_ref(q, k, v, causal=not decode)
    y = _merge_heads(attn)
    y = _linear(y, weights.w_o, quantized, (weights.o_i8, weights.o_scale))
    x = x + y

    h = norm(x, weights.norm2, eps)
    gate = _linear(h, weights.w_gate, quantized, (weights.gate_i8, weights.gate_scale))
    up = _linear(h, weights.w_up, quantized, (weights.up_i8, weights.up_scale))
    y = act(gate, up)
    y = _linear(y, weights.w_down, quantized, (weights.down_i8, weights.down_scale))
    return x + y

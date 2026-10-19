"""Qwen2-shaped causal LM.

The block is the Qwen2.5-0.5B pattern: RMSNorm, grouped-query attention,
``rotate_half`` RoPE, SwiGLU, tied embeddings. Projections stay in PyTorch.
Triton covers norm, RoPE, attention, and SwiGLU.

KV heads are read directly by the Triton kernels. A query head uses KV head
``q_head // (Hq / Hkv)``. On long decode caches one program loads each K/V
tile once and reuses it across the query group, instead of materializing a
repeated cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tiny_infer_kernels.kernels.flash_attn import flash_attn
from tiny_infer_kernels.kernels.flash_decode import flash_decode
from tiny_infer_kernels.kernels.rmsnorm import rmsnorm, rmsnorm_ref
from tiny_infer_kernels.kernels.rope import build_rope_cache, rope_rotate_half, rope_rotate_half_ref
from tiny_infer_kernels.kernels.swiglu import swiglu, swiglu_ref


@dataclass
class Qwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 2048

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must divide num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("query heads must be a multiple of kv heads")
        return self.num_attention_heads // self.num_key_value_heads


QWEN2_5_0_5B = Qwen2Config()


def _empty(shape, dtype, device):
    return (torch.randn(*shape, device=device, dtype=torch.float32) * 0.02).to(dtype)


class Qwen2Layer:
    def __init__(self, config: Qwen2Config, dtype: torch.dtype, device: str):
        h = config.hidden_size
        d = config.head_dim
        self.input_norm = _empty((h,), dtype, device)
        self.post_norm = _empty((h,), dtype, device)
        self.wq = _empty((config.num_attention_heads * d, h), dtype, device)
        self.wk = _empty((config.num_key_value_heads * d, h), dtype, device)
        self.wv = _empty((config.num_key_value_heads * d, h), dtype, device)
        self.wo = _empty((h, config.num_attention_heads * d), dtype, device)
        self.bq = torch.zeros(config.num_attention_heads * d, device=device, dtype=dtype)
        self.bk = torch.zeros(config.num_key_value_heads * d, device=device, dtype=dtype)
        self.bv = torch.zeros(config.num_key_value_heads * d, device=device, dtype=dtype)
        self.w_gate = _empty((config.intermediate_size, h), dtype, device)
        self.w_up = _empty((config.intermediate_size, h), dtype, device)
        self.w_down = _empty((h, config.intermediate_size), dtype, device)


class Qwen2Model:
    def __init__(self, config: Qwen2Config, dtype: torch.dtype = torch.float16, device: str = "cuda"):
        self.config = config
        self.dtype = dtype
        self.device = device
        self.embed = _empty((config.vocab_size, config.hidden_size), dtype, device)
        self.layers = [Qwen2Layer(config, dtype, device) for _ in range(config.num_hidden_layers)]
        self.final_norm = _empty((config.hidden_size,), dtype, device)
        cos, sin = build_rope_cache(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            device=device,
            dtype=dtype,
        )
        self.cos = cos
        self.sin = sin

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        backend: str = "eager",
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if backend not in ("eager", "triton"):
            raise ValueError("backend must be 'eager' or 'triton'")
        cfg = self.config
        batch, seq = input_ids.shape
        past = 0 if cache is None else cache[0][0].shape[-2]
        if past + seq > cfg.max_position_embeddings:
            raise ValueError(
                f"position {past + seq} exceeds max_position_embeddings={cfg.max_position_embeddings}"
            )
        x = self.embed[input_ids]
        cos = self.cos[past : past + seq]
        sin = self.sin[past : past + seq]
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            prev = None if cache is None else cache[i]
            x, layer_cache = _layer_forward(x, cos, sin, layer, cfg, prev, backend)
            new_cache.append(layer_cache)
        norm = rmsnorm if backend == "triton" else rmsnorm_ref
        x = norm(x, self.final_norm, cfg.rms_norm_eps)
        logits = F.linear(x, self.embed)
        return logits, new_cache


def repeat_kv(x: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return x
    batch, heads, seq, dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, groups, seq, dim)
        .reshape(batch, heads * groups, seq, dim)
    )


def _split_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
    batch, seq, _ = t.shape
    return t.view(batch, seq, heads, -1).transpose(1, 2).contiguous()


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = t.shape
    return t.transpose(1, 2).contiguous().view(batch, seq, heads * dim)


def _eager_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    try:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
    except TypeError:
        groups = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(
            q, repeat_kv(k, groups), repeat_kv(v, groups), is_causal=causal
        )


def _layer_forward(x, cos, sin, layer: Qwen2Layer, cfg: Qwen2Config, cache, backend: str):
    norm = rmsnorm if backend == "triton" else rmsnorm_ref
    apply_rope = rope_rotate_half if backend == "triton" else rope_rotate_half_ref
    act = swiglu if backend == "triton" else swiglu_ref

    h = norm(x, layer.input_norm, cfg.rms_norm_eps)
    q = F.linear(h, layer.wq, layer.bq)
    k = F.linear(h, layer.wk, layer.bk)
    v = F.linear(h, layer.wv, layer.bv)
    q = apply_rope(_split_heads(q, cfg.num_attention_heads), cos, sin)
    k = apply_rope(_split_heads(k, cfg.num_key_value_heads), cos, sin)
    v = _split_heads(v, cfg.num_key_value_heads)
    if cache is not None:
        k = torch.cat((cache[0], k), dim=2)
        v = torch.cat((cache[1], v), dim=2)
    new_cache = (k, v)
    decode = q.shape[-2] == 1 and k.shape[-2] > 1
    if backend == "triton":
        if decode:
            attn = flash_decode(q, k, v)
        else:
            attn = flash_attn(q, k, v, causal=True)
    else:
        attn = _eager_attn(q, k, v, causal=not decode)
    y = F.linear(_merge_heads(attn), layer.wo)
    x = x + y
    h = norm(x, layer.post_norm, cfg.rms_norm_eps)
    y = act(F.linear(h, layer.w_gate), F.linear(h, layer.w_up))
    x = x + F.linear(y, layer.w_down)
    return x, new_cache


@torch.no_grad()
def generate(
    model: Qwen2Model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    backend: str = "eager",
) -> torch.Tensor:
    """Greedy decode. Returns only the new token ids, shape ``(batch, max_new_tokens)``."""
    logits, cache = model.forward(input_ids, cache=None, backend=backend)
    next_id = logits[:, -1].argmax(dim=-1)
    tokens = [next_id]
    for _ in range(max_new_tokens - 1):
        logits, cache = model.forward(next_id[:, None], cache=cache, backend=backend)
        next_id = logits[:, -1].argmax(dim=-1)
        tokens.append(next_id)
    return torch.stack(tokens, dim=1)

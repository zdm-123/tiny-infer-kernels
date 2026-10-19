import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

from tiny_infer_kernels.models.qwen2 import Qwen2Config, Qwen2Model
from tiny_infer_kernels.utils.check import assert_close


def _tiny():
    cfg = Qwen2Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2Model(cfg, dtype=torch.float16, device="cuda")


def test_prefill_logits_match_eager():
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(0, 128, (1, 16), device="cuda")
    eager, _ = model.forward(ids, backend="eager")
    triton, _ = model.forward(ids, backend="triton")
    assert_close(triton, eager, atol=3e-2, rtol=3e-2, name="qwen_prefill")


def test_decode_step_matches_eager():
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(0, 128, (1, 8), device="cuda")
    eager_logits, eager_cache = model.forward(ids, backend="eager")
    triton_logits, triton_cache = model.forward(ids, backend="triton")
    nxt = eager_logits[:, -1].argmax(dim=-1, keepdim=True)
    eager_step, _ = model.forward(nxt, cache=eager_cache, backend="eager")
    triton_step, _ = model.forward(nxt, cache=triton_cache, backend="triton")
    assert_close(triton_step, eager_step, atol=3e-2, rtol=3e-2, name="qwen_decode")

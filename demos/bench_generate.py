"""Prefill and decode tokens/s for a Qwen2.5-0.5B-shaped model.

Weights are random. The shapes match Qwen2.5-0.5B (hidden 896, 14 query
heads, 2 kv heads, intermediate 4864). ``--layers`` defaults to 4 so a
laptop run finishes quickly; 24 is the real depth.

This does not download a checkpoint. A matching Hugging Face state dict can
be wired in later; the module layout follows those parameter names.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tiny_infer_kernels.models.qwen2 import Qwen2Config, Qwen2Model
from tiny_infer_kernels.utils.bench import bench_ms
from tiny_infer_kernels.utils.check import max_abs_err
from tiny_infer_kernels.utils.device import gpu_summary


def _model(layers: int) -> Qwen2Model:
    cfg = Qwen2Config(num_hidden_layers=layers)
    return Qwen2Model(cfg, dtype=torch.float16, device="cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    info = gpu_summary()
    print("GPU:", info)
    if not info.get("cuda"):
        raise SystemExit("CUDA GPU required")

    print(f"building Qwen2.5-0.5B shape, layers={args.layers}")
    model = _model(args.layers)
    ids = torch.randint(0, model.config.vocab_size, (1, args.prompt), device="cuda")

    eager_logits, eager_cache = model.forward(ids, backend="eager")
    triton_logits, triton_cache = model.forward(ids, backend="triton")
    print(f"prefill logit max_abs_err: {max_abs_err(triton_logits, eager_logits):.4e}")

    prefill_eager = bench_ms(lambda: model.forward(ids, backend="eager"), warmup=2, rep=5)
    prefill_triton = bench_ms(lambda: model.forward(ids, backend="triton"), warmup=2, rep=5)
    print(f"prefill eager  {prefill_eager:8.2f} ms")
    print(f"prefill triton {prefill_triton:8.2f} ms")

    token = eager_logits[:, -1].argmax(dim=-1, keepdim=True)

    def decode_eager():
        cache = eager_cache
        nxt = token
        for _ in range(args.new_tokens):
            logits, cache = model.forward(nxt, cache=cache, backend="eager")
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)

    def decode_triton():
        cache = triton_cache
        nxt = token
        for _ in range(args.new_tokens):
            logits, cache = model.forward(nxt, cache=cache, backend="triton")
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)

    # One step of agreement, then time a fixed-length decode.
    step_e, _ = model.forward(token, cache=eager_cache, backend="eager")
    step_t, _ = model.forward(token, cache=triton_cache, backend="triton")
    print(f"decode-step logit max_abs_err: {max_abs_err(step_t, step_e):.4e}")

    ms_e = bench_ms(decode_eager, warmup=1, rep=3)
    ms_t = bench_ms(decode_triton, warmup=1, rep=3)
    print(f"decode {args.new_tokens} tokens eager  {ms_e:8.2f} ms  {args.new_tokens / (ms_e / 1e3):6.1f} tok/s")
    print(f"decode {args.new_tokens} tokens triton {ms_t:8.2f} ms  {args.new_tokens / (ms_t / 1e3):6.1f} tok/s")


if __name__ == "__main__":
    main()

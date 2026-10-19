"""Numerical correctness against a PyTorch reference."""

from __future__ import annotations

import torch


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    name: str = "tensor",
) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{name}: shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}"
        )
    if actual.dtype != expected.dtype:
        actual = actual.float()
        expected = expected.float()
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max().item()
    denom = expected.float().abs().clamp_min(1e-8)
    max_rel = (diff / denom).max().item()
    ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(
            f"{name}: mismatch max_abs={max_abs:.4e} max_rel={max_rel:.4e} "
            f"(atol={atol}, rtol={rtol})"
        )


def max_abs_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()

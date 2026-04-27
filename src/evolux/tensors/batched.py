"""Batched-tensor primitives shared across modules."""

from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.nn import functional as F

# Set EVOLUX_FAST=1 to skip finite assertions in hot paths.
_FAST = os.environ.get("EVOLUX_FAST", "0") == "1"


def assert_finite(t: Tensor, name: str = "tensor") -> None:
    """Raise if *t* contains NaN or Inf. No-op when EVOLUX_FAST=1."""
    if _FAST:
        return
    if not torch.isfinite(t).all():
        raise ValueError(f"{name}: contains NaN or Inf (shape={tuple(t.shape)})")


def flatten_batch(t: Tensor, n_batch_dims: int = 2) -> tuple[Tensor, tuple[int, ...]]:
    """Collapse leading batch dims into one. Returns (flat, original_shape)."""
    orig = tuple(t.shape[:n_batch_dims])
    return t.reshape(-1, *t.shape[n_batch_dims:]), orig


def unflatten_batch(t: Tensor, orig: tuple[int, ...]) -> Tensor:
    """Inverse of flatten_batch."""
    return t.reshape(*orig, *t.shape[1:])


def one_hot(idx: Tensor, n: int, dtype: torch.dtype = torch.float32) -> Tensor:
    """Differentiable-friendly one-hot. ``idx`` is integer with arbitrary shape."""
    return F.one_hot(idx.long(), num_classes=n).to(dtype=dtype)


def soft_attention(
    query: Tensor, keys: Tensor, values: Tensor, *, temperature: float = 1.0
) -> Tensor:
    """Cosine-similarity soft attention.

    Shapes
    ------
    query  : (B, D)
    keys   : (B, M, D)
    values : (B, M, V)
    return : (B, V)
    """
    q = F.normalize(query, dim=-1).unsqueeze(1)  # (B, 1, D)
    k = F.normalize(keys, dim=-1)  # (B, M, D)
    sims = (q * k).sum(-1) / max(temperature, 1e-6)  # (B, M)
    weights = F.softmax(sims, dim=-1).unsqueeze(-1)  # (B, M, 1)
    return (weights * values).sum(dim=1)  # (B, V)

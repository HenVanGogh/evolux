"""Batched-tensor primitives shared across modules."""

from __future__ import annotations

import os
from collections.abc import Callable

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


def masked_mean(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Mean of *x* along *dim*, ignoring positions where *mask* is False/zero.

    Parameters
    ----------
    x    : arbitrary shape
    mask : same shape as *x* (bool or float)
    dim  : dimension to reduce
    """
    m = mask.to(dtype=x.dtype)
    m_sum = m.sum(dim=dim).clamp(min=1.0)
    return (x * m).sum(dim=dim) / m_sum


def masked_softmax(x: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Softmax of *x* along *dim* with masked positions set to ``-inf``.

    Parameters
    ----------
    x    : arbitrary shape
    mask : same shape as *x* (bool) — True where the position is *valid*
    dim  : dimension along which softmax is computed
    """
    x_masked = x.masked_fill(~mask.bool(), float("-inf"))
    return F.softmax(x_masked, dim=dim)


def where_done(
    state: dict[str, Tensor],
    done_mask: Tensor,
    init_fn: Callable[[], dict[str, Tensor]],
) -> dict[str, Tensor]:
    """Replace batch rows where *done_mask* is True with fresh state from *init_fn*.

    Parameters
    ----------
    state     : dict of tensors, each with leading batch dimension B
    done_mask : bool tensor of shape (B,) — True where episode is finished
    init_fn   : zero-argument callable returning a dict with the same keys /
                shapes as *state* (e.g. ``lambda: module.init_state(B, device)``)

    Returns a new dict; the original tensors are never mutated.
    """
    if not done_mask.any():
        return state
    fresh = init_fn()
    out: dict[str, Tensor] = {}
    for key, val in state.items():
        d = done_mask
        # Broadcast done_mask from (B,) to the same number of dims as val.
        while d.dim() < val.dim():
            d = d.unsqueeze(-1)
        out[key] = torch.where(d.expand_as(val), fresh[key], val)
    return out


def gather_indices(x: Tensor, idx: Tensor) -> Tensor:
    """Vectorised batched gather along the T dimension.

    Parameters
    ----------
    x   : (B, T, D)
    idx : (B,) — integer index into T for each batch element

    Returns (B, D).
    """
    # (B,) → (B, 1, D) for gather, then squeeze back.
    idx_exp = idx.long().view(-1, 1, 1).expand(-1, 1, x.size(-1))  # (B, 1, D)
    return x.gather(1, idx_exp).squeeze(1)  # (B, D)

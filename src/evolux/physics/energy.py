"""Energy / metabolism cost computation for discrete physics.

Provides :func:`compute_action_cost` — the canonical action-cost primitive
shared by discrete and continuous physics back-ends.
"""

from __future__ import annotations

from torch import Tensor


def compute_action_cost(actions: Tensor) -> Tensor:
    """Sum-of-squares action cost for a batch of action vectors.

    Parameters
    ----------
    actions:
        ``(B, A)`` float tensor of action values in any range.

    Returns
    -------
    Tensor
        ``(B,)`` non-negative energy cost per batch element.
    """
    if actions.dim() == 1:
        actions = actions.unsqueeze(-1)
    return (actions**2).sum(dim=-1)  # (B,)

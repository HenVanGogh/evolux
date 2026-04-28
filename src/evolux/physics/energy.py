"""Energy bookkeeping — metabolism and action-cost computation.

Phase 1 implements a simple sum-of-squares action cost.  More sophisticated
metabolic models (fatigue, mass-dependent costs) are deferred to Phase 2.
"""

from __future__ import annotations

import torch
from torch import Tensor


def compute_action_cost(actions: Tensor) -> Tensor:
    """Return per-creature action cost as sum of squared action components.

    Parameters
    ----------
    actions:
        ``(B, A)`` float tensor of action values.

    Returns
    -------
    Tensor
        ``(B,)`` float32 tensor of non-negative costs, one per creature.
    """
    return torch.sum(actions**2, dim=-1)

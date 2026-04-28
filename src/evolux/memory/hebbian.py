"""Hebbian memory -- slow-weight associative matrix.

State layout
------------
W : (B, D, D)  -- per-batch Hebbian weight matrix

Update rule (Miconi et al., 2018 -- Differentiable Plasticity)::

    W <- (1 - decay) * W  +  lr * (value outer key)

where ``decay = sigmoid(log_decay)`` and ``lr = softplus(log_lr)`` are
learnable scalars.  Weights are clamped to ``[-w_max, w_max]`` to prevent
runaway growth.

Read::

    output = W @ query                 # (B, D)

References
----------
Miconi et al., *Differentiable plasticity: training plastic neural networks
with backpropagation* (ICML 2018). https://arxiv.org/abs/1804.02464
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evolux.core.types import MemoryState
from evolux.memory import MEMORY_REGISTRY


@MEMORY_REGISTRY.register("hebbian")
class HebbianMemory:
    """Slow-weight associative matrix implementing the Memory Protocol.

    Maintains a per-batch Hebbian weight matrix ``W: (B, D, D)`` that
    accumulates key-value associations via the generalised Hebb rule.

    Parameters
    ----------
    dim:
        Dimensionality ``D`` of both keys and values.
    w_max:
        Clamp limit applied after every write to bound weight magnitude.

    Notes
    -----
    ``key_dim == val_dim == dim`` for Hebbian memory since the square weight
    matrix ``W`` maps ``D``-dimensional queries to ``D``-dimensional outputs.
    """

    def __init__(self, dim: int, *, w_max: float = 5.0) -> None:
        self.capacity: int = dim  # nominal; Hebbian has no discrete slot limit
        self.key_dim: int = dim
        self.val_dim: int = dim
        self.w_max: float = w_max

        # Learnable scalars stored in unconstrained space.
        # lr    = softplus(log_lr)   > 0
        # decay = sigmoid(log_decay) ∈ (0, 1)
        # Initialise so that lr ≈ 0.01 and decay ≈ 0.1 at the start of training.
        # softplus_inv(0.01) ≈ log(exp(0.01) - 1) ≈ -4.60
        # logit(0.1) = log(0.1 / 0.9) ≈ -2.20
        self.log_lr: nn.Parameter = nn.Parameter(torch.tensor(-4.60))
        self.log_decay: nn.Parameter = nn.Parameter(torch.tensor(-2.20))

    # ── Derived scalars ──────────────────────────────────────────────────────

    @property
    def lr(self) -> Tensor:
        """Effective learning rate (always > 0)."""
        return F.softplus(self.log_lr)

    @property
    def decay(self) -> Tensor:
        """Effective decay rate in (0, 1)."""
        return torch.sigmoid(self.log_decay)

    # ── Protocol methods ─────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Return a zero-initialised state for *batch_size* environments.

        Returns
        -------
        dict with:
            ``W`` : ``(B, D, D)`` float32 zeros
        """
        return {
            "W": torch.zeros(batch_size, self.key_dim, self.val_dim, device=device),
        }

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState:
        """Pure write: accumulate the outer product of ``(value, key)`` into ``W``.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        key:
            ``(B, D)`` key tensor.
        value:
            ``(B, D)`` value tensor.

        Returns
        -------
        New state with updated weight matrix::

            W <- (1 - decay) * W  +  lr * (value outer key)
            W <- clamp(W, -w_max, w_max)
        """
        W = state["W"]  # (B, D, D)

        # Outer product: value (B, D, 1) x key (B, 1, D) -> (B, D, D)
        # W[b, i, j] += lr * value[b, i] * key[b, j]
        # so that W @ key ≈ value * ||key||^2  (recovers value direction).
        outer = torch.bmm(value.unsqueeze(2), key.unsqueeze(1))  # (B, D, D)

        new_W = (1.0 - self.decay) * W + self.lr * outer
        new_W = new_W.clamp(-self.w_max, self.w_max)

        return {"W": new_W}

    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor:
        """Matrix-vector read: return ``W @ query``.

        Parameters
        ----------
        state:
            Current memory state.
        query:
            ``(B, D)`` query tensor.
        top_k:
            Unused (present for Protocol compatibility).

        Returns
        -------
        ``(B, D)`` output tensor.
        """
        W = state["W"]  # (B, D, D)
        # (B, D, D) @ (B, D, 1) → (B, D, 1) → squeeze → (B, D)
        return torch.bmm(W, query.unsqueeze(2)).squeeze(2)

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState:
        """Zero out ``W`` for environments indicated by *mask*.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        mask:
            ``(B,)`` bool tensor.  If ``None``, all environments are reset.

        Returns
        -------
        New state with masked environments zeroed.

        Notes
        -----
        Hebbian weights conceptually represent lifetime structural learning and
        therefore *persist* across episodes by default.  This method exists to
        satisfy the Protocol and is useful for ablations or curriculum resets.
        """
        if mask is None:
            return self.init_state(state["W"].shape[0], state["W"].device)

        new_W = state["W"].clone()
        new_W[mask] = 0.0
        return {"W": new_W}

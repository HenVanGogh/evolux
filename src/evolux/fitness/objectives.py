"""Phase-1 single-objective fitness functions.

Each class implements the :class:`~evolux.core.protocols.Objective` Protocol
and is registered in :data:`~evolux.fitness.OBJECTIVE_REGISTRY`.

Objectives
----------
- ``survival_time``    — steps alive (sum of ``~done`` over the trajectory)
- ``food_collected``   — sum of positive reward signals
- ``distance_travelled`` — L2 path length from ``trajectory.aux["positions"]``
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from evolux.core.protocols import World
from evolux.core.types import Trajectory
from evolux.fitness import OBJECTIVE_REGISTRY

log = logging.getLogger(__name__)


@OBJECTIVE_REGISTRY.register("survival_time")
class SurvivalTime:
    """Count time steps the creature was alive (i.e. ``done`` was False).

    Notes
    -----
    ``score[b] = Σ_t (~dones[b, t])``
    """

    name: str = "survival_time"
    higher_is_better: bool = True

    def __init__(self, weight: float = 1.0) -> None:
        self.weight: float = weight

    def evaluate(self, trajectory: Trajectory, world: World) -> Tensor:
        """Return ``(B,)`` tensor: number of alive steps per creature."""
        # dones: (B, T) bool — alive when False
        return (~trajectory.dones).float().sum(dim=1)  # (B,)


@OBJECTIVE_REGISTRY.register("food_collected")
class FoodCollected:
    """Sum of positive reward signals (food eaten).

    Notes
    -----
    ``score[b] = Σ_t max(0, rewards[b, t])``
    """

    name: str = "food_collected"
    higher_is_better: bool = True

    def __init__(self, weight: float = 1.0) -> None:
        self.weight: float = weight

    def evaluate(self, trajectory: Trajectory, world: World) -> Tensor:
        """Return ``(B,)`` tensor: total food collected per creature."""
        return trajectory.rewards.clamp(min=0.0).sum(dim=1)  # (B,)


@OBJECTIVE_REGISTRY.register("distance_travelled")
class DistanceTravelled:
    """L2 path length computed from positions stored in ``trajectory.aux``.

    Notes
    -----
    Expects ``trajectory.aux[positions_key]`` of shape ``(B, T, 2)``.
    Falls back to zeros when the key is absent and logs a warning.
    """

    name: str = "distance_travelled"
    higher_is_better: bool = True

    def __init__(self, weight: float = 1.0, positions_key: str = "positions") -> None:
        self.weight: float = weight
        self._positions_key: str = positions_key

    def evaluate(self, trajectory: Trajectory, world: World) -> Tensor:
        """Return ``(B,)`` tensor: total path length per creature."""
        if self._positions_key not in trajectory.aux:
            log.warning(
                "DistanceTravelled: key '%s' not found in trajectory.aux; returning zeros.",
                self._positions_key,
            )
            return torch.zeros(
                trajectory.batch_size,
                dtype=torch.float32,
                device=trajectory.dones.device,
            )

        positions = trajectory.aux[self._positions_key]  # (B, T, 2)
        deltas = positions[:, 1:] - positions[:, :-1]  # (B, T-1, 2)
        return deltas.norm(dim=-1).sum(dim=-1)  # (B,)

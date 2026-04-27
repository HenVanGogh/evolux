"""Grid environment: extends BaseEnvironment with seasonal and patch dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sim_env.environment.base_env import BaseEnvironment
from sim_env.environment.resources import ResourceManager

if TYPE_CHECKING:
    from sim_env.core.world import World


class GridEnvironment(BaseEnvironment):
    """Wraps the World grid and drives resource / hazard dynamics each step.

    Extends basic food respawn with:
    - Patchy food distribution (clustered resources)
    - Optional seasonal cycles (food density varies sinusoidally)
    - Optional predator pressure (energy drain in danger zones)
    """

    def __init__(self, world: "World", cfg: dict) -> None:
        self._world = world
        self._cfg = cfg
        self._resource_mgr = ResourceManager(cfg)
        self._step_count: int = 0
        self._season: float = 0.0   # [0, 2π)

    def reset(self, rng: np.random.Generator) -> None:
        self._step_count = 0
        self._season = 0.0
        self._resource_mgr.seed_patches(self._world, rng)

    def step(self, rng: np.random.Generator) -> None:
        self._step_count += 1
        self._season = (self._step_count * 2 * np.pi / 500) % (2 * np.pi)
        season_factor = 0.5 + 0.5 * np.sin(self._season)  # [0, 1]

        self._resource_mgr.respawn(self._world, rng, season_factor=season_factor)

    def observe(self) -> dict:
        return {
            "step": self._step_count,
            "season": float(self._season),
            "total_food": self._world.total_food(),
        }

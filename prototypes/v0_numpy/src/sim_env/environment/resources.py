"""Resource manager: handles food patches, respawn logic, and clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.core.world import World


class ResourceManager:
    """Manages food placement with clustered patch seeding and respawn."""

    def __init__(self, cfg: dict) -> None:
        rc = cfg["environment"]["resources"]
        self._food_energy = rc.get("food_energy", 10.0)
        self._density = rc.get("food_density", 0.05)
        self._respawn_rate = rc.get("food_respawn_rate", 0.002)
        self._n_patches = max(1, int(cfg["world"]["width"] * self._density))

    def seed_patches(self, world: "World", rng: np.random.Generator) -> None:
        """Place food in Gaussian clusters across the world."""
        world.food_grid[:] = 0.0
        patch_sigma = world.width / max(self._n_patches, 1) * 0.5

        for _ in range(self._n_patches):
            cx = int(rng.integers(0, world.width))
            cy = int(rng.integers(0, world.height))
            # Drop a Gaussian blob of food around (cx, cy)
            xs = rng.normal(cx, patch_sigma, size=20).astype(int)
            ys = rng.normal(cy, patch_sigma, size=20).astype(int)
            for x, y in zip(xs, ys):
                wx, wy = world.clamp(x, y)
                world.food_grid[wy, wx] = min(
                    world.food_grid[wy, wx] + self._food_energy * 0.5,
                    self._food_energy,
                )

    def respawn(
        self,
        world: "World",
        rng: np.random.Generator,
        season_factor: float = 1.0,
    ) -> None:
        """Stochastically respawn food on empty cells, scaled by season."""
        rate = self._respawn_rate * season_factor
        empty = world.food_grid < 1e-3
        spawn = empty & (rng.random((world.height, world.width)) < rate)
        world.food_grid[spawn] = self._food_energy

"""World / environment container.

The World holds the spatial grid, resources, and all living creatures.
Creatures are managed by the Population; the World owns only the physical state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.creatures.creature import Creature


@dataclass
class WorldCell:
    food: float = 0.0
    hazard: bool = False
    # Creature occupying this cell (None if empty)
    occupant: "Creature | None" = field(default=None, repr=False)


class World:
    """Discrete 2-D grid world with toroidal or bounded topology.

    Layers
    ------
    food_grid  : (H, W) float array — food energy at each cell
    hazard_grid: (H, W) bool  array — hazard flags
    """

    def __init__(self, config: dict, rng: np.random.Generator) -> None:
        wc = config["world"]
        ec = config["environment"]

        self.width: int = wc["width"]
        self.height: int = wc["height"]
        self.wrap: bool = wc.get("wrap", True)

        self._food_cfg = ec["resources"]
        self._hazard_cfg = ec.get("hazards", {"enabled": False})

        self.food_grid: np.ndarray = np.zeros((self.height, self.width), dtype=np.float32)
        self.hazard_grid: np.ndarray = np.zeros((self.height, self.width), dtype=bool)

        # Occupancy map: creature references (None = empty)
        self.occupancy: list[list["Creature | None"]] = [
            [None] * self.width for _ in range(self.height)
        ]

        self._rng = rng
        self._initialise(rng)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def _initialise(self, rng: np.random.Generator) -> None:
        density = self._food_cfg.get("food_density", 0.05)
        mask = rng.random((self.height, self.width)) < density
        self.food_grid[mask] = self._food_cfg.get("food_energy", 10.0)
        if self._hazard_cfg.get("enabled"):
            h_density = self._hazard_cfg.get("hazard_density", 0.02)
            self.hazard_grid = rng.random((self.height, self.width)) < h_density

    def reset_generation(self) -> None:
        """Reset transient state (occupancy) but keep persistent geography."""
        for row in self.occupancy:
            row[:] = [None] * self.width

    def step(self, rng: np.random.Generator) -> None:
        """Advance world physics by one step (food respawn, etc.)."""
        respawn_rate = self._food_cfg.get("food_respawn_rate", 0.002)
        max_food = self._food_cfg.get("food_energy", 10.0)
        # Stochastic food respawn on empty cells
        empty_mask = self.food_grid < 1e-3
        spawn_mask = empty_mask & (rng.random((self.height, self.width)) < respawn_rate)
        self.food_grid[spawn_mask] = max_food

    # ── Spatial helpers ─────────────────────────────────────────────────────

    def clamp(self, x: int, y: int) -> tuple[int, int]:
        if self.wrap:
            return x % self.width, y % self.height
        return int(np.clip(x, 0, self.width - 1)), int(np.clip(y, 0, self.height - 1))

    def neighbours(self, x: int, y: int, radius: int = 1) -> list[tuple[int, int]]:
        cells = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.clamp(x + dx, y + dy)
                cells.append((nx, ny))
        return cells

    def consume_food(self, x: int, y: int) -> float:
        """Remove and return all food at (x, y)."""
        energy = float(self.food_grid[y, x])
        self.food_grid[y, x] = 0.0
        return energy

    def is_hazardous(self, x: int, y: int) -> bool:
        return bool(self.hazard_grid[y, x])

    # ── Queries ─────────────────────────────────────────────────────────────

    def food_sense(self, x: int, y: int, radius: int = 3) -> np.ndarray:
        """Return a flattened view of food values in a square neighbourhood."""
        side = 2 * radius + 1
        out = np.zeros((side, side), dtype=np.float32)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = self.clamp(x + dx, y + dy)
                out[dy + radius, dx + radius] = self.food_grid[ny, nx]
        return out.ravel()

    def total_food(self) -> float:
        return float(self.food_grid.sum())

    def __repr__(self) -> str:
        return f"World(w={self.width}, h={self.height}, wrap={self.wrap})"

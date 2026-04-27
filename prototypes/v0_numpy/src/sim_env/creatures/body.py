"""Body: physical representation and movement of a creature in the world."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Body parameter indices within Genome.body_params
_SPEED_IDX = 0
_SENSE_RADIUS_IDX = 1
_SIZE_IDX = 2
_METABOLISM_BONUS_IDX = 3


@dataclass
class Body:
    """Decoded physical traits of a creature.

    Traits are derived from ``genome.body_params`` via sigmoid scaling so
    every value is bounded regardless of the raw gene value.
    """

    x: int = 0
    y: int = 0
    direction: int = 0       # 0=N, 1=E, 2=S, 3=W
    energy: float = 50.0
    age: int = 0

    # Decoded traits (set by decode_from_genome)
    speed: float = 1.0          # cells per step (1 or 2)
    sense_radius: int = 2       # vision/food sensing radius
    size: float = 1.0           # influences metabolism cost
    metabolism_bonus: float = 0.0

    # Cumulative stats
    energy_collected: float = 0.0
    offspring_count: int = 0
    steps_alive: int = 0

    @classmethod
    def decode_from_genome(cls, params: np.ndarray, cfg: dict, x: int = 0, y: int = 0) -> "Body":
        """Construct Body from raw genome body_params array."""
        def _sig(v: float, lo: float, hi: float) -> float:
            s = 1.0 / (1.0 + np.exp(-float(v)))
            return lo + s * (hi - lo)

        b = cls(x=x, y=y)
        b.energy = cfg["creature"].get("initial_energy", 50.0)
        b.speed = 1 if _sig(params[_SPEED_IDX], 0, 1) < 0.5 else 2
        b.sense_radius = int(_sig(params[_SENSE_RADIUS_IDX], 1, 5))
        b.size = _sig(params[_SIZE_IDX], 0.5, 3.0)
        b.metabolism_bonus = _sig(params[_METABOLISM_BONUS_IDX], -0.5, 0.5)
        return b

    # ── Movement ─────────────────────────────────────────────────────────────

    _DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N, E, S, W

    def move(self, action: int, world: "Any") -> None:  # noqa: F821
        """Apply movement action: 0=forward, 1=turn_left, 2=turn_right, 3=stay."""
        if action == 1:
            self.direction = (self.direction - 1) % 4
        elif action == 2:
            self.direction = (self.direction + 1) % 4
        elif action == 0:
            dx, dy = self._DIRS[self.direction]
            nx, ny = world.clamp(self.x + dx * self.speed, self.y + dy * self.speed)
            # Only move if target cell is unoccupied
            if world.occupancy[ny][nx] is None:
                world.occupancy[self.y][self.x] = None
                self.x, self.y = nx, ny
                world.occupancy[ny][nx] = self  # will be set by creature wrapper
        # action == 3: stay — do nothing

    def tick(self, world: "Any", cfg: dict) -> bool:  # noqa: F821
        """Consume energy per step; return False if dead."""
        base = cfg["creature"].get("metabolism_rate", 0.1)
        cost = base * self.size + self.metabolism_bonus
        self.energy -= max(cost, 0.0)
        self.age += 1
        self.steps_alive += 1

        # Eat food at current location
        gained = world.consume_food(self.x, self.y)
        if gained > 0:
            self.energy += gained
            self.energy_collected += gained

        # Hazard damage
        if world.is_hazardous(self.x, self.y):
            self.energy -= 5.0

        max_age = cfg["creature"].get("max_age", 1000)
        return self.energy > 0 and self.age < max_age

    def can_reproduce(self, cfg: dict) -> bool:
        threshold = cfg["creature"].get("reproduction_threshold", 80.0)
        return self.energy >= threshold

    def pay_reproduction_cost(self, cfg: dict) -> None:
        self.energy -= cfg["creature"].get("initial_energy", 50.0) * 0.5
        self.offspring_count += 1

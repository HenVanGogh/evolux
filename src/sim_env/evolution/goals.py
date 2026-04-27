"""Goal system: defines goal objects that can be placed in the world.

Goals produce a scalar gradient field that the directed evolution engine
reads as a directional signal.  Goals can be static (fixed location/value)
or dynamic (moving, decaying, multi-modal).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.core.world import World
    from sim_env.creatures.creature import Creature


class BaseGoal(ABC):
    """Abstract goal / objective attached to the simulation world."""

    @abstractmethod
    def signal(self, creature: "Creature", world: "World") -> float:
        """Return a scalar value in [0, 1] indicating goal achievement."""

    @abstractmethod
    def step(self, world: "World", rng: np.random.Generator) -> None:
        """Advance the goal's own dynamics (optional)."""

    @abstractmethod
    def gradient_at(self, x: int, y: int, world: "World") -> np.ndarray:
        """Return a 2-D vector pointing in the direction of increasing goal signal."""


class PositionalGoal(BaseGoal):
    """Creature should reach a target (x, y) region on the map."""

    def __init__(self, target_x: int, target_y: int, radius: int = 4) -> None:
        self.target_x = target_x
        self.target_y = target_y
        self.radius = radius

    def signal(self, creature: "Creature", world: "World") -> float:
        dx = creature.body.x - self.target_x
        dy = creature.body.y - self.target_y
        dist = np.sqrt(dx * dx + dy * dy)
        return float(np.exp(-dist / max(self.radius, 1)))

    def step(self, world: "World", rng: np.random.Generator) -> None:
        pass  # static goal

    def gradient_at(self, x: int, y: int, world: "World") -> np.ndarray:
        dx = self.target_x - x
        dy = self.target_y - y
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        return np.array([dx / dist, dy / dist], dtype=np.float32)


class EnergyMaximisationGoal(BaseGoal):
    """Creatures should maximise accumulated energy."""

    def signal(self, creature: "Creature", world: "World") -> float:
        return np.tanh(creature.body.energy_collected / 100.0)

    def step(self, world: "World", rng: np.random.Generator) -> None:
        pass

    def gradient_at(self, x: int, y: int, world: "World") -> np.ndarray:
        # Point toward highest food density in local neighbourhood
        food = world.food_sense(x, y, radius=3)
        side = 7  # 2*3+1
        grid = food.reshape(side, side)
        cy, cx = np.unravel_index(grid.argmax(), grid.shape)
        dy = cy - side // 2
        dx = cx - side // 2
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        return np.array([dx / dist, dy / dist], dtype=np.float32)


class MovingGoal(BaseGoal):
    """A goal region that moves around the world randomly."""

    def __init__(self, x: int, y: int, speed: float = 0.5) -> None:
        self.x = float(x)
        self.y = float(y)
        self.speed = speed

    def signal(self, creature: "Creature", world: "World") -> float:
        dx = creature.body.x - int(self.x)
        dy = creature.body.y - int(self.y)
        return float(np.exp(-(dx * dx + dy * dy) / 25.0))

    def step(self, world: "World", rng: np.random.Generator) -> None:
        angle = rng.uniform(0, 2 * np.pi)
        self.x = (self.x + self.speed * np.cos(angle)) % world.width
        self.y = (self.y + self.speed * np.sin(angle)) % world.height

    def gradient_at(self, x: int, y: int, world: "World") -> np.ndarray:
        dx = int(self.x) - x
        dy = int(self.y) - y
        dist = np.sqrt(dx * dx + dy * dy) + 1e-6
        return np.array([dx / dist, dy / dist], dtype=np.float32)


class GoalSystem:
    """Container for multiple goals; aggregates their signals."""

    def __init__(self, goals: list[BaseGoal] | None = None) -> None:
        self.goals: list[BaseGoal] = goals or []

    def add_goal(self, goal: BaseGoal) -> None:
        self.goals.append(goal)

    def aggregate_signal(self, creature: "Creature", world: "World") -> float:
        if not self.goals:
            return 0.0
        return float(np.mean([g.signal(creature, world) for g in self.goals]))

    def step(self, world: "World", rng: np.random.Generator) -> None:
        for g in self.goals:
            g.step(world, rng)

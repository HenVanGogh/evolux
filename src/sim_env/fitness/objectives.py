"""Fitness objective definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.creatures.creature import Creature
    from sim_env.core.world import World


class Objective(ABC):
    """Single fitness component."""

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    @abstractmethod
    def evaluate(self, creature: "Creature", world: "World") -> float:
        """Return a non-negative scalar score for this creature."""

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SurvivalObjective(Objective):
    """Reward creatures that stayed alive longer."""

    name = "survival"

    def evaluate(self, creature: "Creature", world: "World") -> float:
        return float(creature.body.steps_alive)


class EnergyObjective(Objective):
    """Reward cumulative energy collected."""

    name = "energy_collected"

    def evaluate(self, creature: "Creature", world: "World") -> float:
        return float(creature.body.energy_collected)


class OffspringObjective(Objective):
    """Reward number of offspring produced."""

    name = "offspring_count"

    def evaluate(self, creature: "Creature", world: "World") -> float:
        return float(creature.body.offspring_count)


class ExplorationObjective(Objective):
    """Reward creatures for covering more unique cells."""

    name = "exploration"

    def evaluate(self, creature: "Creature", world: "World") -> float:
        # Placeholder: extend Creature to track visited cells for full impl
        return float(creature.body.steps_alive) * 0.1


# Registry for config-driven instantiation
_OBJECTIVE_REGISTRY: dict[str, type[Objective]] = {
    "survival": SurvivalObjective,
    "energy_collected": EnergyObjective,
    "offspring_count": OffspringObjective,
    "exploration": ExplorationObjective,
}


def build_objective(name: str, weight: float) -> Objective:
    cls = _OBJECTIVE_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown objective '{name}'. Available: {list(_OBJECTIVE_REGISTRY)}")
    return cls(weight=weight)

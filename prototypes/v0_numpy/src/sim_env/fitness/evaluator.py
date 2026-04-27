"""Fitness evaluator: computes and assigns creature fitness from objectives."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from sim_env.fitness.objectives import Objective, build_objective

if TYPE_CHECKING:
    from sim_env.core.world import World
    from sim_env.evolution.population import Population

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """Computes multi-objective fitness and writes it to each creature.

    Objectives are blended as a weighted sum.  Optionally the combined score
    is normalised across the population (z-score or min-max).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        fc = cfg.get("fitness", {})
        self._normalize: bool = fc.get("normalize", True)

        obj_specs = fc.get(
            "objectives",
            [{"name": "survival", "weight": 1.0}],
        )
        self.objectives: list[Objective] = [
            build_objective(o["name"], o.get("weight", 1.0)) for o in obj_specs
        ]

        logger.info(
            "FitnessEvaluator: %s",
            ", ".join(f"{o.name}×{o.weight}" for o in self.objectives),
        )

    def evaluate(self, population: "Population", world: "World") -> None:
        """Assign fitness to all creatures in *population*."""
        for creature in population.creatures:
            raw_scores = [obj.evaluate(creature, world) for obj in self.objectives]
            weighted = sum(s * o.weight for s, o in zip(raw_scores, self.objectives))
            creature.fitness = float(weighted)

        if self._normalize:
            self._normalise_population(population)

    def _normalise_population(self, population: "Population") -> None:
        fits = np.array([c.fitness for c in population.creatures], dtype=np.float64)
        std = fits.std()
        if std < 1e-10:
            return
        normed = (fits - fits.mean()) / std
        for c, f in zip(population.creatures, normed):
            c.fitness = float(f)

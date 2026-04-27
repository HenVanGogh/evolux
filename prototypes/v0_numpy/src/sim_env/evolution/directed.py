"""Directed evolution: augments raw fitness with gradient estimates and novelty.

Two modes
---------
gradient
    Approximates the evolutionary gradient via finite differences over the
    fitness landscape. The "direction" of improvement is estimated from the
    population and used to bias mutation step size or direction.

novelty_objective
    Blends fitness with a novelty score (behavioural distance to nearest
    neighbours in behaviour space), following the NS-ES / MAP-Elites approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.creatures.creature import Creature


class DirectedEvolution:
    """Augments creature fitness scores with directional signals.

    Call ``augment_fitness(creatures)`` after raw fitness evaluation to
    blend in gradient / novelty contributions.
    """

    def __init__(self, cfg: dict) -> None:
        dc = cfg["evolution"].get("directed", {})
        self.enabled: bool = dc.get("enabled", False)
        self.gradient_weight: float = dc.get("gradient_weight", 0.3)
        self.novelty_weight: float = dc.get("novelty_weight", 0.2)
        self.novelty_k: int = dc.get("novelty_k", 15)
        self.goal_field: str = dc.get("goal_field", "energy")

        # Behaviour archive (for novelty estimation across generations)
        self._behaviour_archive: list[np.ndarray] = []
        self._archive_max = 500

    # ── Main interface ───────────────────────────────────────────────────────

    def augment_fitness(self, creatures: list["Creature"]) -> None:
        """Modify creature.fitness in-place by blending gradient + novelty."""
        if not self.enabled:
            return

        # Build behaviour descriptors for this generation
        behaviours = [self._behaviour_descriptor(c) for c in creatures]

        if self.novelty_weight > 0.0:
            novelty_scores = self._compute_novelty(behaviours)
        else:
            novelty_scores = np.zeros(len(creatures))

        if self.gradient_weight > 0.0:
            gradient_scores = self._estimate_gradient(creatures)
        else:
            gradient_scores = np.zeros(len(creatures))

        # Normalise
        novelty_scores = _safe_normalize(novelty_scores)
        gradient_scores = _safe_normalize(gradient_scores)
        raw_fits = np.array([c.fitness for c in creatures])
        raw_fits = _safe_normalize(raw_fits)

        obj_w = 1.0 - self.novelty_weight - self.gradient_weight
        for i, c in enumerate(creatures):
            c.fitness = float(
                obj_w * raw_fits[i]
                + self.novelty_weight * novelty_scores[i]
                + self.gradient_weight * gradient_scores[i]
            )

        # Update archive with a random subset
        for b in behaviours[: max(1, len(behaviours) // 5)]:
            self._behaviour_archive.append(b)
        # Trim archive
        if len(self._behaviour_archive) > self._archive_max:
            self._behaviour_archive = self._behaviour_archive[-self._archive_max :]

    # ── Behaviour descriptor ────────────────────────────────────────────────

    def _behaviour_descriptor(self, creature: "Creature") -> np.ndarray:
        """Compact vector characterising creature behaviour for novelty search."""
        b = creature.body
        return np.array(
            [
                b.x / 64.0,
                b.y / 64.0,
                b.energy_collected / 100.0,
                b.steps_alive / 1000.0,
                float(b.offspring_count),
            ],
            dtype=np.float32,
        )

    # ── Novelty ─────────────────────────────────────────────────────────────

    def _compute_novelty(self, behaviours: list[np.ndarray]) -> np.ndarray:
        """Average distance to k nearest neighbours in behaviour space."""
        all_b = np.stack(behaviours + self._behaviour_archive, axis=0)
        current = np.stack(behaviours, axis=0)
        n = len(behaviours)
        k = min(self.novelty_k, all_b.shape[0] - 1)

        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            dists = np.linalg.norm(all_b - current[i], axis=1)
            dists[i] = np.inf  # exclude self
            knn = np.sort(dists)[:k]
            scores[i] = knn.mean() if len(knn) > 0 else 0.0
        return scores

    # ── Gradient estimation ─────────────────────────────────────────────────

    def _estimate_gradient(self, creatures: list["Creature"]) -> np.ndarray:
        """Estimate local gradient by correlating fitness with goal-field signal.

        The "gradient score" rewards creatures that moved in the direction of
        increasing goal signal relative to the population mean.
        """
        goal_vals = np.array([self._goal_signal(c) for c in creatures], dtype=np.float32)
        fits = np.array([c.fitness for c in creatures], dtype=np.float32)

        # Gradient proxy: creatures whose goal signal is high relative to mean
        # AND whose fitness increased get a positive boost.
        mean_goal = goal_vals.mean()
        mean_fit = fits.mean()
        gradient_scores = (goal_vals - mean_goal) * (fits - mean_fit)
        return gradient_scores

    def _goal_signal(self, creature: "Creature") -> float:
        """Extract the goal-relevant signal from a creature."""
        b = creature.body
        field = self.goal_field
        if field == "energy":
            return b.energy
        elif field == "energy_collected":
            return b.energy_collected
        elif field == "age":
            return float(b.age)
        elif field == "offspring_count":
            return float(b.offspring_count)
        else:
            return creature.fitness


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std

"""Selection strategies for choosing parents from a population."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.creatures.creature import Creature


class SelectionStrategy:
    """Factory + implementations for parent selection."""

    def __init__(self, cfg: dict) -> None:
        evo = cfg["evolution"]
        self.method: str = evo.get("selection", "tournament")
        self.tournament_k: int = evo.get("tournament_k", 5)
        self.elitism: int = evo.get("elitism", 2)

    def select_parents(
        self,
        creatures: list["Creature"],
        n: int,
        rng: np.random.Generator,
    ) -> list["Creature"]:
        """Return *n* selected parents (with replacement)."""
        if self.method == "tournament":
            return [self._tournament(creatures, rng) for _ in range(n)]
        elif self.method == "roulette":
            return self._roulette(creatures, n, rng)
        elif self.method == "lexicase":
            return self._lexicase(creatures, n, rng)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

    def get_elites(self, creatures: list["Creature"]) -> list["Creature"]:
        """Return the top-k creatures unchanged."""
        if self.elitism <= 0:
            return []
        sorted_c = sorted(creatures, key=lambda c: c.fitness, reverse=True)
        return sorted_c[: self.elitism]

    # ── Tournament ─────────────────────────────────────────────────────────

    def _tournament(
        self, creatures: list["Creature"], rng: np.random.Generator
    ) -> "Creature":
        k = min(self.tournament_k, len(creatures))
        contestants = [creatures[i] for i in rng.choice(len(creatures), size=k, replace=False)]
        return max(contestants, key=lambda c: c.fitness)

    # ── Roulette (fitness-proportionate) ──────────────────────────────────

    def _roulette(
        self,
        creatures: list["Creature"],
        n: int,
        rng: np.random.Generator,
    ) -> list["Creature"]:
        fits = np.array([max(c.fitness, 0.0) for c in creatures], dtype=np.float64)
        total = fits.sum()
        if total < 1e-12:
            probs = np.ones(len(creatures)) / len(creatures)
        else:
            probs = fits / total
        idxs = rng.choice(len(creatures), size=n, p=probs, replace=True)
        return [creatures[i] for i in idxs]

    # ── Lexicase ───────────────────────────────────────────────────────────
    # Simplified: uses fitness as the single case here; extend with
    # multi-objective fitness vectors for true lexicase.

    def _lexicase(
        self,
        creatures: list["Creature"],
        n: int,
        rng: np.random.Generator,
    ) -> list["Creature"]:
        selected = []
        for _ in range(n):
            pool = list(creatures)
            rng.shuffle(pool)
            # Single-objective simplified lexicase: best on the shuffled order
            selected.append(max(pool, key=lambda c: c.fitness))
        return selected

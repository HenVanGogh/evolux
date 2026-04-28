"""k-tournament selection with configurable elitism (Phase 1).

Tournament selection draws ``k`` individuals at random from the population and
returns the best-fitness individual. Repeating this ``n`` times yields ``n``
parent genomes.  Elitism preserves the top ``n_elites`` individuals unchanged
into the next generation.
"""

from __future__ import annotations

import torch

from evolux.core.protocols import Genome
from evolux.core.protocols import Population as PopulationProtocol


class Tournament:
    """k-tournament selection with configurable elitism.

    Parameters
    ----------
    k:
        Tournament group size.  Larger ``k`` raises selection pressure.
        Must be ≥ 1.
    n_elites:
        Number of top-fitness individuals copied verbatim to the next
        generation.  Must be ≥ 0.
    """

    def __init__(self, k: int = 3, n_elites: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}")
        if n_elites < 0:
            raise ValueError(f"n_elites must be ≥ 0, got {n_elites}")
        self.k: int = k
        self.n_elites: int = n_elites

    # ── Selector Protocol ────────────────────────────────────────────────────

    def select_parents(
        self,
        pop: PopulationProtocol,
        n: int,
        rng: torch.Generator,
    ) -> list[Genome]:
        """Select *n* parent genomes from *pop* using k-tournament.

        Each tournament:
        1. Samples ``k`` candidate indices (with replacement).
        2. Returns the genome with the highest fitness among those candidates.

        Parameters
        ----------
        pop:
            Current population (must be non-empty).
        n:
            Number of parents to select.
        rng:
            Torch generator for reproducible sampling.

        Returns
        -------
        list[Genome]
            ``n`` selected parent genomes (may contain duplicates).

        Raises
        ------
        ValueError
            If *pop* is empty.
        """
        P = len(pop.genomes)
        if P == 0:
            raise ValueError("select_parents: population is empty.")

        selected: list[Genome] = []
        for _ in range(n):
            indices = torch.randint(0, P, (self.k,), generator=rng).tolist()
            best_idx = int(max(indices, key=lambda i: pop.fitness[i].item()))
            selected.append(pop.genomes[best_idx])
        return selected

    def get_elites(self, pop: PopulationProtocol) -> list[Genome]:
        """Return the ``n_elites`` top-fitness genomes from *pop*.

        Parameters
        ----------
        pop:
            Current population.

        Returns
        -------
        list[Genome]
            Elite genomes, ordered best-first.  Returns fewer than
            ``n_elites`` if the population is smaller.
        """
        n = min(self.n_elites, len(pop.genomes))
        if n == 0:
            return []
        top_indices = torch.topk(pop.fitness, n).indices.tolist()
        return [pop.genomes[i] for i in top_indices]

    def __repr__(self) -> str:
        return f"Tournament(k={self.k}, n_elites={self.n_elites})"

"""Shared genetic operator utilities for all evolution strategies.

These helpers operate on lists of genomes and GenomeOperators, providing
building blocks for strategies like tournament selection, MAP-Elites, etc.
"""

from __future__ import annotations

import torch

from evolux.core.protocols import Genome, GenomeOperator


def reproduce(
    parents: list[Genome],
    operator: GenomeOperator,
    n: int,
    rng: torch.Generator,
) -> list[Genome]:
    """Create *n* offspring from *parents* using crossover then mutation.

    Each offspring is produced by:
    1. Sampling two parents from the pool uniformly at random (with replacement).
    2. Applying ``operator.crossover`` to produce a child.
    3. Applying ``operator.mutate`` to the child.

    Parameters
    ----------
    parents:
        Pool of parent genomes to breed from.  Must be non-empty.
    operator:
        ``GenomeOperator`` providing ``crossover`` and ``mutate``.
    n:
        Number of offspring to produce.
    rng:
        Torch generator for reproducible sampling.

    Returns
    -------
    list[Genome]
        ``n`` freshly-created offspring genomes.

    Raises
    ------
    ValueError
        If *parents* is empty.
    """
    if not parents:
        raise ValueError("reproduce: parents list must be non-empty.")

    P = len(parents)
    offspring: list[Genome] = []
    for _ in range(n):
        idx_a = int(torch.randint(0, P, (1,), generator=rng).item())
        idx_b = int(torch.randint(0, P, (1,), generator=rng).item())
        child = operator.crossover(parents[idx_a], parents[idx_b], rng)
        child = operator.mutate(child, rng)
        offspring.append(child)
    return offspring

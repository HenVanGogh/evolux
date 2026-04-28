"""Concrete Population implementation for the evolution module (Phase 1).

A ``Population`` is the container that carries the current generation of
genomes together with their fitness scores.  It implements the
``evolux.core.protocols.Population`` Protocol (the three required attributes)
and adds the ``add``, ``select``, and ``step_generation`` helpers used by the
evolution loop.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evolux.core.protocols import Genome, GenomeOperator, Selector
from evolux.core.protocols import Population as PopulationProtocol
from evolux.evolution.operators import reproduce


class Population:
    """Concrete ``Population`` implementation.

    Holds a generation of genomes and their cached fitness scores.

    Attributes
    ----------
    genomes:
        Ordered list of :class:`~evolux.core.protocols.Genome` objects in the
        current generation.
    fitness:
        Per-genome fitness values, shape ``(P,)`` (float32).  Offspring that
        have not yet been evaluated carry ``-inf``.
    behaviour:
        Optional behavioural descriptor tensor, shape ``(P, D_b)``.  Used by
        novelty search and MAP-Elites (Phase 3).  ``None`` until set.
    """

    def __init__(
        self,
        genomes: list[Genome],
        fitness: Tensor,
        behaviour: Tensor | None = None,
    ) -> None:
        if len(genomes) != fitness.shape[0]:
            raise ValueError(
                f"genomes length ({len(genomes)}) must match fitness length ({fitness.shape[0]})"
            )
        self.genomes: list[Genome] = list(genomes)
        self.fitness: Tensor = fitness.float().clone()
        self.behaviour: Tensor | None = behaviour

    # ── Protocol conformance check ───────────────────────────────────────────

    @staticmethod
    def _assert_protocol() -> None:
        """Assert this class satisfies the Population Protocol at import time."""
        # Checked lazily via isinstance in tests; nothing to do here.

    # ── Factory helpers ──────────────────────────────────────────────────────

    @classmethod
    def from_genomes(
        cls,
        genomes: list[Genome],
        device: torch.device | str = "cpu",
    ) -> Population:
        """Create a population from a list of genomes (all fitness = ``-inf``).

        Parameters
        ----------
        genomes:
            Initial genome list.
        device:
            Device for the fitness tensor.

        Returns
        -------
        Population
            New population with unevaluated fitness.
        """
        P = len(genomes)
        fitness = torch.full((P,), float("-inf"), dtype=torch.float32, device=torch.device(device))
        return cls(genomes, fitness)

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def add(self, genome: Genome, fitness: float = float("-inf")) -> None:
        """Append a genome and its fitness score to the population.

        Parameters
        ----------
        genome:
            Genome to append.
        fitness:
            Initial fitness value.  Defaults to ``-inf`` (= not yet evaluated).
        """
        self.genomes.append(genome)
        new_fit = torch.tensor([fitness], dtype=torch.float32, device=self.fitness.device)
        self.fitness = torch.cat([self.fitness, new_fit])

    def select(
        self,
        selector: Selector,
        n: int,
        rng: torch.Generator,
    ) -> list[Genome]:
        """Select *n* parent genomes using *selector*.

        Convenience wrapper around ``selector.select_parents``.

        Parameters
        ----------
        selector:
            Selector implementation (e.g. :class:`~evolux.evolution.tournament.Tournament`).
        n:
            Number of parents to select.
        rng:
            Torch generator.

        Returns
        -------
        list[Genome]
            ``n`` selected genomes.
        """
        return selector.select_parents(self, n, rng)

    def step_generation(
        self,
        selector: Selector,
        operator: GenomeOperator,
        rng: torch.Generator,
    ) -> None:
        """Advance the population by one generation **in-place**.

        Algorithm
        ---------
        1. Collect elites via ``selector.get_elites``.
        2. Select ``P - n_elites`` parents for offspring.
        3. Produce offspring via :func:`~evolux.evolution.operators.reproduce`.
        4. Replace ``genomes`` with elites + offspring.
        5. Reset ``fitness``: elites keep their previous scores; offspring get
           ``-inf`` (they need to be re-evaluated).

        Parameters
        ----------
        selector:
            Drives parent selection and elitism.
        operator:
            Provides ``crossover`` and ``mutate`` for offspring generation.
        rng:
            Torch generator for deterministic reproduction.
        """
        P = len(self.genomes)
        if P == 0:
            return

        # ── Elites ──────────────────────────────────────────────────────────
        elites = selector.get_elites(self)
        n_elites = len(elites)

        # Preserve elite fitness values (same ordering as get_elites returns)
        if n_elites > 0:
            elite_indices = torch.topk(self.fitness, n_elites).indices
            elite_fitness = self.fitness[elite_indices]
        else:
            elite_fitness = torch.empty(0, dtype=torch.float32, device=self.fitness.device)

        # ── Offspring ────────────────────────────────────────────────────────
        n_offspring = P - n_elites
        if n_offspring > 0:
            parents = selector.select_parents(self, n_offspring, rng)
            offspring = reproduce(parents, operator, n_offspring, rng)
        else:
            offspring = []

        # ── Update state ─────────────────────────────────────────────────────
        new_fitness = torch.full(
            (P,), float("-inf"), dtype=torch.float32, device=self.fitness.device
        )
        if n_elites > 0:
            new_fitness[:n_elites] = elite_fitness

        self.genomes = elites + offspring
        self.fitness = new_fitness
        self.behaviour = None

    def __len__(self) -> int:
        return len(self.genomes)

    def __repr__(self) -> str:
        P = len(self.genomes)
        best = self.fitness.max().item() if P > 0 else float("-inf")
        return f"Population(size={P}, best_fitness={best:.4g})"


def _check_protocol() -> None:  # pragma: no cover
    """Verify Population satisfies the Protocol at module load (dev-time check)."""
    _: PopulationProtocol = Population([], torch.empty(0))  # type: ignore[assignment]

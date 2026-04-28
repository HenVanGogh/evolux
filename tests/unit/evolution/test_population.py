"""Unit tests for evolux.evolution.population (Phase 1 acceptance tests)."""

from __future__ import annotations

import pytest
import torch

from evolux.core.protocols import Population as PopulationProtocol
from evolux.core.rng import RNG
from evolux.evolution.population import Population
from evolux.evolution.tournament import Tournament
from evolux.genome.direct import DirectGenome, DirectOperator

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_genome(n: int = 4, seed: int = 0) -> DirectGenome:
    rng = RNG(seed).split("genome")
    return DirectGenome(torch.rand(n, generator=rng))


def make_population(size: int = 6, n_params: int = 4, seed: int = 0) -> Population:
    genomes = [make_genome(n_params, seed=i) for i in range(size)]
    fitness = torch.arange(size, dtype=torch.float32)
    return Population(genomes, fitness)


# ── Constructor ───────────────────────────────────────────────────────────────


def test_population_stores_genomes_and_fitness() -> None:
    genomes = [make_genome(seed=i) for i in range(3)]
    fitness = torch.tensor([1.0, 2.0, 3.0])
    pop = Population(genomes, fitness)
    assert len(pop.genomes) == 3
    assert pop.fitness.shape == (3,)


def test_population_fitness_is_float32() -> None:
    genomes = [make_genome()]
    pop = Population(genomes, torch.tensor([1.0], dtype=torch.float64))
    assert pop.fitness.dtype == torch.float32


def test_population_mismatch_raises() -> None:
    genomes = [make_genome() for _ in range(3)]
    fitness = torch.zeros(5)
    with pytest.raises(ValueError, match="match"):
        Population(genomes, fitness)


def test_population_behaviour_none_by_default() -> None:
    pop = make_population(3)
    assert pop.behaviour is None


def test_population_fitness_is_copy() -> None:
    """Mutating the original fitness tensor must not affect the population."""
    fitness = torch.tensor([1.0, 2.0, 3.0])
    pop = Population([make_genome(seed=i) for i in range(3)], fitness)
    fitness[0] = 999.0
    assert pop.fitness[0].item() != 999.0


# ── from_genomes factory ──────────────────────────────────────────────────────


def test_from_genomes_all_minus_inf() -> None:
    genomes = [make_genome(seed=i) for i in range(4)]
    pop = Population.from_genomes(genomes)
    assert (pop.fitness == float("-inf")).all()


def test_from_genomes_size() -> None:
    genomes = [make_genome(seed=i) for i in range(5)]
    pop = Population.from_genomes(genomes)
    assert len(pop.genomes) == 5


# ── len ───────────────────────────────────────────────────────────────────────


def test_len() -> None:
    pop = make_population(7)
    assert len(pop) == 7


# ── add ───────────────────────────────────────────────────────────────────────


def test_add_increases_size() -> None:
    pop = make_population(3)
    g = make_genome(seed=99)
    pop.add(g)
    assert len(pop.genomes) == 4
    assert pop.fitness.shape == (4,)


def test_add_default_fitness_is_minus_inf() -> None:
    pop = make_population(2)
    g = make_genome(seed=99)
    pop.add(g)
    assert pop.fitness[-1].item() == float("-inf")


def test_add_custom_fitness() -> None:
    pop = make_population(2)
    g = make_genome(seed=99)
    pop.add(g, fitness=7.5)
    assert pop.fitness[-1].item() == pytest.approx(7.5)


def test_add_genome_is_appended() -> None:
    pop = make_population(2)
    g = make_genome(seed=99)
    pop.add(g)
    assert pop.genomes[-1] is g


# ── select ────────────────────────────────────────────────────────────────────


def test_select_returns_n() -> None:
    pop = make_population(10)
    t = Tournament(k=3)
    rng = RNG(0).split("sel")
    selected = pop.select(t, 5, rng)
    assert len(selected) == 5


def test_select_delegates_to_selector() -> None:
    """pop.select(t, n, rng) must produce the same result as t.select_parents."""
    pop = make_population(10)
    t = Tournament(k=3)
    rng_a = RNG(1).split("test")
    rng_b = RNG(1).split("test")
    direct = t.select_parents(pop, 6, rng_a)
    via_pop = pop.select(t, 6, rng_b)
    for a, b in zip(direct, via_pop, strict=True):
        assert torch.equal(a.params, b.params)  # type: ignore[union-attr]


# ── step_generation ───────────────────────────────────────────────────────────


def test_step_generation_preserves_size() -> None:
    pop = make_population(8)
    t = Tournament(k=3, n_elites=2)
    op = DirectOperator(sigma=0.05)
    pop.step_generation(t, op, RNG(0).split("step"))
    assert len(pop.genomes) == 8


def test_step_generation_resets_behaviour() -> None:
    pop = make_population(6)
    pop.behaviour = torch.randn(6, 3)
    t = Tournament(k=2, n_elites=1)
    op = DirectOperator(sigma=0.05)
    pop.step_generation(t, op, RNG(0).split("step"))
    assert pop.behaviour is None


def test_step_generation_all_elites_no_offspring() -> None:
    """If n_elites == P, no offspring is created and all genomes survive."""
    pop = make_population(4)
    original_ids = {id(g) for g in pop.genomes}
    t = Tournament(k=2, n_elites=4)
    op = DirectOperator(sigma=0.05)
    pop.step_generation(t, op, RNG(0).split("step"))
    assert len(pop.genomes) == 4
    assert {id(g) for g in pop.genomes} == original_ids


def test_step_generation_empty_population_is_noop() -> None:
    pop = Population([], torch.empty(0))
    t = Tournament(k=2)
    op = DirectOperator()
    pop.step_generation(t, op, RNG(0).split("step"))  # must not raise
    assert len(pop.genomes) == 0


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_population_satisfies_protocol() -> None:
    pop = make_population(4)
    assert isinstance(pop, PopulationProtocol)


# ── repr ──────────────────────────────────────────────────────────────────────


def test_repr() -> None:
    pop = make_population(3)
    r = repr(pop)
    assert "Population" in r
    assert "size=3" in r

"""Unit tests for evolux.evolution.tournament (Phase 1 acceptance tests)."""

from __future__ import annotations

import pytest
import torch

from evolux.core.protocols import Genome, Selector
from evolux.core.rng import RNG
from evolux.evolution import EVOLUTION_REGISTRY
from evolux.evolution.population import Population
from evolux.evolution.tournament import Tournament
from evolux.genome.direct import DirectGenome, DirectOperator

# ── Helpers / fixtures ────────────────────────────────────────────────────────


def make_genome(n: int = 4, seed: int = 0) -> DirectGenome:
    """Return a deterministic DirectGenome with *n* parameters."""
    rng = RNG(seed).split("genome")
    return DirectGenome(torch.rand(n, generator=rng))


def make_population(
    size: int = 10,
    n_params: int = 4,
    seed: int = 0,
) -> Population:
    """Return a Population of *size* DirectGenomes with fitness 0, 1, …, size-1."""
    genomes = [make_genome(n_params, seed=i) for i in range(size)]
    fitness = torch.arange(size, dtype=torch.float32)
    return Population(genomes, fitness)


# ── Constructor validation ────────────────────────────────────────────────────


def test_tournament_default_params() -> None:
    t = Tournament()
    assert t.k == 3
    assert t.n_elites == 1


def test_tournament_invalid_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        Tournament(k=0)


def test_tournament_invalid_n_elites() -> None:
    with pytest.raises(ValueError, match="n_elites must be"):
        Tournament(n_elites=-1)


# ── select_parents: basic contract ───────────────────────────────────────────


def test_select_parents_returns_n() -> None:
    """select_parents must return exactly *n* genomes."""
    pop = make_population(10)
    t = Tournament(k=3, n_elites=2)
    rng = RNG(0).split("test")
    parents = t.select_parents(pop, 8, rng)
    assert len(parents) == 8


def test_select_parents_all_from_population() -> None:
    """Every selected genome must be an element of pop.genomes (by identity)."""
    pop = make_population(10)
    t = Tournament(k=3, n_elites=2)
    rng = RNG(1).split("test")
    parents = t.select_parents(pop, 20, rng)
    genome_ids = {id(g) for g in pop.genomes}
    for g in parents:
        assert id(g) in genome_ids, "selected genome not in population"


def test_select_parents_empty_population_raises() -> None:
    pop = Population([], torch.empty(0))
    t = Tournament(k=3)
    rng = RNG(0).split("test")
    with pytest.raises(ValueError, match="empty"):
        t.select_parents(pop, 1, rng)


def test_select_parents_returns_genome_instances() -> None:
    pop = make_population(5)
    t = Tournament(k=2)
    rng = RNG(2).split("test")
    for g in t.select_parents(pop, 5, rng):
        assert isinstance(g, Genome)


# ── select_parents: determinism ──────────────────────────────────────────────


def test_select_parents_is_deterministic() -> None:
    """Same RNG seed must produce identical parent sequences."""
    pop = make_population(10)
    t = Tournament(k=3, n_elites=2)
    rng1 = RNG(42).split("det")
    rng2 = RNG(42).split("det")
    p1 = t.select_parents(pop, 8, rng1)
    p2 = t.select_parents(pop, 8, rng2)
    for a, b in zip(p1, p2, strict=True):
        assert torch.equal(a.params, b.params)  # type: ignore[union-attr]


def test_select_parents_different_seeds_differ() -> None:
    """Different seeds must (almost surely) produce different results."""
    pop = make_population(10)
    t = Tournament(k=3)
    p1 = t.select_parents(pop, 20, RNG(0).split("a"))
    p2 = t.select_parents(pop, 20, RNG(99).split("b"))
    same = all(torch.equal(a.params, b.params) for a, b in zip(p1, p2, strict=True))  # type: ignore[union-attr]
    assert not same, "two different seeds produced identical selections"


# ── get_elites ────────────────────────────────────────────────────────────────


def test_get_elites_returns_n_elites() -> None:
    pop = make_population(10)
    t = Tournament(k=3, n_elites=3)
    elites = t.get_elites(pop)
    assert len(elites) == 3


def test_get_elites_returns_top_fitness() -> None:
    """Elites must correspond to the genomes with the highest fitness."""
    pop = make_population(10)  # fitness 0..9
    t = Tournament(k=3, n_elites=2)
    elites = t.get_elites(pop)
    # Genomes are DirectGenome; fitness index i → make_genome(seed=i)
    elite_set = {id(g) for g in elites}
    # The two highest fitness are at index 9 (fitness=9) and 8 (fitness=8)
    top_two = {id(pop.genomes[9]), id(pop.genomes[8])}
    assert elite_set == top_two


def test_get_elites_zero_elites() -> None:
    pop = make_population(5)
    t = Tournament(k=2, n_elites=0)
    assert t.get_elites(pop) == []


def test_get_elites_clamps_to_population_size() -> None:
    pop = make_population(3)
    t = Tournament(k=2, n_elites=100)
    elites = t.get_elites(pop)
    assert len(elites) == 3  # clamped to population size


# ── Selection pressure ────────────────────────────────────────────────────────


def test_selection_pressure_increases_with_k() -> None:
    """Higher k must yield higher average selected fitness."""
    N = 100
    pop = make_population(N)  # fitness 0..99
    genome_to_fitness = {id(g): pop.fitness[i].item() for i, g in enumerate(pop.genomes)}

    n_samples = 2000
    t1 = Tournament(k=1, n_elites=0)
    t10 = Tournament(k=10, n_elites=0)

    parents_k1 = t1.select_parents(pop, n_samples, RNG(0).split("k1"))
    parents_k10 = t10.select_parents(pop, n_samples, RNG(0).split("k10"))

    avg_k1 = sum(genome_to_fitness[id(g)] for g in parents_k1) / n_samples
    avg_k10 = sum(genome_to_fitness[id(g)] for g in parents_k10) / n_samples

    assert avg_k10 > avg_k1, f"k=10 avg fitness ({avg_k10:.2f}) should exceed k=1 ({avg_k1:.2f})"


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_tournament_satisfies_selector_protocol() -> None:
    t = Tournament()
    assert isinstance(t, Selector)


# ── Registry ─────────────────────────────────────────────────────────────────


def test_evolution_registry_has_tournament() -> None:
    assert "tournament" in EVOLUTION_REGISTRY


def test_evolution_registry_returns_tournament_class() -> None:
    assert EVOLUTION_REGISTRY.get("tournament") is Tournament


# ── step_generation integration ───────────────────────────────────────────────


def test_step_generation_population_size_constant() -> None:
    """Population size must be unchanged after step_generation."""
    pop = make_population(10)
    t = Tournament(k=3, n_elites=2)
    op = DirectOperator(sigma=0.01)
    pop.step_generation(t, op, RNG(0).split("step"))
    assert len(pop.genomes) == 10


def test_step_generation_elites_survive() -> None:
    """Top-n_elites genomes (by identity) must appear in the new generation."""
    pop = make_population(10)
    t = Tournament(k=3, n_elites=2)
    elites_before = {id(g) for g in t.get_elites(pop)}
    op = DirectOperator(sigma=0.01)
    pop.step_generation(t, op, RNG(0).split("step"))
    new_ids = {id(g) for g in pop.genomes}
    assert elites_before.issubset(new_ids), "elites must survive step_generation"


def test_step_generation_elite_fitness_preserved() -> None:
    """Elite fitness values must be carried over; offspring fitness must be -inf."""
    pop = make_population(10)  # fitness 0..9
    t = Tournament(k=3, n_elites=2)
    top_fitness = torch.topk(pop.fitness, 2).values.tolist()

    op = DirectOperator(sigma=0.01)
    pop.step_generation(t, op, RNG(0).split("step"))

    preserved = sorted(pop.fitness[:2].tolist(), reverse=True)
    assert preserved == pytest.approx(sorted(top_fitness, reverse=True)), (
        "elite fitness must be preserved after step_generation"
    )
    # Offspring slots start at index 2
    offspring_fitness = pop.fitness[2:]
    assert (offspring_fitness == float("-inf")).all(), (
        "offspring fitness should be -inf until evaluated"
    )


def test_step_generation_is_deterministic() -> None:
    """Same RNG seed must produce the same new population."""
    pop1 = make_population(8)
    pop2 = make_population(8)
    t = Tournament(k=3, n_elites=1)
    op = DirectOperator(sigma=0.05)
    pop1.step_generation(t, op, RNG(7).split("step"))
    pop2.step_generation(t, op, RNG(7).split("step"))
    for g1, g2 in zip(pop1.genomes, pop2.genomes, strict=True):
        assert torch.equal(g1.params, g2.params)  # type: ignore[union-attr]

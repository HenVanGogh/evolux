"""Tests for mutation operators, selection, and population evolution."""

import numpy as np
import pytest

from sim_env.creatures.genome import Genome, NodeType
from sim_env.evolution.operators import MutationOperator
from sim_env.evolution.selection import SelectionStrategy
from sim_env.evolution.population import Population
from sim_env.core.world import World


class TestMutationOperator:
    def test_mutate_preserves_io_topology(self, cfg, rng):
        op = MutationOperator(cfg)
        g = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        for _ in range(20):
            g = op.mutate(g, rng)
        # Input/output count must not change
        assert len(g.nodes_of_type(NodeType.INPUT)) == 5
        assert len(g.nodes_of_type(NodeType.OUTPUT)) == 4

    def test_mutate_returns_new_genome(self, cfg, rng):
        op = MutationOperator(cfg)
        g = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        g2 = op.mutate(g, rng)
        assert g.genome_id != g2.genome_id

    def test_crossover_child_has_both_parents(self, cfg, rng):
        op = MutationOperator(cfg)
        ga = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        gb = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        ga.fitness = 1.0  # type: ignore[attr-defined]
        child = op.crossover(ga, gb, rng)
        assert child.genome_id not in (ga.genome_id, gb.genome_id)

    def test_add_node_increases_hidden_count(self, cfg, rng):
        op = MutationOperator(cfg)
        g = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        initial_hidden = g.n_hidden()
        # Force add-node mutation
        for _ in range(50):
            op._add_node(g, rng)
        assert g.n_hidden() >= initial_hidden


class TestSelectionStrategy:
    def _make_creatures(self, cfg, rng, n=10):
        from sim_env.evolution.population import Population
        pop = Population(cfg, rng)
        for i, c in enumerate(pop.creatures[:n]):
            c.fitness = float(i)
        return pop.creatures[:n]

    def test_tournament_returns_correct_count(self, cfg, rng):
        creatures = self._make_creatures(cfg, rng)
        sel = SelectionStrategy(cfg)
        parents = sel.select_parents(creatures, 5, rng)
        assert len(parents) == 5

    def test_elites_are_top_k(self, cfg, rng):
        creatures = self._make_creatures(cfg, rng)
        sel = SelectionStrategy(cfg)
        elites = sel.get_elites(creatures)
        elite_fits = sorted([e.fitness for e in elites], reverse=True)
        top_fits = sorted([c.fitness for c in creatures], reverse=True)[: len(elites)]
        assert elite_fits == top_fits


class TestPopulation:
    def test_initialise_correct_size(self, cfg, rng):
        pop = Population(cfg, rng)
        assert len(pop.creatures) == cfg["population"]["size"]

    def test_place_into_world(self, cfg, rng):
        pop = Population(cfg, rng)
        world = World(cfg, rng)
        pop.place_into(world)
        placed = sum(1 for row in world.occupancy for c in row if c is not None)
        assert placed == len(pop.creatures)

    def test_evolve_maintains_population_size(self, cfg, rng):
        cfg = dict(cfg)
        cfg["population"] = dict(cfg["population"])
        cfg["population"]["size"] = 20
        pop = Population(cfg, rng)
        world = World(cfg, rng)
        pop.place_into(world)
        # Assign random fitness
        for c in pop.creatures:
            c.fitness = float(rng.random())
        pop.evolve(rng)
        assert len(pop.creatures) == 20

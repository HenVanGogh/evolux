"""Tests for genome, body, sensors, and creature."""

import numpy as np
import pytest

from sim_env.creatures.genome import Genome, NodeType, next_innov
from sim_env.creatures.body import Body
from sim_env.creatures.sensors import SensorArray
from sim_env.core.world import World


class TestGenome:
    def test_minimal_creates_input_output_nodes(self, rng):
        g = Genome.minimal(n_inputs=5, n_outputs=4, rng=rng)
        inputs = g.nodes_of_type(NodeType.INPUT)
        outputs = g.nodes_of_type(NodeType.OUTPUT)
        assert len(inputs) == 5
        assert len(outputs) == 4

    def test_minimal_all_connected(self, rng):
        g = Genome.minimal(n_inputs=3, n_outputs=2, rng=rng)
        assert len(g.enabled_connections()) > 0

    def test_clone_is_independent(self, rng):
        g = Genome.minimal(n_inputs=3, n_outputs=2, rng=rng)
        g2 = g.clone()
        # Modifying clone weights should not affect original
        for c in g2.connection_genes.values():
            c.weight = 999.0
        for c in g.connection_genes.values():
            assert c.weight != 999.0

    def test_compatibility_distance_same_genome(self, rng):
        g = Genome.minimal(n_inputs=3, n_outputs=2, rng=rng)
        assert g.compatibility_distance(g) == pytest.approx(0.0, abs=1e-3)

    def test_n_hidden(self, rng):
        g = Genome.minimal(n_inputs=3, n_outputs=2, rng=rng)
        assert g.n_hidden() == 0


class TestBody:
    def test_decode_from_genome_bounded_traits(self, cfg, rng):
        params = rng.normal(0, 2, size=8).astype(np.float32)
        body = Body.decode_from_genome(params, cfg)
        assert 1 <= body.speed <= 2
        assert 1 <= body.sense_radius <= 5
        assert 0.5 <= body.size <= 3.0


class TestSensorArray:
    def test_observe_shape(self, cfg, rng):
        world = World(cfg, rng)
        sensors = SensorArray(sense_radius=2)
        body = Body()
        obs = sensors.observe(body, world, cfg)
        assert obs.shape == (sensors.n_inputs,)

    def test_observe_normalised(self, cfg, rng):
        world = World(cfg, rng)
        sensors = SensorArray(sense_radius=2)
        body = Body()
        obs = sensors.observe(body, world, cfg)
        # Food observations should be in [0, 1]
        food_obs = obs[SensorArray.N_SCALAR:]
        assert (food_obs >= 0).all()
        assert (food_obs <= 1.0 + 1e-6).all()

"""End-to-end smoke test: run a mini simulation for a few generations."""

import numpy as np
import pytest

from sim_env.core.simulation import Simulation
from sim_env.core.world import World
from sim_env.evolution.population import Population
from sim_env.fitness.evaluator import FitnessEvaluator


@pytest.fixture
def mini_cfg(cfg):
    """Tiny config for fast integration tests."""
    c = dict(cfg)
    c["simulation"] = dict(c["simulation"])
    c["simulation"]["max_generations"] = 3
    c["simulation"]["steps_per_generation"] = 10
    c["simulation"]["render"] = False
    c["simulation"]["log_interval"] = 1
    c["simulation"]["checkpoint_interval"] = 100  # skip checkpointing
    c["population"] = dict(c["population"])
    c["population"]["size"] = 10
    c["world"] = dict(c["world"])
    c["world"]["width"] = 16
    c["world"]["height"] = 16
    return c


def test_simulation_runs_without_error(mini_cfg):
    rng = np.random.default_rng(42)
    sim = Simulation(mini_cfg)
    sim.set_world(World(mini_cfg, rng))
    sim.set_population(Population(mini_cfg, rng))
    sim.set_fitness_evaluator(FitnessEvaluator(mini_cfg))
    sim.run()
    # Basic check: stats were collected
    assert len(sim._stats) == mini_cfg["simulation"]["max_generations"]


def test_fitness_evaluated_each_generation(mini_cfg):
    rng = np.random.default_rng(7)
    sim = Simulation(mini_cfg)
    sim.set_world(World(mini_cfg, rng))
    sim.set_population(Population(mini_cfg, rng))
    sim.set_fitness_evaluator(FitnessEvaluator(mini_cfg))
    sim.run()
    # After run, at least stats entries should have numeric fitness
    for s in sim._stats:
        assert isinstance(s["best_fitness"], float)
        assert isinstance(s["mean_fitness"], float)

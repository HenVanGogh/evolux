"""Experiment runner with config merging and output organisation."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a named experiment")
    parser.add_argument("--config", required=True, help="Experiment YAML config (overrides default)")
    parser.add_argument("--base", default="config/default.yaml", help="Base config to merge into")
    parser.add_argument("--name", default=None, help="Run name (default: config filename stem)")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.base).read_text())
    exp_cfg = yaml.safe_load(Path(args.config).read_text())
    cfg = _deep_merge(base_cfg, exp_cfg)

    run_name = args.name or Path(args.config).stem
    cfg["output"]["dir"] = f"runs/{run_name}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so module path doesn't matter when running from project root
    import numpy as np
    from sim_env.core.simulation import Simulation
    from sim_env.core.world import World
    from sim_env.evolution.population import Population
    from sim_env.fitness.evaluator import FitnessEvaluator

    rng = np.random.default_rng(cfg["simulation"].get("seed", None))
    sim = Simulation(cfg)
    sim.set_world(World(cfg, rng))
    sim.set_population(Population(cfg, rng))
    sim.set_fitness_evaluator(FitnessEvaluator(cfg))
    sim.run()


if __name__ == "__main__":
    main()

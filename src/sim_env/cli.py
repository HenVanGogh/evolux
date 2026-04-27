"""Command-line interface for sim_env."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_simulation(cfg: dict):
    """Wire up and return a ready-to-run Simulation."""
    import numpy as np

    from sim_env.core.simulation import Simulation
    from sim_env.core.world import World
    from sim_env.evolution.population import Population
    from sim_env.fitness.evaluator import FitnessEvaluator

    rng = np.random.default_rng(cfg["simulation"].get("seed", None))
    sim = Simulation(cfg)
    world = World(cfg, rng)
    population = Population(cfg, rng)
    evaluator = FitnessEvaluator(cfg)

    sim.set_world(world)
    sim.set_population(population)
    sim.set_fitness_evaluator(evaluator)

    if cfg["simulation"].get("render"):
        from sim_env.visualization.renderer import Renderer
        renderer = Renderer(world.width, world.height, cfg["simulation"].get("render_fps", 30))
        sim.set_renderer(renderer)

    return sim


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sim = _build_simulation(cfg)
    sim.run()


def cmd_validate(args: argparse.Namespace) -> None:
    """Quick smoke-test: run 1 generation and print stats."""
    cfg = _load_config(args.config)
    cfg["simulation"]["max_generations"] = 1
    cfg["simulation"]["steps_per_generation"] = 10
    cfg["simulation"]["render"] = False
    logging.basicConfig(level=logging.DEBUG)
    sim = _build_simulation(cfg)
    sim.run()
    print("Validation passed.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="sim-env",
        description="Simulated Evolution Framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a full experiment")
    run_p.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    run_p.set_defaults(func=cmd_run)

    val_p = sub.add_parser("validate", help="Smoke-test a config (1 generation)")
    val_p.add_argument("--config", default="config/default.yaml")
    val_p.set_defaults(func=cmd_validate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

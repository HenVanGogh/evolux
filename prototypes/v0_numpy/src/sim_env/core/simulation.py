"""Main simulation loop orchestrating world, population, evolution, and fitness."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sim_env.core.registry import ComponentRegistry

if TYPE_CHECKING:
    from sim_env.core.world import World
    from sim_env.evolution.population import Population
    from sim_env.fitness.evaluator import FitnessEvaluator
    from sim_env.visualization.renderer import Renderer

logger = logging.getLogger(__name__)


class Simulation:
    """Top-level simulation controller.

    Responsibilities:
    - Advance the world step-by-step within a generation.
    - Trigger fitness evaluation at end-of-generation.
    - Delegate reproduction / selection to the evolution engine.
    - Optionally render each step.
    - Checkpoint and log statistics.
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config["simulation"].get("seed", None))
        self.generation: int = 0
        self.step: int = 0

        self.registry = ComponentRegistry()
        self._world: World | None = None
        self._population: Population | None = None
        self._fitness_evaluator: FitnessEvaluator | None = None
        self._renderer: Renderer | None = None

        self._output_dir = Path(config.get("output", {}).get("dir", "runs"))
        self._stats: list[dict] = []

    # ── Wiring ──────────────────────────────────────────────────────────────

    def set_world(self, world: "World") -> None:
        self._world = world

    def set_population(self, population: "Population") -> None:
        self._population = population

    def set_fitness_evaluator(self, evaluator: "FitnessEvaluator") -> None:
        self._fitness_evaluator = evaluator

    def set_renderer(self, renderer: "Renderer") -> None:
        self._renderer = renderer

    # ── Main entry point ────────────────────────────────────────────────────

    def run(self) -> None:
        """Run the full evolutionary experiment."""
        assert self._world is not None, "World not set"
        assert self._population is not None, "Population not set"
        assert self._fitness_evaluator is not None, "FitnessEvaluator not set"

        max_gen = self.cfg["simulation"]["max_generations"]
        steps_per_gen = self.cfg["simulation"]["steps_per_generation"]
        log_interval = self.cfg["simulation"].get("log_interval", 10)
        checkpoint_interval = self.cfg["simulation"].get("checkpoint_interval", 50)

        logger.info("Starting simulation: %d generations × %d steps", max_gen, steps_per_gen)
        t_start = time.perf_counter()

        for gen in range(max_gen):
            self.generation = gen
            self._run_generation(steps_per_gen)

            stats = self._collect_stats()
            self._stats.append(stats)

            if gen % log_interval == 0:
                elapsed = time.perf_counter() - t_start
                logger.info(
                    "Gen %4d | best_fit=%.4f | mean_fit=%.4f | species=%d | elapsed=%.1fs",
                    gen,
                    stats["best_fitness"],
                    stats["mean_fitness"],
                    stats.get("n_species", 1),
                    elapsed,
                )

            if gen % checkpoint_interval == 0:
                self._checkpoint(gen)

        logger.info("Simulation complete after %d generations.", max_gen)

    def _run_generation(self, steps: int) -> None:
        """Run one generation: reset world, step creatures, evaluate fitness."""
        assert self._world is not None
        assert self._population is not None

        self._world.reset_generation()
        self._population.place_into(self._world)

        for s in range(steps):
            self.step = s
            self._world.step(self.rng)
            self._population.step_all(self._world)

            if self._renderer is not None and self.cfg["simulation"].get("render"):
                self._renderer.render(self._world, self._population, self.generation, s)

        # Evaluate & evolve
        assert self._fitness_evaluator is not None
        self._fitness_evaluator.evaluate(self._population, self._world)
        self._population.evolve(self.rng)

    # ── Statistics ──────────────────────────────────────────────────────────

    def _collect_stats(self) -> dict:
        assert self._population is not None
        fits = [c.fitness for c in self._population.creatures]
        return {
            "generation": self.generation,
            "best_fitness": float(np.max(fits)) if fits else 0.0,
            "mean_fitness": float(np.mean(fits)) if fits else 0.0,
            "std_fitness": float(np.std(fits)) if fits else 0.0,
            "n_creatures": len(fits),
            "n_species": len(self._population.species) if hasattr(self._population, "species") else 1,
        }

    # ── Checkpointing ───────────────────────────────────────────────────────

    def _checkpoint(self, generation: int) -> None:
        import json

        self._output_dir.mkdir(parents=True, exist_ok=True)
        stats_path = self._output_dir / "stats.json"
        with stats_path.open("w") as f:
            json.dump(self._stats, f, indent=2)

        if self.cfg.get("output", {}).get("save_best_genome") and self._population:
            best = max(self._population.creatures, key=lambda c: c.fitness, default=None)
            if best is not None:
                import pickle

                genome_path = self._output_dir / f"best_genome_gen{generation:04d}.pkl"
                with genome_path.open("wb") as gf:
                    pickle.dump(best.genome, gf)

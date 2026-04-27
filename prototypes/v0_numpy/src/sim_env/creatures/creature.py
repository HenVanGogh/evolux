"""Creature: the agent that lives in the simulation world."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from sim_env.creatures.body import Body
from sim_env.creatures.sensors import SensorArray

if TYPE_CHECKING:
    from sim_env.brain.base_brain import BaseBrain
    from sim_env.core.world import World
    from sim_env.creatures.genome import Genome


class Creature:
    """Composite agent: genome + body + brain + sensors.

    The creature is intentionally a thin coordinator; all heavy logic lives in
    its components (Brain, Body, SensorArray).
    """

    def __init__(
        self,
        genome: "Genome",
        brain: "BaseBrain",
        body: Body,
        sensors: SensorArray,
    ) -> None:
        self.genome = genome
        self.brain = brain
        self.body = body
        self.sensors = sensors

        self.alive: bool = True
        self.fitness: float = 0.0

        # Per-generation behaviour log (reset each generation)
        self._action_history: list[int] = []

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, world: "World", cfg: dict) -> None:
        """One simulation step: sense → think → act → survive."""
        if not self.alive:
            return

        # 1. Build input vector
        obs = self.sensors.observe(self.body, world, cfg)

        # 2. Brain forward pass
        action_logits = self.brain.forward(obs)

        # 3. Decode action (argmax over movement outputs)
        action = int(np.argmax(action_logits))
        self._action_history.append(action)

        # 4. Move
        self.body.move(action, world)

        # 5. World physics + energy update
        alive = self.body.tick(world, cfg)
        if not alive:
            self.alive = False
            world.occupancy[self.body.y][self.body.x] = None

    def reset_generation(self) -> None:
        """Clear per-generation state; keep genome and learned weights."""
        self._action_history.clear()
        self.brain.reset_episode()

    # ── Reproduction ─────────────────────────────────────────────────────────

    def can_reproduce(self, cfg: dict) -> bool:
        return self.alive and self.body.can_reproduce(cfg)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.genome.genome_id

    @property
    def position(self) -> tuple[int, int]:
        return self.body.x, self.body.y

    @property
    def energy(self) -> float:
        return self.body.energy

    @property
    def age(self) -> int:
        return self.body.age

    def __repr__(self) -> str:
        return (
            f"Creature(id={self.id}, alive={self.alive}, "
            f"energy={self.body.energy:.1f}, fit={self.fitness:.3f})"
        )

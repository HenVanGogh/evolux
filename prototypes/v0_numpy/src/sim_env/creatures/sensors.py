"""Sensor array: constructs the brain's input vector from world observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.core.world import World
    from sim_env.creatures.body import Body


class SensorArray:
    """Builds the sensory input vector fed into the brain each step.

    Sensor channels (each normalised to roughly [-1, 1] or [0, 1]):
      [0]      bias              — always 1.0
      [1]      energy_norm       — own energy / 100
      [2]      age_norm          — own age / max_age
      [3]      direction_sin     — sin(direction * π/2)
      [4]      direction_cos     — cos(direction * π/2)
      [5..end] food_sense_flat   — food values in neighbourhood (flattened)
    """

    N_SCALAR = 5  # bias + energy + age + dir_sin + dir_cos

    def __init__(self, sense_radius: int = 2) -> None:
        self.sense_radius = sense_radius
        side = 2 * sense_radius + 1
        self.n_food_cells = side * side
        self.n_inputs = self.N_SCALAR + self.n_food_cells

    def observe(self, body: "Body", world: "World", cfg: dict) -> np.ndarray:
        max_age = cfg["creature"].get("max_age", 1000)
        max_food = cfg["environment"]["resources"].get("food_energy", 10.0)

        obs = np.empty(self.n_inputs, dtype=np.float32)
        obs[0] = 1.0
        obs[1] = body.energy / 100.0
        obs[2] = body.age / max_age
        obs[3] = np.sin(body.direction * np.pi / 2)
        obs[4] = np.cos(body.direction * np.pi / 2)

        food = world.food_sense(body.x, body.y, radius=self.sense_radius)
        obs[self.N_SCALAR :] = food / max(max_food, 1e-6)

        return obs

    def __repr__(self) -> str:
        return f"SensorArray(radius={self.sense_radius}, n_inputs={self.n_inputs})"

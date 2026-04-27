"""Matplotlib renderer for the simulation world (no pygame dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sim_env.core.world import World
    from sim_env.evolution.population import Population


class Renderer:
    """Simple matplotlib-based renderer.  Import is deferred so the package
    works without a display.
    """

    def __init__(self, world_w: int, world_h: int, fps: int = 30) -> None:
        import matplotlib.pyplot as plt

        self._fig, self._ax = plt.subplots(figsize=(8, 8))
        plt.ion()
        self._im = self._ax.imshow(
            np.zeros((world_h, world_w, 3), dtype=np.uint8),
            origin="lower",
            interpolation="nearest",
        )
        self._title = self._ax.set_title("")
        self._fps = fps
        self._fig.tight_layout()

    def render(
        self,
        world: "World",
        population: "Population",
        generation: int,
        step: int,
    ) -> None:
        import matplotlib.pyplot as plt

        h, w = world.food_grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Green channel: food density
        food_norm = (world.food_grid / (world.food_grid.max() + 1e-6) * 200).astype(np.uint8)
        rgb[:, :, 1] = food_norm

        # Red dots: hazards
        rgb[world.hazard_grid, 0] = 255

        # Blue dots: creatures
        for c in population.creatures:
            if c.alive:
                rgb[c.body.y, c.body.x] = [100, 100, 255]

        self._im.set_data(rgb)
        self._ax.set_title(f"Gen {generation} | Step {step}")
        self._fig.canvas.draw_idle()
        plt.pause(1.0 / max(self._fps, 1))

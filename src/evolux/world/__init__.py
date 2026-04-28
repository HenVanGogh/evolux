"""world — vectorised B-parallel world tensors implementing the World Protocol."""

from __future__ import annotations

from evolux.core.registry import Registry

WORLD_REGISTRY: Registry = Registry("world")

# Import submodules after registry creation so registration can reference it.
from evolux.world.grid import GridWorld  # noqa: E402

WORLD_REGISTRY.add("grid_v1", GridWorld)

__all__ = ["WORLD_REGISTRY", "GridWorld"]

"""world — vectorised B-parallel world tensors implementing the World Protocol."""

from __future__ import annotations

from evolux.core.registry import Registry

WORLD_REGISTRY: Registry = Registry("world")

__all__ = ["WORLD_REGISTRY"]

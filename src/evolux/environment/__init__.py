"""environment — task-level dynamics: spawn, terminate, reset, procedural generation."""

from __future__ import annotations

from evolux.core.registry import Registry

ENVIRONMENT_REGISTRY: Registry = Registry("environment")

__all__ = ["ENVIRONMENT_REGISTRY"]

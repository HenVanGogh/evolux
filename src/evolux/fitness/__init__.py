"""fitness — multi-objective fitness aggregation + behaviour descriptors for QD."""

from __future__ import annotations

from evolux.core.registry import Registry

OBJECTIVE_REGISTRY: Registry = Registry("objective")
DESCRIPTOR_REGISTRY: Registry = Registry("behaviour_descriptor")

__all__ = ["DESCRIPTOR_REGISTRY", "OBJECTIVE_REGISTRY"]

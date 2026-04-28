"""evolution — selection / reproduction algorithms.

Strategies
----------
- tournament : k-tournament selection + elitism (Phase 1)
- neat        : NeuroEvolution of Augmenting Topologies (Phase 3)
- map_elites  : Quality-Diversity grid (Phase 3)
- cma_es      : Covariance Matrix Adaptation Evolution Strategy (Phase 3)
- pbt         : Population-Based Training (Phase 3)
- novelty     : Novelty Search (Phase 3)
- directed    : User-driven directed-GA (Phase 3+)
"""

from __future__ import annotations

from evolux.core.registry import Registry

EVOLUTION_REGISTRY: Registry = Registry("evolution")

# Register Phase-1 implementations (imported after the registry is created).
from evolux.evolution.tournament import Tournament  # noqa: E402

EVOLUTION_REGISTRY.add("tournament", Tournament)

from evolux.evolution.population import Population  # noqa: E402

__all__ = [
    "EVOLUTION_REGISTRY",
    "Population",
    "Tournament",
]

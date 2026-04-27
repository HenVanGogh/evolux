"""evolution — selection / reproduction algorithms.

Strategies
----------
- neat       : NeuroEvolution of Augmenting Topologies
- map_elites : Quality-Diversity grid
- cma_es     : Covariance Matrix Adaptation Evolution Strategy
- pbt        : Population-Based Training
- novelty    : Novelty Search
- directed   : User-driven directed-GA (Phase 3+)
"""

from __future__ import annotations

from evolux.core.registry import Registry

EVOLUTION_REGISTRY: Registry = Registry("evolution")

__all__ = ["EVOLUTION_REGISTRY"]

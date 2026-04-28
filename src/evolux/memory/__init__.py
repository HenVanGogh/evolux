"""memory — hierarchical memory systems for the brain.

Tiers
-----
- working   : KV cache / sliding window
- episodic  : DNC-style differentiable key-value bank (per-episode)
- semantic  : retrieval-augmented vector store (cross-episode)
- hebbian   : slow associative weight matrix (cross-episode)
"""

from __future__ import annotations

from evolux.core.registry import Registry

MEMORY_REGISTRY: Registry = Registry("memory")

__all__ = ["MEMORY_REGISTRY", "WorkingMemory"]

# Import submodules last to trigger registration with MEMORY_REGISTRY.
from evolux.memory.working import WorkingMemory  # noqa: E402

"""perception — sensory encoders feeding the brain.

Submodules will host concrete encoders (vision CNN, proprioception MLP,
chemoreception, audio). Concrete classes register themselves with
``PERCEPTION_REGISTRY``.
"""

from __future__ import annotations

from evolux.core.registry import Registry

PERCEPTION_REGISTRY: Registry = Registry("perception")

__all__ = ["PERCEPTION_REGISTRY"]

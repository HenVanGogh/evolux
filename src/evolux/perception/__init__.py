"""perception — sensory encoders feeding the brain.

Submodules host concrete encoders (vision CNN, proprioception MLP,
chemoreception, audio). Concrete classes register themselves with
``PERCEPTION_REGISTRY`` via the ``@PERCEPTION_REGISTRY.register`` decorator.
"""

from __future__ import annotations

from evolux.core.registry import Registry

PERCEPTION_REGISTRY: Registry = Registry("perception")

# Submodule imports are deferred to avoid triggering heavy torch imports at
# package-import time.  They are placed *after* PERCEPTION_REGISTRY is defined
# so that the decorators in each submodule find the registry ready.
from evolux.perception.audio import AudioEncoder  # noqa: E402
from evolux.perception.chemo import ChemoEncoder  # noqa: E402
from evolux.perception.multimodal import ConcatEncoder  # noqa: E402
from evolux.perception.proprio import ProprioMLP  # noqa: E402
from evolux.perception.vision import VisionCNN  # noqa: E402

__all__ = [
    "PERCEPTION_REGISTRY",
    "AudioEncoder",
    "ChemoEncoder",
    "ConcatEncoder",
    "ProprioMLP",
    "VisionCNN",
]

"""brain — modern brain architectures.

Available architectures (registered)
------------------------------------
- transformer  : causal Transformer policy (Phase 1)
- ssm          : Mamba/S4 state-space model    (Phase 2)
- hybrid       : Transformer + SSM hybrid      (Phase 2)
- world_model  : Dreamer-V3-lite               (Phase 2)
- neuromod     : neuromodulator gating wrapper (Phase 2)
- meta         : MAML/PEARL adapter wrapper    (Phase 2)
"""

from __future__ import annotations

from evolux.core.registry import Registry

BRAIN_REGISTRY: Registry = Registry("brain")

__all__ = ["BRAIN_REGISTRY"]

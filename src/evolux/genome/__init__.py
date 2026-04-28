"""genome — encodings of brain + body.

Encodings to deliver
--------------------
- direct    : weight tensors stored verbatim (Phase 1)
- hyperneat : CPPN that generates connection weights from coordinates (Phase 3)
- indirect  : tiny diffusion model that decodes a latent → weight tensor (Phase 3)
"""

from __future__ import annotations

from evolux.core.registry import Registry

GENOME_REGISTRY: Registry = Registry("genome")
GENOME_OPERATOR_REGISTRY: Registry = Registry("genome_operator")

# Register Phase-1 implementations (import after registries are created).
from evolux.genome.direct import DirectGenome, DirectOperator  # noqa: E402

GENOME_REGISTRY.add("direct", DirectGenome)
GENOME_OPERATOR_REGISTRY.add("direct", DirectOperator)

__all__ = [
    "GENOME_OPERATOR_REGISTRY",
    "GENOME_REGISTRY",
    "DirectGenome",
    "DirectOperator",
]

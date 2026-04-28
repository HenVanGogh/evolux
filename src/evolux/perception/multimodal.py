"""Multimodal encoder — concatenates outputs from multiple named encoders."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evolux.perception import PERCEPTION_REGISTRY


@PERCEPTION_REGISTRY.register("concat")
class ConcatEncoder(nn.Module):
    """Concatenates the outputs of a collection of named encoders.

    Each encoder receives its own slice of the observation dict and produces
    a ``(B, encoder.output_dim)`` vector.  The outputs are concatenated along
    the feature dimension so the resulting tensor has shape
    ``(B, sum_of_output_dims)``.

    Args:
        encoders: Mapping from modality name to a compatible encoder module.
            Every value must expose an ``output_dim: int`` attribute.

    Input:  ``dict[str, Tensor]`` — one entry per encoder, keyed by name.
    Output: ``(B, output_dim)``
    """

    output_dim: int

    def __init__(self, encoders: dict[str, nn.Module]) -> None:
        super().__init__()
        self._encoders = nn.ModuleDict(encoders)
        self.output_dim: int = sum(int(enc.output_dim) for enc in encoders.values())

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        """Map ``{name: (B, ...)}`` → ``(B, output_dim)``."""
        parts = [self._encoders[name](x[name]) for name in self._encoders]
        return torch.cat(parts, dim=-1)

"""Proprioception encoder — two-layer MLP for (B, P) vectors."""

from __future__ import annotations

from torch import Tensor, nn

from evolux.perception import PERCEPTION_REGISTRY


@PERCEPTION_REGISTRY.register("proprio_mlp_v1")
class ProprioMLP(nn.Module):
    """Two-layer MLP encoder for proprioceptive observations.

    Args:
        in_dim: Dimensionality of the proprioceptive input vector (P).
        output_dim: Width of the output feature vector.
        hidden_dim: Width of the hidden layer (default 64).

    Input shape:  ``(B, P)``
    Output shape: ``(B, output_dim)``
    """

    output_dim: int

    def __init__(
        self,
        in_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        self._net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, P)`` → ``(B, output_dim)``."""
        return self._net(x)  # type: ignore[no-any-return]

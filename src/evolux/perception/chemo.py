"""Chemo encoder — radial-basis-function encoder for chemical-gradient sensing."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evolux.perception import PERCEPTION_REGISTRY


@PERCEPTION_REGISTRY.register("chemo_rbf_v1")
class ChemoEncoder(nn.Module):
    """RBF encoder for chemical-gradient observations.

    A bank of learnable RBF centres projects the input into an ``n_rbf``-dim
    activation vector, which is then mapped to ``output_dim`` via a small
    two-layer MLP.

    Args:
        n_channels: Number of chemical input channels (C).
        output_dim: Width of the output feature vector.
        n_rbf: Number of radial-basis-function centres (default 32).
        hidden_dim: Width of the MLP hidden layer (default 64).

    Input shape:  ``(B, n_channels)``
    Output shape: ``(B, output_dim)``
    """

    output_dim: int

    def __init__(
        self,
        n_channels: int,
        output_dim: int,
        n_rbf: int = 32,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # Learnable RBF centres: (n_rbf, n_channels)
        self._centres = nn.Parameter(torch.empty(n_rbf, n_channels))
        nn.init.normal_(self._centres)
        # Learnable log-bandwidth per centre (scalar per centre)
        self._log_gamma = nn.Parameter(torch.zeros(n_rbf))

        self._mlp = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, n_channels)`` → ``(B, output_dim)``."""
        # x: (B, C), centres: (n_rbf, C)
        # Compute squared Euclidean distances: (B, n_rbf)
        diff = x.unsqueeze(1) - self._centres.unsqueeze(0)  # (B, n_rbf, C)
        dist_sq = (diff * diff).sum(dim=-1)  # (B, n_rbf)
        gamma = self._log_gamma.exp()  # (n_rbf,)
        rbf_out = torch.exp(-gamma * dist_sq)  # (B, n_rbf)
        return self._mlp(rbf_out)  # type: ignore[no-any-return]

"""Vision encoder — NatureCNN-lite for (B, C, H, W) images."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from evolux.perception import PERCEPTION_REGISTRY


@PERCEPTION_REGISTRY.register("vision_cnn_v1")
class VisionCNN(nn.Module):
    """NatureCNN-lite: three convolutional layers (32→64→64) + linear head.

    The flatten size is computed once at construction via a dummy forward pass,
    making the architecture agnostic to the input spatial resolution.

    Args:
        in_channels: Number of input image channels (C).
        output_dim: Width of the output feature vector.
        img_size: ``(H, W)`` of the expected input images; used only to
            pre-compute the post-conv flatten dimension.

    Input shape:  ``(B, C, H, W)``
    Output shape: ``(B, output_dim)``
    """

    output_dim: int

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        img_size: tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size[0], img_size[1])
            flat_dim = int(self._conv(dummy).flatten(1).shape[1])

        self._head = nn.Linear(flat_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, C, H, W)`` → ``(B, output_dim)``."""
        return self._head(self._conv(x).flatten(1))  # type: ignore[no-any-return]

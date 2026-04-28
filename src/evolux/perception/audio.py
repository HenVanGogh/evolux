"""Audio encoder — 1-D conv stack for (B, C, T) audio windows."""

from __future__ import annotations

from torch import Tensor, nn

from evolux.perception import PERCEPTION_REGISTRY


@PERCEPTION_REGISTRY.register("audio_conv_v1")
class AudioEncoder(nn.Module):
    """Three-layer 1-D convolutional encoder for audio windows.

    Architecture: Conv1d(32) → GELU → Conv1d(64) → GELU → Conv1d(128) → GELU
                  → AdaptiveAvgPool1d(1) → Linear → output.

    Args:
        in_channels: Number of audio input channels (C).
        output_dim: Width of the output feature vector.

    Input shape:  ``(B, C, T)``
    Output shape: ``(B, output_dim)``
    """

    output_dim: int

    # Intermediate channel widths for the three conv layers.
    _CONV_CHANNELS: tuple[int, int, int] = (32, 64, 128)

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        c1, c2, c3 = self._CONV_CHANNELS

        self._conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self._head = nn.Linear(c3, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``(B, C, T)`` → ``(B, output_dim)``."""
        features = self._conv(x).squeeze(-1)  # (B, 128)
        return self._head(features)  # type: ignore[no-any-return]

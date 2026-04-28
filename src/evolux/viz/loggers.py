"""Logging backends: JsonlLogger, TensorboardLogger, WandbLogger.

All three implement the ``StatsLogger`` protocol from ``evolux.core.protocols``.
Optional backends (tensorboard, wandb) fall back gracefully to ``JsonlLogger``
if the dependency is not installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from torch import Tensor

log = logging.getLogger(__name__)

__all__ = ["JsonlLogger", "TensorboardLogger", "WandbLogger"]


class JsonlLogger:
    """Append-only JSONL writer for scalars, histograms and image metadata.

    Always available — no optional dependencies. One JSON object per line.

    Args:
        path: File path for the ``.jsonl`` output.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("a", encoding="utf-8")
        self._closed = False

    # ── internal ────────────────────────────────────────────────────────────

    def _write(self, record: dict[str, Any]) -> None:
        if not self._closed:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()

    # ── StatsLogger protocol ─────────────────────────────────────────────────

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Append a scalar measurement."""
        self._write({"type": "scalar", "key": key, "value": value, "step": step})

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        """Append a flat dict of scalars."""
        self._write({"type": "dict", "data": data, "step": step})

    def log_image(self, key: str, image: Tensor, step: int) -> None:
        """Append image metadata (pixel data is not stored in JSONL)."""
        self._write(
            {
                "type": "image",
                "key": key,
                "shape": list(image.shape),
                "step": step,
            }
        )

    def log_hist(self, tag: str, values: Tensor, step: int) -> None:
        """Append summary statistics for a histogram.

        Args:
            tag: Metric tag.
            values: 1-D (or any shape) tensor of raw values.
            step: Global step counter.
        """
        flat = values.detach().float().flatten()
        self._write(
            {
                "type": "hist",
                "tag": tag,
                "min": flat.min().item(),
                "max": flat.max().item(),
                "mean": flat.mean().item(),
                "std": flat.std().item(),
                "step": step,
            }
        )

    def close(self) -> None:
        """Flush and close the underlying file handle (idempotent)."""
        if not self._closed:
            self._fh.flush()
            self._fh.close()
            self._closed = True


class TensorboardLogger:
    """TensorBoard SummaryWriter wrapper.

    Falls back to :class:`JsonlLogger` (with a one-time warning) when
    ``tensorboard`` is not installed.

    Args:
        log_dir: Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str | Path) -> None:
        self._fallback: JsonlLogger | None = None
        self._writer: Any = None
        log_dir = Path(log_dir)
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]

            self._writer = SummaryWriter(str(log_dir))
        except ImportError:
            log.warning(
                "tensorboard is not installed; falling back to JsonlLogger. "
                "Install it with: pip install tensorboard"
            )
            self._fallback = JsonlLogger(log_dir / "events.jsonl")

    # ── StatsLogger protocol ─────────────────────────────────────────────────

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._fallback is not None:
            self._fallback.log_scalar(key, value, step)
        else:
            self._writer.add_scalar(key, value, step)

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        """Log a flat dict of scalars."""
        if self._fallback is not None:
            self._fallback.log_dict(data, step)
        else:
            for k, v in data.items():
                self._writer.add_scalar(k, v, step)

    def log_image(self, key: str, image: Tensor, step: int) -> None:
        """Log an image tensor (CHW or HW)."""
        if self._fallback is not None:
            self._fallback.log_image(key, image, step)
        else:
            self._writer.add_image(key, image, step)

    def log_hist(self, tag: str, values: Tensor, step: int) -> None:
        """Log a histogram of values."""
        if self._fallback is not None:
            self._fallback.log_hist(tag, values, step)
        else:
            self._writer.add_histogram(tag, values, step)

    def close(self) -> None:
        """Close the writer (idempotent)."""
        if self._fallback is not None:
            self._fallback.close()
        elif self._writer is not None:
            self._writer.close()
            self._writer = None


class WandbLogger:
    """Weights & Biases logger.

    Falls back to :class:`JsonlLogger` (with a one-time warning) when
    ``wandb`` is not installed.

    Args:
        log_dir: Fallback JSONL directory when wandb is unavailable.
        project: W&B project name (used only when wandb is available).
        name: W&B run name (used only when wandb is available).
        config: Config dict passed to ``wandb.init`` (optional).
    """

    def __init__(
        self,
        log_dir: str | Path,
        project: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._fallback: JsonlLogger | None = None
        self._wandb: Any = None
        log_dir = Path(log_dir)
        try:
            import wandb  # type: ignore[import-untyped]

            wandb.init(project=project, name=name, config=config or {})
            self._wandb = wandb
        except ImportError:
            log.warning(
                "wandb is not installed; falling back to JsonlLogger. "
                "Install it with: pip install wandb"
            )
            self._fallback = JsonlLogger(log_dir / "events.jsonl")

    # ── StatsLogger protocol ─────────────────────────────────────────────────

    def log_scalar(self, key: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self._fallback is not None:
            self._fallback.log_scalar(key, value, step)
        else:
            self._wandb.log({key: value}, step=step)

    def log_dict(self, data: dict[str, Any], step: int) -> None:
        """Log a flat dict of scalars."""
        if self._fallback is not None:
            self._fallback.log_dict(data, step)
        else:
            self._wandb.log(data, step=step)

    def log_image(self, key: str, image: Tensor, step: int) -> None:
        """Log an image tensor."""
        if self._fallback is not None:
            self._fallback.log_image(key, image, step)
        else:
            self._wandb.log({key: self._wandb.Image(image)}, step=step)

    def log_hist(self, tag: str, values: Tensor, step: int) -> None:
        """Log a histogram of values."""
        if self._fallback is not None:
            self._fallback.log_hist(tag, values, step)
        else:
            self._wandb.log({tag: self._wandb.Histogram(values.cpu().numpy())}, step=step)

    def close(self) -> None:
        """Finish the W&B run (idempotent)."""
        if self._fallback is not None:
            self._fallback.close()
        elif self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

"""Device management — picks CPU/GPU and exposes a single source of truth."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec.

    Parameters
    ----------
    spec:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``, or ``"mps"``.
        ``"auto"`` picks ``cuda`` if available else ``cpu``.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


class DeviceManager:
    """Holds the single global device + amp configuration for a run."""

    def __init__(self, device_spec: str = "auto", amp: bool = False) -> None:
        self.device: torch.device = get_device(device_spec)
        self.amp_enabled: bool = amp and self.device.type == "cuda"
        logger.info("DeviceManager: device=%s amp=%s", self.device, self.amp_enabled)

    def autocast(self) -> torch.amp.autocast_mode.autocast:
        if self.amp_enabled:
            return torch.amp.autocast(self.device.type, dtype=torch.float16)
        return _NullCtx()

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class _NullCtx:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None

"""Acceptance tests for evolux.core.device."""

from __future__ import annotations

import torch

from evolux.core.device import DeviceManager, get_device


def test_auto_returns_cpu_when_no_gpu() -> None:
    """On CPU-only machines ``get_device("auto")`` must return a CPU device."""
    d = get_device("auto")
    # We are in a CPU-only CI environment; cuda/mps are not available.
    if not torch.cuda.is_available() and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        assert d.type == "cpu"
    else:
        assert d.type in {"cpu", "cuda", "mps"}


def test_explicit_cpu() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_explicit_spec_passthrough() -> None:
    """Any explicit spec should be forwarded to torch.device."""
    d = get_device("cpu")
    assert d == torch.device("cpu")


def test_device_manager_defaults_to_cpu() -> None:
    dm = DeviceManager("cpu")
    assert dm.device == torch.device("cpu")
    assert dm.amp_enabled is False


def test_device_manager_amp_disabled_on_cpu() -> None:
    """AMP must be disabled when device is CPU, even if amp=True is requested."""
    dm = DeviceManager("cpu", amp=True)
    assert dm.amp_enabled is False


def test_device_manager_autocast_is_context_manager() -> None:
    dm = DeviceManager("cpu")
    ctx = dm.autocast()
    with ctx:
        pass  # must not raise


def test_device_manager_synchronize_noop_on_cpu() -> None:
    dm = DeviceManager("cpu")
    dm.synchronize()  # must not raise

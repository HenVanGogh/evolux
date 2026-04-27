"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
import torch

from evolux.core.device import get_device
from evolux.core.rng import RNG


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def rng() -> RNG:
    return RNG(seed=0)


@pytest.fixture
def auto_device() -> torch.device:
    return get_device("auto")

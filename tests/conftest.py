"""Shared fixtures for all tests."""

import numpy as np
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def cfg():
    base = Path(__file__).parent.parent / "config" / "default.yaml"
    return yaml.safe_load(base.read_text())


@pytest.fixture
def rng():
    return np.random.default_rng(0)

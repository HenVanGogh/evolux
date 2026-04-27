"""Abstract base environment interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEnvironment(ABC):
    """Protocol for pluggable environments.

    An environment wraps the World and adds domain-specific logic
    (e.g., seasonal cycles, predators, resource dynamics beyond simple respawn).
    """

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        """Reset to initial state for a new generation."""

    @abstractmethod
    def step(self, rng: np.random.Generator) -> None:
        """Advance environment dynamics by one simulation tick."""

    @abstractmethod
    def observe(self) -> dict:
        """Return a dict of observable quantities for logging."""

"""Abstract base class for all brain implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBrain(ABC):
    """Contract that every brain must fulfil.

    A brain consumes a flat observation vector and produces a flat output
    vector (action logits or raw values).

    It is also responsible for managing its own memory state: calling
    ``reset_episode()`` should wipe all transient activation/memory but
    leave slow weights (Hebbian traces) intact.
    """

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute output activations from inputs.

        Parameters
        ----------
        inputs:
            1-D float array of length ``n_inputs``.

        Returns
        -------
        outputs:
            1-D float array of length ``n_outputs``.
        """

    @abstractmethod
    def reset_episode(self) -> None:
        """Reset transient state (working memory, recurrent activations).
        Called at the start of each new generation / episode.
        """

    @property
    @abstractmethod
    def n_inputs(self) -> int: ...

    @property
    @abstractmethod
    def n_outputs(self) -> int: ...

    @abstractmethod
    def n_parameters(self) -> int:
        """Total number of learnable / evolvable parameters."""

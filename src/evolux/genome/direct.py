"""Direct (flat weight-vector) genome encoding.

A ``DirectGenome`` stores the neural-network parameters as a single contiguous
float32 tensor.  Genetic operators add Gaussian noise (mutation), blend two
vectors (crossover), and compute L2 distance (speciation).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from evolux.core.protocols import Brain, BrainFactory, Morphology
from evolux.core.types import ActionSpec, ObsSpec


class DirectGenome:
    """Flat float32 vector representing a neural network's parameter set.

    Attributes
    ----------
    encoding:
        Always ``"direct"``; identifies the genome flavour to registries.
    params:
        1-D float32 tensor of length ``n_params``.
    """

    encoding: str = "direct"

    def __init__(self, params: Tensor) -> None:
        self.params: Tensor = params.float().clone().detach()

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int:
        """Number of scalar parameters in this genome."""
        return int(self.params.numel())

    # ── Genome Protocol ─────────────────────────────────────────────────────

    def decode_brain(self, brain_factory: BrainFactory) -> Brain:
        """Decode this genome into a Brain via *brain_factory*.

        The factory must expose:

        * ``parameter_shapes: dict[str, tuple[int, ...]]`` — ordered mapping of
          parameter-block names to shapes; the flat ``params`` vector is sliced
          and reshaped accordingly.
        * (optional) ``obs_spec: ObsSpec`` and ``action_spec: ActionSpec`` —
          used when calling ``brain_factory.build``; fall back to empty specs if
          absent.

        The extracted weight dict is forwarded in ``hparams["weights"]``.
        """
        hparams: dict[str, Any] = {}

        if hasattr(brain_factory, "parameter_shapes"):
            weights: dict[str, Tensor] = {}
            offset = 0
            for name, shape in brain_factory.parameter_shapes.items():  # type: ignore[union-attr]
                n = math.prod(shape)
                weights[name] = self.params[offset : offset + n].reshape(shape)
                offset += n
            hparams["weights"] = weights

        obs_spec: ObsSpec = getattr(brain_factory, "obs_spec", ObsSpec(fields={}))
        action_spec: ActionSpec = getattr(
            brain_factory, "action_spec", ActionSpec(discrete=True, n=1)
        )
        return brain_factory.build(obs_spec, action_spec, hparams)

    def decode_morphology(self) -> Morphology:
        """Decode this genome into a Morphology.

        Notes
        -----
        Phase-1 placeholder: raises ``NotImplementedError``.  When the
        ``morphology`` module lands (Layer 2) this method will be wired up via
        a factory / registry.
        """
        raise NotImplementedError(
            "decode_morphology is not available in Phase 1. "
            "Wire a MorphologyFactory once the morphology module is implemented."
        )

    def serialize(self) -> bytes:
        """Serialise to a raw float32 byte string (little-endian)."""
        arr: np.ndarray = self.params.detach().cpu().numpy().astype(np.float32)
        return arr.tobytes()

    @classmethod
    def deserialize(cls, blob: bytes) -> DirectGenome:
        """Reconstruct a ``DirectGenome`` from bytes produced by :meth:`serialize`."""
        arr = np.frombuffer(blob, dtype=np.float32).copy()
        return cls(torch.from_numpy(arr))

    # ── Dunder helpers ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_params

    def __repr__(self) -> str:
        return f"DirectGenome(n_params={self.n_params})"


class DirectOperator:
    """Genetic operators for :class:`DirectGenome`.

    Parameters
    ----------
    sigma:
        Standard deviation of Gaussian mutation noise.
    """

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma: float = sigma

    # ── GenomeOperator Protocol ──────────────────────────────────────────────

    def mutate(self, g: DirectGenome, rng: torch.Generator) -> DirectGenome:
        """Return a new genome with Gaussian noise added to every parameter.

        The input genome is **never** mutated in-place.
        """
        noise = torch.empty_like(g.params)
        noise.normal_(mean=0.0, std=self.sigma, generator=rng)
        return DirectGenome(g.params + noise)

    def crossover(self, a: DirectGenome, b: DirectGenome, rng: torch.Generator) -> DirectGenome:
        """Uniform crossover: each parameter taken from *a* or *b* with p=0.5.

        The input genomes are **never** mutated in-place.
        """
        mask: Tensor = torch.rand(a.params.shape, generator=rng) < 0.5
        child_params: Tensor = torch.where(mask, a.params, b.params)
        return DirectGenome(child_params)

    def distance(self, a: DirectGenome, b: DirectGenome) -> float:
        """L2 (Euclidean) distance between two genomes (for speciation)."""
        return float(torch.linalg.norm(a.params - b.params).item())

    def __repr__(self) -> str:
        return f"DirectOperator(sigma={self.sigma})"

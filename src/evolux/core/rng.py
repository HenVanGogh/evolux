"""Random-number management.

A single seed is split into per-module sub-generators (à la JAX). This makes
runs reproducible *and* lets unrelated modules share a seed without correlated
streams.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def _hash_to_int(name: str) -> int:
    """Deterministic 63-bit hash of a string (cross-process stable)."""
    h = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big") & ((1 << 63) - 1)


class RNG:
    """Hierarchical reproducible RNG.

    Usage::

        root = RNG(seed=42)
        brain_rng  = root.split("brain")          # torch.Generator
        evo_rng    = root.split("evolution")
        np_rng     = root.numpy("worldgen")        # np.random.Generator
        children   = root.fork(4)                  # list of 4 child RNGs

    Notes
    -----
    Design rationale: ``docs/ARCHITECTURE.md`` §7 — *Determinism is opt-in but
    supported*. A single global seed is fanned out via a collision-resistant
    hash so each named sub-stream is statistically independent. This is the
    same strategy as JAX's ``jax.random.split``, adapted for PyTorch generators.
    All randomness in the framework must go through this class so that any run
    is exactly reproducible by re-using the same seed.
    """

    def __init__(self, seed: int, device: torch.device | str = "cpu") -> None:
        self.seed: int = int(seed)
        self.device: torch.device = torch.device(device)

    def split(self, name: str) -> torch.Generator:
        """Return a child torch.Generator deterministically derived from name."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed ^ _hash_to_int(name))
        return gen

    def numpy(self, name: str) -> np.random.Generator:
        """Return a child numpy Generator deterministically derived from name."""
        return np.random.default_rng(self.seed ^ _hash_to_int(name))

    def child(self, name: str) -> RNG:
        """Return a sub-RNG (same API, different seed)."""
        return RNG(self.seed ^ _hash_to_int(name), device=self.device)

    def fork(self, n: int) -> list[RNG]:
        """Return ``n`` independent child :class:`RNG` instances.

        The children are deterministic: calling ``fork(n)`` twice on the same
        parent always yields the same sequence of child seeds.  Children are
        indexed ``"fork_0"``, ``"fork_1"``, … so they have independent streams.

        Parameters
        ----------
        n:
            Number of child RNGs to create.  Must be ≥ 1.
        """
        if n < 1:
            raise ValueError(f"n must be ≥ 1, got {n}")
        return [self.child(f"fork_{i}") for i in range(n)]

    def __repr__(self) -> str:
        return f"RNG(seed={self.seed}, device={self.device})"

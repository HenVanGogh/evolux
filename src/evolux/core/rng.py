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
        return np.random.default_rng(self.seed ^ _hash_to_int(name))

    def child(self, name: str) -> RNG:
        """Return a sub-RNG (same API, different seed)."""
        return RNG(self.seed ^ _hash_to_int(name), device=self.device)

    def __repr__(self) -> str:
        return f"RNG(seed={self.seed}, device={self.device})"

"""Acceptance tests for evolux.core.rng — RNG.split and RNG.fork."""

from __future__ import annotations

import pytest
import torch

from evolux.core.rng import RNG


def test_split_deterministic_same_name() -> None:
    """Same seed + same name must produce identical generator state."""
    a = RNG(42).split("brain")
    b = RNG(42).split("brain")
    assert torch.rand(8, generator=a).tolist() == torch.rand(8, generator=b).tolist()


def test_split_different_names_independent() -> None:
    """Different names must yield statistically independent streams."""
    root = RNG(42)
    g1 = root.split("alpha")
    g2 = root.split("beta")
    s1 = torch.rand(16, generator=g1)
    s2 = torch.rand(16, generator=g2)
    assert not torch.allclose(s1, s2)


def test_split_different_seeds_independent() -> None:
    """Different seeds must yield different streams for the same name."""
    g1 = RNG(1).split("x")
    g2 = RNG(2).split("x")
    s1 = torch.rand(8, generator=g1)
    s2 = torch.rand(8, generator=g2)
    assert not torch.allclose(s1, s2)


def test_numpy_deterministic() -> None:
    """numpy() stream is also deterministic given seed+name."""

    a = RNG(99).numpy("world")
    b = RNG(99).numpy("world")
    assert a.random() == b.random()


def test_child_is_rng() -> None:
    child = RNG(10).child("sub")
    assert isinstance(child, RNG)
    # Child seed must differ from parent
    assert child.seed != 10


def test_fork_returns_n_rngs() -> None:
    children = RNG(0).fork(4)
    assert len(children) == 4
    assert all(isinstance(c, RNG) for c in children)


def test_fork_deterministic() -> None:
    """fork(n) is deterministic: same parent seed → same child seeds."""
    a = RNG(7).fork(3)
    b = RNG(7).fork(3)
    for ca, cb in zip(a, b, strict=True):
        assert ca.seed == cb.seed


def test_fork_children_independent() -> None:
    """fork(n) children must produce different random streams."""
    children = RNG(42).fork(4)
    samples = [torch.rand(8, generator=c.split("test")).tolist() for c in children]
    # All four streams must differ from each other
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            assert samples[i] != samples[j], f"child {i} and {j} are identical"


def test_fork_invalid_n() -> None:
    with pytest.raises(ValueError):
        RNG(0).fork(0)


def test_repr() -> None:
    r = repr(RNG(42))
    assert "42" in r

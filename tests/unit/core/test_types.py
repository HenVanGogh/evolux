"""Tests for shared types — Trajectory and spec dataclasses."""

from __future__ import annotations

import pytest
import torch

from evolux.core.types import Trajectory


def _make_trajectory(B: int = 2, T: int = 4) -> Trajectory:
    """Helper: construct a small Trajectory of shape (B, T, ...)."""
    return Trajectory(
        obs={"x": torch.ones(B, T, 3)},
        actions=torch.zeros(B, T, 2),
        rewards=torch.zeros(B, T),
        dones=torch.zeros(B, T, dtype=torch.bool),
        length=torch.full((B,), T, dtype=torch.long),
        aux={"step": torch.arange(T).unsqueeze(0).expand(B, -1).float()},
    )


def test_trajectory_batch_size_and_horizon() -> None:
    t = _make_trajectory(B=3, T=5)
    assert t.batch_size == 3
    assert t.horizon == 5


def test_trajectory_to_device() -> None:
    t = _make_trajectory()
    t2 = t.to(torch.device("cpu"))
    assert t2.actions.device == torch.device("cpu")


# ── Trajectory.concat ────────────────────────────────────────────────────────


def test_concat_time_dim() -> None:
    """Concatenating two trajectories should double the time dimension."""
    a = _make_trajectory(B=2, T=3)
    b = _make_trajectory(B=2, T=5)
    merged = Trajectory.concat([a, b])
    assert merged.horizon == 8
    assert merged.batch_size == 2


def test_concat_length_is_sum() -> None:
    """Lengths should be element-wise summed."""
    a = _make_trajectory(B=2, T=3)
    b = _make_trajectory(B=2, T=5)
    merged = Trajectory.concat([a, b])
    assert merged.length.tolist() == [8, 8]


def test_concat_obs_keys_preserved() -> None:
    a = _make_trajectory(B=2, T=4)
    b = _make_trajectory(B=2, T=4)
    merged = Trajectory.concat([a, b])
    assert set(merged.obs.keys()) == {"x"}


def test_concat_aux_keys_preserved() -> None:
    a = _make_trajectory(B=2, T=4)
    b = _make_trajectory(B=2, T=4)
    merged = Trajectory.concat([a, b])
    assert "step" in merged.aux


def test_concat_single_element() -> None:
    """concat of a single trajectory should be equivalent to a copy."""
    t = _make_trajectory(B=2, T=6)
    merged = Trajectory.concat([t])
    assert merged.horizon == 6
    assert torch.equal(merged.actions, t.actions)


def test_concat_deterministic() -> None:
    """concat is deterministic — same inputs give same outputs."""
    a = _make_trajectory(B=2, T=3)
    b = _make_trajectory(B=2, T=3)
    m1 = Trajectory.concat([a, b])
    m2 = Trajectory.concat([a, b])
    assert torch.equal(m1.actions, m2.actions)


def test_concat_empty_raises() -> None:
    with pytest.raises(ValueError):
        Trajectory.concat([])


def test_concat_three_trajectories() -> None:
    parts = [_make_trajectory(B=2, T=2) for _ in range(3)]
    merged = Trajectory.concat(parts)
    assert merged.horizon == 6

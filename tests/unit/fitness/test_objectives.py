"""Tests for evolux.fitness.objectives (Phase 1)."""

from __future__ import annotations

import pytest
import torch

from evolux.core.protocols import Objective
from evolux.core.types import Trajectory
from evolux.fitness import OBJECTIVE_REGISTRY
from evolux.fitness.objectives import DistanceTravelled, FoodCollected, SurvivalTime

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_trajectory(
    B: int = 4,
    T: int = 8,
    rewards: torch.Tensor | None = None,
    dones: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> Trajectory:
    """Build a minimal Trajectory for testing."""
    if rewards is None:
        rewards = torch.zeros(B, T)
    if dones is None:
        dones = torch.zeros(B, T, dtype=torch.bool)
    aux = {}
    if positions is not None:
        aux["positions"] = positions
    return Trajectory(
        obs={},
        actions=torch.zeros(B, T, 2),
        rewards=rewards,
        dones=dones,
        length=torch.full((B,), T, dtype=torch.long),
        aux=aux,
    )


class _FakeWorld:
    """Minimal stub satisfying the World Protocol interface."""

    batch_size: int = 4
    device: torch.device = torch.device("cpu")

    def reset(self, mask=None):
        pass

    def observe(self):
        return {}

    def step(self, action):
        return torch.zeros(4), torch.zeros(4, dtype=torch.bool), {}


_WORLD = _FakeWorld()


# ── Registry ────────────────────────────────────────────────────────────────


def test_objectives_registered() -> None:
    assert "survival_time" in OBJECTIVE_REGISTRY
    assert "food_collected" in OBJECTIVE_REGISTRY
    assert "distance_travelled" in OBJECTIVE_REGISTRY


# ── Protocol conformance ────────────────────────────────────────────────────


@pytest.mark.parametrize("cls", [SurvivalTime, FoodCollected, DistanceTravelled])
def test_isinstance_objective_protocol(cls) -> None:
    assert isinstance(cls(), Objective)


# ── SurvivalTime ────────────────────────────────────────────────────────────


def test_survival_time_shape() -> None:
    B, T = 6, 10
    traj = _make_trajectory(B=B, T=T)
    score = SurvivalTime().evaluate(traj, _WORLD)
    assert score.shape == (B,), f"expected ({B},), got {score.shape}"


def test_survival_time_all_alive() -> None:
    B, T = 3, 5
    traj = _make_trajectory(B=B, T=T, dones=torch.zeros(B, T, dtype=torch.bool))
    score = SurvivalTime().evaluate(traj, _WORLD)
    assert torch.all(score == T)


def test_survival_time_all_dead() -> None:
    B, T = 2, 4
    traj = _make_trajectory(B=B, T=T, dones=torch.ones(B, T, dtype=torch.bool))
    score = SurvivalTime().evaluate(traj, _WORLD)
    assert torch.all(score == 0)


def test_survival_time_partial_done() -> None:
    B, T = 2, 4
    dones = torch.tensor([[False, False, True, True], [False, True, True, True]])
    traj = _make_trajectory(B=B, T=T, dones=dones)
    score = SurvivalTime().evaluate(traj, _WORLD)
    assert score[0].item() == 2
    assert score[1].item() == 1


def test_survival_time_weight_attribute() -> None:
    obj = SurvivalTime(weight=2.5)
    assert obj.weight == pytest.approx(2.5)
    assert obj.name == "survival_time"
    assert obj.higher_is_better is True


# ── FoodCollected ───────────────────────────────────────────────────────────


def test_food_collected_shape() -> None:
    B, T = 5, 12
    traj = _make_trajectory(B=B, T=T)
    score = FoodCollected().evaluate(traj, _WORLD)
    assert score.shape == (B,)


def test_food_collected_only_positive_rewards() -> None:
    B, T = 2, 4
    rewards = torch.tensor([[-1.0, 2.0, -0.5, 3.0], [0.0, 0.0, 0.0, 0.0]])
    traj = _make_trajectory(B=B, T=T, rewards=rewards)
    score = FoodCollected().evaluate(traj, _WORLD)
    assert score[0].item() == pytest.approx(5.0)
    assert score[1].item() == pytest.approx(0.0)


def test_food_collected_all_negative_rewards() -> None:
    B, T = 3, 3
    rewards = torch.full((B, T), -1.0)
    traj = _make_trajectory(B=B, T=T, rewards=rewards)
    score = FoodCollected().evaluate(traj, _WORLD)
    assert torch.all(score == 0.0)


def test_food_collected_weight_attribute() -> None:
    obj = FoodCollected(weight=0.5)
    assert obj.weight == pytest.approx(0.5)
    assert obj.name == "food_collected"
    assert obj.higher_is_better is True


# ── DistanceTravelled ────────────────────────────────────────────────────────


def test_distance_travelled_shape() -> None:
    B, T = 4, 8
    positions = torch.randn(B, T, 2)
    traj = _make_trajectory(B=B, T=T, positions=positions)
    score = DistanceTravelled().evaluate(traj, _WORLD)
    assert score.shape == (B,)


def test_distance_travelled_known_path() -> None:
    B, T = 1, 4
    # Each step moves 1 unit to the right → 3 steps of distance 1.0
    positions = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    traj = _make_trajectory(B=B, T=T, positions=positions)
    score = DistanceTravelled().evaluate(traj, _WORLD)
    assert score[0].item() == pytest.approx(3.0)


def test_distance_travelled_stationary() -> None:
    B, T = 2, 5
    positions = torch.zeros(B, T, 2)
    traj = _make_trajectory(B=B, T=T, positions=positions)
    score = DistanceTravelled().evaluate(traj, _WORLD)
    assert torch.all(score == 0.0)


def test_distance_travelled_missing_key_returns_zeros(caplog) -> None:
    """Falls back to zeros and logs a warning when positions key is absent."""
    B, T = 3, 6
    traj = _make_trajectory(B=B, T=T)  # no positions in aux
    with caplog.at_level("WARNING"):
        score = DistanceTravelled().evaluate(traj, _WORLD)
    assert score.shape == (B,)
    assert torch.all(score == 0.0)
    assert "not found in trajectory.aux" in caplog.text


def test_distance_travelled_custom_key() -> None:
    B, T = 2, 3
    positions = torch.tensor(
        [[[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]
    )
    traj = _make_trajectory(B=B, T=T)
    traj.aux["pos2d"] = positions
    score = DistanceTravelled(positions_key="pos2d").evaluate(traj, _WORLD)
    assert score[0].item() == pytest.approx(5.0)  # sqrt(9+16) = 5
    assert score[1].item() == pytest.approx(1.0)


def test_distance_travelled_weight_attribute() -> None:
    obj = DistanceTravelled(weight=3.0)
    assert obj.weight == pytest.approx(3.0)
    assert obj.name == "distance_travelled"
    assert obj.higher_is_better is True

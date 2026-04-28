"""Tests for evolux.fitness.aggregators (Phase 1)."""

from __future__ import annotations

import pytest
import torch

from evolux.core.protocols import FitnessAggregator
from evolux.fitness import AGGREGATOR_REGISTRY
from evolux.fitness.aggregators import WeightedSum
from evolux.fitness.objectives import FoodCollected, SurvivalTime

# ── Registry ────────────────────────────────────────────────────────────────


def test_weighted_sum_registered() -> None:
    assert "weighted_sum" in AGGREGATOR_REGISTRY


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_isinstance_fitness_aggregator_protocol() -> None:
    agg = WeightedSum(objectives=[SurvivalTime()])
    assert isinstance(agg, FitnessAggregator)


# ── WeightedSum ──────────────────────────────────────────────────────────────


def test_weighted_sum_single_objective_identity() -> None:
    """With one objective, WeightedSum output == that objective's score."""
    B = 8
    obj = SurvivalTime(weight=1.0)
    agg = WeightedSum(objectives=[obj])
    scores = {"survival_time": torch.arange(B, dtype=torch.float32)}
    result = agg.aggregate(scores)
    expected = scores["survival_time"]
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_weighted_sum_single_objective_with_weight() -> None:
    """Single objective with weight w → output == w * score."""
    B = 4
    w = 3.0
    obj = FoodCollected(weight=w)
    agg = WeightedSum(objectives=[obj])
    raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = agg.aggregate({"food_collected": raw})
    assert torch.allclose(result, w * raw)


def test_weighted_sum_two_objectives() -> None:
    B = 5
    obj1 = SurvivalTime(weight=1.0)
    obj2 = FoodCollected(weight=2.0)
    agg = WeightedSum(objectives=[obj1, obj2])
    s1 = torch.ones(B)
    s2 = torch.ones(B) * 3.0
    result = agg.aggregate({"survival_time": s1, "food_collected": s2})
    expected = 1.0 * s1 + 2.0 * s2
    assert torch.allclose(result, expected)


def test_weighted_sum_missing_score_treated_as_zero() -> None:
    """Objective not in scores contributes zero."""
    B = 3
    obj1 = SurvivalTime(weight=1.0)
    obj2 = FoodCollected(weight=5.0)
    agg = WeightedSum(objectives=[obj1, obj2])
    # Only supply survival_time; food_collected is missing
    raw = torch.tensor([2.0, 4.0, 6.0])
    result = agg.aggregate({"survival_time": raw})
    assert torch.allclose(result, raw)


def test_weighted_sum_returns_correct_shape() -> None:
    B = 16
    obj = SurvivalTime(weight=1.0)
    agg = WeightedSum(objectives=[obj])
    scores = {"survival_time": torch.randn(B)}
    result = agg.aggregate(scores)
    assert result.shape == (B,)


def test_weighted_sum_empty_objectives_returns_zeros() -> None:
    """No objectives registered → result is zero for each batch element."""
    B = 4
    agg = WeightedSum(objectives=[])
    scores = {"survival_time": torch.ones(B)}
    result = agg.aggregate(scores)
    assert torch.all(result == 0.0)
    assert result.shape == (B,)


def test_weighted_sum_empty_scores_raises() -> None:
    """Empty scores dict should raise ValueError."""
    agg = WeightedSum(objectives=[SurvivalTime()])
    with pytest.raises(ValueError, match="empty"):
        agg.aggregate({})


def test_weighted_sum_recovers_single_objective_nonzero_weight() -> None:
    """WeightedSum with one nonzero weight recovers single-objective score (SPEC)."""
    B = 6
    obj_a = SurvivalTime(weight=0.0)
    obj_b = FoodCollected(weight=1.0)
    agg = WeightedSum(objectives=[obj_a, obj_b])
    food = torch.rand(B)
    result = agg.aggregate({"survival_time": torch.rand(B), "food_collected": food})
    assert torch.allclose(result, food)

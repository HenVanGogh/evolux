"""Phase-1 fitness aggregators.

Aggregators combine per-objective score tensors into a single fitness value
used by selection algorithms.

Aggregators
-----------
- ``weighted_sum`` — weighted linear combination of objectives
"""

from __future__ import annotations

import torch
from torch import Tensor

from evolux.core.protocols import Objective
from evolux.fitness import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("weighted_sum")
class WeightedSum:
    """Weighted linear combination of per-objective score tensors.

    Notes
    -----
    ``fitness[b] = Σ_i weight_i · score_i[b]``

    When a single objective is registered with weight 1.0 the result
    equals that objective's raw score (single-objective special case).
    """

    def __init__(self, objectives: list[Objective]) -> None:
        self.objectives: list[Objective] = objectives

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor:
        """Combine per-objective scores into a single ``(B,)`` fitness tensor.

        Parameters
        ----------
        scores:
            Mapping from objective ``name`` to ``(B,)`` score tensor.
            Objectives not present in *scores* contribute zero.

        Returns
        -------
        Tensor
            ``(B,)`` weighted fitness values.

        Raises
        ------
        ValueError
            If *scores* is empty (cannot determine batch size).
        """
        if not scores:
            raise ValueError("WeightedSum.aggregate: 'scores' dict must not be empty.")

        first = next(iter(scores.values()))
        total = torch.zeros(first.shape[0], dtype=torch.float32, device=first.device)

        for obj in self.objectives:
            if obj.name in scores:
                total = total + obj.weight * scores[obj.name]

        return total

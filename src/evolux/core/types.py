"""Shared type aliases and dataclasses used across modules.

Everything here is **storage** — no logic. Concrete brains/worlds/etc. live
in their own modules but exchange these types at boundaries.

Tensor shape conventions
------------------------
B  = batch (parallel envs/agents)
T  = sequence length
H  = hidden dim
D  = generic feature dim
M  = memory slots
A  = action dim
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

# ── Aliases for self-documenting type hints ─────────────────────────────────
Action = Tensor  # (B, A) or (B,) for discrete
Reward = Tensor  # (B,)
Done = Tensor  # (B,) bool
AuxInfo = dict[str, Any]


# ── Spec dataclasses ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ObsSpec:
    """Description of the observation a Brain expects.

    `fields` maps each observation channel name to its (per-sample) shape and
    dtype. The leading batch dim B is implicit and not included in `shape`.
    """

    fields: dict[str, tuple[tuple[int, ...], torch.dtype]]
    batch_invariant: bool = True

    def keys(self) -> list[str]:
        return list(self.fields.keys())


@dataclass(frozen=True)
class ActionSpec:
    """Description of a Brain's action output."""

    discrete: bool
    n: int  # # of discrete options (if discrete) or action dim
    bounds: tuple[float, float] | None = None  # for continuous actions


@dataclass(frozen=True)
class StateSpec:
    """Description of recurrent / memory state shape."""

    fields: dict[str, tuple[tuple[int, ...], torch.dtype]]


# ── Runtime state containers (plain dicts but typed) ────────────────────────

Obs = dict[str, Tensor]
BrainState = dict[str, Tensor]
MemoryState = dict[str, Tensor]
WorldState = dict[str, Tensor]
BodyState = dict[str, Tensor]


# ── Trajectory record ───────────────────────────────────────────────────────


@dataclass
class Trajectory:
    """Per-creature record of one episode (rollout).

    Tensors are stored with shape (B, T, ...). For variable-length episodes
    use ``length`` to mask invalid timesteps.
    """

    obs: dict[str, Tensor]
    actions: Tensor
    rewards: Tensor
    dones: Tensor
    length: Tensor  # (B,) int — number of valid steps per env
    aux: dict[str, Tensor] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return self.actions.shape[0]

    @property
    def horizon(self) -> int:
        return self.actions.shape[1]

    def to(self, device: torch.device) -> Trajectory:
        def _move(t: Tensor) -> Tensor:
            return t.to(device, non_blocking=True)

        return Trajectory(
            obs={k: _move(v) for k, v in self.obs.items()},
            actions=_move(self.actions),
            rewards=_move(self.rewards),
            dones=_move(self.dones),
            length=_move(self.length),
            aux={k: _move(v) for k, v in self.aux.items()},
        )

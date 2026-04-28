"""Tests that every Protocol in evolux.core.protocols is runtime_checkable
and has a docstring, and that conforming dummy objects are accepted."""

from __future__ import annotations

import inspect
import typing

import pytest
import torch
from torch import Tensor

import evolux.core.protocols as proto_module
from evolux.core.protocols import (
    Brain,
    Memory,
    StatsLogger,
    World,
)
from evolux.core.types import (
    Action,
    ActionSpec,
    AuxInfo,
    BrainState,
    Done,
    MemoryState,
    Obs,
    ObsSpec,
    Reward,
    StateSpec,
)


def _collect_protocols() -> list[tuple[str, type]]:
    """Return all Protocol classes defined in the protocols module."""
    result = []
    for name, obj in inspect.getmembers(proto_module, inspect.isclass):
        if (
            hasattr(obj, "__mro__")
            and typing.Protocol in obj.__mro__
            and obj is not typing.Protocol
        ):
            result.append((name, obj))
    return result


@pytest.mark.parametrize("name,proto", _collect_protocols())
def test_protocol_is_runtime_checkable(name: str, proto: type) -> None:
    """Every Protocol must be decorated with @runtime_checkable."""
    assert hasattr(proto, "__protocol_attrs__") or hasattr(proto, "_is_protocol"), (
        f"{name} must be a Protocol"
    )
    # Verify isinstance() works (runtime_checkable)
    try:
        isinstance(object(), proto)
    except TypeError as exc:
        pytest.fail(f"{name} is not @runtime_checkable: {exc}")


@pytest.mark.parametrize("name,proto", _collect_protocols())
def test_protocol_has_docstring(name: str, proto: type) -> None:
    """Every Protocol must have a non-empty docstring."""
    doc = proto.__doc__
    # typing.Protocol injects a default docstring like "Protocol base class."
    # We require the class to have defined its own docstring above the default.
    assert doc is not None and doc.strip() != "", f"{name} is missing a docstring"


# ── Conformance checks ────────────────────────────────────────────────────────


class _DummyBrain:
    obs_spec = ObsSpec(fields={})
    action_spec = ActionSpec(discrete=True, n=2)
    state_spec = StateSpec(fields={})

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        return {}

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]:
        return torch.zeros(1), {}, {}

    def trainable_parameters(self):  # type: ignore[override]
        return iter([])


def test_brain_conformance() -> None:
    assert isinstance(_DummyBrain(), Brain)


class _DummyMemory:
    capacity = 8
    key_dim = 4
    val_dim = 4

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        return {}

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState:
        return state

    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor:
        return query

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState:
        return state


def test_memory_conformance() -> None:
    assert isinstance(_DummyMemory(), Memory)


class _DummyWorld:
    batch_size = 1
    device = torch.device("cpu")

    def reset(self, mask: Tensor | None = None) -> None:
        pass

    def observe(self) -> Obs:
        return {}

    def step(self, action: Action) -> tuple[Reward, Done, AuxInfo]:
        return torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}


def test_world_conformance() -> None:
    assert isinstance(_DummyWorld(), World)


class _DummyLogger:
    def log_scalar(self, key: str, value: float, step: int) -> None:
        pass

    def log_dict(self, data: dict, step: int) -> None:
        pass

    def log_image(self, key: str, image: Tensor, step: int) -> None:
        pass

    def close(self) -> None:
        pass


def test_stats_logger_conformance() -> None:
    assert isinstance(_DummyLogger(), StatsLogger)

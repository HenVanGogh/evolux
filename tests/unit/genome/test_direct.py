"""Unit tests for evolux.genome.direct (Phase 1 acceptance tests)."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor

from evolux.core.protocols import Brain, Genome, GenomeOperator
from evolux.core.rng import RNG
from evolux.core.types import ActionSpec, AuxInfo, BrainState, ObsSpec, StateSpec
from evolux.genome import GENOME_OPERATOR_REGISTRY, GENOME_REGISTRY
from evolux.genome.direct import DirectGenome, DirectOperator

# ── Helpers / fixtures ────────────────────────────────────────────────────────


def make_genome(n: int = 12, seed: int = 0) -> DirectGenome:
    """Return a deterministic DirectGenome with *n* parameters."""
    rng = RNG(seed).split("genome")
    params = torch.rand(n, generator=rng)
    return DirectGenome(params)


class _DummyBrain:
    """Minimal Brain implementation for smoke-testing decode_brain."""

    def __init__(self, weights: dict[str, Tensor]) -> None:
        self._weights = weights
        self.obs_spec = ObsSpec(fields={"obs": ((4,), torch.float32)})
        self.action_spec = ActionSpec(discrete=False, n=2)
        self.state_spec = StateSpec(fields={})

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        return {}

    def forward(
        self, obs: dict[str, Tensor], state: BrainState
    ) -> tuple[Tensor, BrainState, AuxInfo]:
        B = next(iter(obs.values())).shape[0]
        action = torch.zeros(B, self.action_spec.n)
        return action, state, {}

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        return iter([])


class _DummyBrainFactory:
    """Minimal BrainFactory that exposes parameter_shapes and builds a _DummyBrain."""

    parameter_shapes: ClassVar[dict[str, tuple[int, ...]]] = {
        "layer1.weight": (4, 3),
        "layer1.bias": (4,),
    }
    obs_spec = ObsSpec(fields={"obs": ((4,), torch.float32)})
    action_spec = ActionSpec(discrete=False, n=2)

    def build(self, obs_spec: ObsSpec, action_spec: ActionSpec, hparams: dict[str, Any]) -> Brain:
        weights: dict[str, Tensor] = hparams.get("weights", {})
        return _DummyBrain(weights)  # type: ignore[return-value]


# ── DirectGenome: construction & properties ───────────────────────────────────


def test_genome_stores_float32() -> None:
    g = DirectGenome(torch.arange(5, dtype=torch.float64))
    assert g.params.dtype == torch.float32


def test_genome_n_params() -> None:
    g = make_genome(n=16)
    assert g.n_params == 16


def test_genome_no_alias() -> None:
    """params must be a copy — not a view of the constructor argument."""
    raw = torch.ones(4)
    g = DirectGenome(raw)
    raw[0] = 999.0
    assert g.params[0].item() != 999.0


# ── serialize / deserialize round-trip ────────────────────────────────────────


def test_serialize_deserialize_round_trip_bit_exact() -> None:
    g = make_genome(n=32)
    blob = g.serialize()
    g2 = DirectGenome.deserialize(blob)
    assert torch.equal(g.params, g2.params), "round-trip must be bit-exact"


def test_serialize_produces_bytes() -> None:
    g = make_genome(n=8)
    blob = g.serialize()
    assert isinstance(blob, bytes)
    assert len(blob) == 8 * 4  # 4 bytes per float32


def test_deserialize_does_not_alias_blob() -> None:
    """Mutating the original blob must not corrupt the deserialized genome."""
    g = make_genome(n=4)
    blob = bytearray(g.serialize())
    g2 = DirectGenome.deserialize(bytes(blob))
    # Overwrite blob
    blob[:] = b"\x00" * len(blob)
    assert not torch.all(g2.params == 0.0)


# ── distance ──────────────────────────────────────────────────────────────────


def test_distance_self_is_zero() -> None:
    g = make_genome(n=20)
    assert DirectOperator().distance(g, g) == pytest.approx(0.0)


def test_distance_symmetric() -> None:
    a = make_genome(n=20, seed=1)
    b = make_genome(n=20, seed=2)
    op = DirectOperator()
    assert op.distance(a, b) == pytest.approx(op.distance(b, a))


def test_distance_non_negative() -> None:
    a = make_genome(n=20, seed=3)
    b = make_genome(n=20, seed=4)
    assert DirectOperator().distance(a, b) >= 0.0


def test_distance_l2_correctness() -> None:
    a = DirectGenome(torch.tensor([1.0, 0.0, 0.0]))
    b = DirectGenome(torch.tensor([0.0, 0.0, 0.0]))
    op = DirectOperator()
    assert op.distance(a, b) == pytest.approx(1.0)


# ── mutation ──────────────────────────────────────────────────────────────────


def test_mutate_no_in_place_modification() -> None:
    """mutate must return a NEW genome without touching the original."""
    g = make_genome(n=16)
    original_params = g.params.clone()
    rng = RNG(7).split("mutate")
    op = DirectOperator(sigma=0.1)
    op.mutate(g, rng)
    assert torch.equal(g.params, original_params), "original genome must be unmodified"


def test_mutate_returns_different_genome() -> None:
    g = make_genome(n=16)
    rng = RNG(7).split("mutate")
    op = DirectOperator(sigma=0.5)
    g2 = op.mutate(g, rng)
    assert not torch.equal(g.params, g2.params), "mutated genome should differ from parent"


def test_mutate_sigma_statistical() -> None:
    """Over many samples the empirical std of Δparams should be ≈ sigma."""
    n_samples = 10_000
    sigma = 0.1
    op = DirectOperator(sigma=sigma)
    g = make_genome(n=1)

    diffs: list[float] = []
    for i in range(n_samples):
        rng = RNG(i).split("mutate")
        g2 = op.mutate(g, rng)
        diffs.append((g2.params[0] - g.params[0]).item())

    empirical_std = float(torch.tensor(diffs).std())
    # Allow ±5% relative tolerance
    assert abs(empirical_std - sigma) / sigma < 0.05, (
        f"expected sigma~{sigma}, got {empirical_std:.4f}"
    )


def test_mutate_never_produces_nan() -> None:
    g = make_genome(n=64)
    op = DirectOperator(sigma=1.0)
    for i in range(20):
        rng = RNG(i).split("mutate")
        g2 = op.mutate(g, rng)
        assert not torch.isnan(g2.params).any(), "mutated genome must not contain NaN"


# ── crossover ────────────────────────────────────────────────────────────────


def test_crossover_no_in_place_modification() -> None:
    a = make_genome(n=16, seed=0)
    b = make_genome(n=16, seed=1)
    params_a = a.params.clone()
    params_b = b.params.clone()
    rng = RNG(9).split("crossover")
    DirectOperator().crossover(a, b, rng)
    assert torch.equal(a.params, params_a), "parent a must be unmodified"
    assert torch.equal(b.params, params_b), "parent b must be unmodified"


def test_crossover_child_uses_only_parent_values() -> None:
    a = make_genome(n=32, seed=0)
    b = make_genome(n=32, seed=1)
    rng = RNG(9).split("crossover")
    child = DirectOperator().crossover(a, b, rng)
    for i in range(child.n_params):
        val = child.params[i].item()
        assert val == pytest.approx(a.params[i].item()) or val == pytest.approx(
            b.params[i].item()
        ), f"child param [{i}] = {val} is not from either parent"


# ── decode_brain smoke test ──────────────────────────────────────────────────


def _factory_n_params(factory: _DummyBrainFactory) -> int:
    """Return the total number of parameters in a _DummyBrainFactory."""
    return sum(math.prod(shape) for shape in factory.parameter_shapes.values())


# ── decode_brain smoke test ──────────────────────────────────────────────────


def test_decode_brain_smoke() -> None:
    """decode_brain must return a Brain that passes forward on dummy obs."""
    factory = _DummyBrainFactory()
    n = _factory_n_params(factory)
    g = DirectGenome(torch.randn(n))
    brain = g.decode_brain(factory)  # type: ignore[arg-type]

    obs = {"obs": torch.randn(2, 4)}
    state = brain.init_state(2, torch.device("cpu"))
    action, _new_state, _aux = brain.forward(obs, state)
    assert action.shape == (2, 2), f"unexpected action shape: {action.shape}"


def test_decode_brain_weights_injected() -> None:
    """Weights extracted from the genome must reach the factory."""
    factory = _DummyBrainFactory()
    n = _factory_n_params(factory)
    params = torch.arange(n, dtype=torch.float32)
    g = DirectGenome(params)
    brain: _DummyBrain = g.decode_brain(factory)  # type: ignore[arg-type,assignment]
    w = brain._weights["layer1.weight"]
    assert w.shape == (4, 3)
    # first 12 params → layer1.weight
    assert torch.equal(w, params[:12].reshape(4, 3))


def test_decode_morphology_raises_not_implemented() -> None:
    g = make_genome(n=4)
    with pytest.raises(NotImplementedError):
        g.decode_morphology()


# ── Registry ──────────────────────────────────────────────────────────────────


def test_genome_registry_has_direct() -> None:
    assert "direct" in GENOME_REGISTRY


def test_genome_operator_registry_has_direct() -> None:
    assert "direct" in GENOME_OPERATOR_REGISTRY


def test_registry_returns_correct_classes() -> None:
    assert GENOME_REGISTRY.get("direct") is DirectGenome
    assert GENOME_OPERATOR_REGISTRY.get("direct") is DirectOperator


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_direct_genome_satisfies_genome_protocol() -> None:
    g = make_genome(n=4)
    assert isinstance(g, Genome)


def test_direct_operator_satisfies_genome_operator_protocol() -> None:
    op = DirectOperator()
    assert isinstance(op, GenomeOperator)

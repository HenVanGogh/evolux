"""Unit tests for evolux.brain.ssm.SSMBrain.

Covers the acceptance bullets from src/evolux/brain/SPEC.md §SSMBrain:

- Forward output shape matches action_spec.
- State recurrence: two consecutive forward calls produce different states.
- Gradients flow to all trainable_parameters.
- Determinism under fixed RNG.
- Linear-time check: T=64 <= ~2x T=32.
"""

from __future__ import annotations

import time

import pytest
import torch

from evolux.brain import BRAIN_REGISTRY
from evolux.brain.ssm import SSMBrain
from evolux.core.protocols import Brain
from evolux.core.rng import RNG
from evolux.core.types import ActionSpec, ObsSpec

# ── fixtures / helpers ─────────────────────────────────────────────────────


def _obs_spec() -> ObsSpec:
    return ObsSpec(
        fields={
            "proprio": ((6,), torch.float32),
            "sensor": ((4,), torch.float32),
        }
    )


def _make(action: ActionSpec, **kwargs) -> SSMBrain:
    return SSMBrain(
        _obs_spec(),
        action,
        hidden_dim=16,
        n_layers=2,
        state_dim=8,
        expand=2,
        conv_kernel_size=4,
        **kwargs,
    )


def _obs(b: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "proprio": torch.randn(b, 6, device=device),
        "sensor": torch.randn(b, 4, device=device),
    }


# ── registration ───────────────────────────────────────────────────────────


def test_registered_in_registry() -> None:
    assert "ssm" in BRAIN_REGISTRY
    assert BRAIN_REGISTRY.get("ssm") is SSMBrain


# ── protocol ───────────────────────────────────────────────────────────────


def test_protocol_compliance(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    assert isinstance(brain, Brain)


# ── forward output shape ────────────────────────────────────────────────────


def test_forward_shape_discrete(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=True, n=5))
    state = brain.init_state(3, cpu_device)
    action, _, aux = brain.forward(_obs(3, cpu_device), state)
    assert action.shape == (3,)
    assert action.dtype == torch.long
    assert aux["logits"].shape == (3, 5)
    assert aux["value"].shape == (3,)


def test_forward_shape_continuous(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3))
    state = brain.init_state(2, cpu_device)
    action, _, aux = brain.forward(_obs(2, cpu_device), state)
    assert action.shape == (2, 3)
    assert aux["logits"].shape == (2, 3)


def test_forward_shape_continuous_with_bounds(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3, bounds=(-1.0, 1.0)))
    state = brain.init_state(2, cpu_device)
    action, _, _ = brain.forward(_obs(2, cpu_device), state)
    assert action.shape == (2, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


# ── state recurrence ────────────────────────────────────────────────────────


def test_state_recurrence(cpu_device: torch.device) -> None:
    """Two consecutive forward calls must produce different SSM hidden states."""
    brain = _make(ActionSpec(discrete=True, n=4))
    s0 = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    _, s1, _ = brain.forward(obs, s0)
    _, s2, _ = brain.forward(obs, s1)

    # h shape: (L, B, E, N)
    assert s1["h"].shape == s0["h"].shape
    assert s2["h"].shape == s0["h"].shape

    # s1 differs from s0 (zeros)
    assert not torch.allclose(s1["h"], s0["h"])
    # s2 differs from s1
    assert not torch.allclose(s2["h"], s1["h"])


def test_state_shape(cpu_device: torch.device) -> None:
    """init_state returns h: (L, B, E, N)."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(5, cpu_device)
    L = brain.n_layers
    E = brain.inner_dim
    N = brain.state_dim
    assert state["h"].shape == (L, 5, E, N)
    assert state["h"].device.type == cpu_device.type
    assert torch.all(state["h"] == 0.0)


# ── gradient flow ───────────────────────────────────────────────────────────


def test_gradients_flow_to_all_trainable_parameters(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    _, _, aux = brain.forward(_obs(2, cpu_device), state)
    loss = aux["logits"].sum() + aux["value"].sum()
    loss.backward()
    params = list(brain.trainable_parameters())
    assert len(params) > 0
    for p in params:
        assert p.grad is not None, f"parameter {p.shape} received no gradient"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for param {p.shape}"


# ── determinism ─────────────────────────────────────────────────────────────


def test_determinism_under_fixed_rng(cpu_device: torch.device) -> None:
    """Two brains with same init seed produce identical forward outputs."""
    rng_a = RNG(seed=42)
    rng_b = RNG(seed=42)

    gen_a = rng_a.split("ssm")
    gen_b = rng_b.split("ssm")

    def _build_with_gen(gen: torch.Generator) -> SSMBrain:
        brain = _make(ActionSpec(discrete=True, n=4))
        # Reinitialize parameters deterministically using the generator
        with torch.no_grad():
            for p in brain.parameters():
                p.copy_(torch.empty_like(p).uniform_(-0.1, 0.1, generator=gen))
        return brain

    brain_a = _build_with_gen(gen_a)
    brain_b = _build_with_gen(gen_b)

    # Same obs for both
    torch.manual_seed(0)
    obs = _obs(2, cpu_device)

    s_a = brain_a.init_state(2, cpu_device)
    s_b = brain_b.init_state(2, cpu_device)

    _, s_a2, aux_a = brain_a.forward(obs, s_a)
    _, s_b2, aux_b = brain_b.forward(obs, s_b)

    assert torch.allclose(aux_a["logits"], aux_b["logits"]), (
        "logits differ between identically-seeded brains"
    )
    assert torch.allclose(s_a2["h"], s_b2["h"]), (
        "hidden states differ between identically-seeded brains"
    )


# ── linear-time scaling ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_linear_time_scaling(cpu_device: torch.device) -> None:
    """Forward pass on T=64 should be at most ~2x T=32 (soft assert, 3x tolerance)."""
    brain = _make(ActionSpec(discrete=True, n=4))
    B = 4
    N_REPS = 20

    def _run_t_steps(t: int) -> float:
        state = brain.init_state(B, cpu_device)
        obs = _obs(B, cpu_device)
        start = time.perf_counter()
        for _ in range(t):
            _, state, _ = brain.forward(obs, state)
        return time.perf_counter() - start

    # Warm-up
    _run_t_steps(8)

    t32 = min(_run_t_steps(32) for _ in range(N_REPS))
    t64 = min(_run_t_steps(64) for _ in range(N_REPS))

    ratio = t64 / (t32 + 1e-9)
    # Linear time -> ratio ~= 2.0.  Allow 3x to account for system noise.
    assert ratio < 3.0, f"T=64 took {ratio:.2f}x T=32 -- expected <= 3.0 for linear-time SSM"


# ── construction validation ─────────────────────────────────────────────────


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        SSMBrain(ObsSpec(fields={}), ActionSpec(discrete=True, n=3))
    with pytest.raises(ValueError):
        SSMBrain(_obs_spec(), ActionSpec(discrete=True, n=0))


# ── immutability of input state ─────────────────────────────────────────────


def test_forward_does_not_mutate_input_state(cpu_device: torch.device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    h0 = state["h"].clone()
    obs = _obs(2, cpu_device)
    _, state1, _ = brain.forward(obs, state)
    # Original state must not have changed
    assert torch.allclose(state["h"], h0), "forward mutated the input state"
    # state1 must differ from state0
    assert not torch.allclose(state1["h"], h0)

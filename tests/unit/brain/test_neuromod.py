"""Unit tests for evolux.brain.neuromod.NeuromodWrapper.

Covers the acceptance bullets from src/evolux/brain/SPEC.md and the issue:

- aux["modulators"] has shape (B, n_modulators) and is finite.
- forward output shape matches inner brain's action spec.
- gradients flow into both wrapper and inner brain parameters.
- state recurrence preserved through wrapper.
- with all modulators forced to 1, output approximately equals raw inner-brain output.
- Registration in BRAIN_REGISTRY under key 'neuromod'.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from evolux.brain import BRAIN_REGISTRY
from evolux.brain.neuromod import NeuromodWrapper
from evolux.brain.transformer import TransformerBrain
from evolux.core.protocols import Brain
from evolux.core.types import ActionSpec, ObsSpec

# ── fixtures / helpers ───────────────────────────────────────────────────────

OBS_FIELDS = {
    "proprio": ((6,), torch.float32),
    "image": ((1, 4, 4), torch.float32),
}
OBS_DIM = 6 + 1 * 4 * 4  # 22


def _obs_spec() -> ObsSpec:
    return ObsSpec(fields=OBS_FIELDS)


def _inner_brain(action_spec: ActionSpec) -> TransformerBrain:
    return TransformerBrain(
        _obs_spec(),
        action_spec,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        max_seq_len=8,
    )


def _wrapped(action_spec: ActionSpec, n_modulators: int = 4) -> NeuromodWrapper:
    return NeuromodWrapper(
        _inner_brain(action_spec),
        _obs_spec(),
        action_spec,
        n_modulators=n_modulators,
        hidden_dim=16,
    )


def _obs(b: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "proprio": torch.randn(b, 6, device=device),
        "image": torch.randn(b, 1, 4, 4, device=device),
    }


# ── registration ─────────────────────────────────────────────────────────────


def test_registered_in_registry() -> None:
    assert "neuromod" in BRAIN_REGISTRY
    assert BRAIN_REGISTRY.get("neuromod") is NeuromodWrapper


# ── protocol conformance ─────────────────────────────────────────────────────


def test_protocol_compliance_discrete(cpu_device: torch.device) -> None:
    wrapper = _wrapped(ActionSpec(discrete=True, n=4))
    assert isinstance(wrapper, Brain)


def test_protocol_compliance_continuous(cpu_device: torch.device) -> None:
    wrapper = _wrapped(ActionSpec(discrete=False, n=3))
    assert isinstance(wrapper, Brain)


# ── modulators shape & finiteness ─────────────────────────────────────────────


@pytest.mark.parametrize("n_modulators", [1, 4, 8])
def test_modulators_shape_and_finite_discrete(cpu_device: torch.device, n_modulators: int) -> None:
    B = 3
    wrapper = _wrapped(ActionSpec(discrete=True, n=5), n_modulators=n_modulators)
    state = wrapper.init_state(B, cpu_device)
    _, _, aux = wrapper.forward(_obs(B, cpu_device), state)
    assert "modulators" in aux
    mods = aux["modulators"]
    assert mods.shape == (B, n_modulators), f"expected ({B}, {n_modulators}), got {mods.shape}"
    assert torch.isfinite(mods).all()


@pytest.mark.parametrize("n_modulators", [1, 4, 8])
def test_modulators_shape_and_finite_continuous(
    cpu_device: torch.device, n_modulators: int
) -> None:
    B = 5
    wrapper = _wrapped(ActionSpec(discrete=False, n=3), n_modulators=n_modulators)
    state = wrapper.init_state(B, cpu_device)
    _, _, aux = wrapper.forward(_obs(B, cpu_device), state)
    mods = aux["modulators"]
    assert mods.shape == (B, n_modulators)
    assert torch.isfinite(mods).all()


# ── output shape ─────────────────────────────────────────────────────────────


def test_forward_shape_discrete(cpu_device: torch.device) -> None:
    B = 3
    wrapper = _wrapped(ActionSpec(discrete=True, n=7))
    state = wrapper.init_state(B, cpu_device)
    action, _, _ = wrapper.forward(_obs(B, cpu_device), state)
    assert action.shape == (B,)
    assert action.dtype == torch.long


def test_forward_shape_continuous(cpu_device: torch.device) -> None:
    B = 4
    wrapper = _wrapped(ActionSpec(discrete=False, n=5))
    state = wrapper.init_state(B, cpu_device)
    action, _, _ = wrapper.forward(_obs(B, cpu_device), state)
    assert action.shape == (B, 5)


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_flow_discrete(cpu_device: torch.device) -> None:
    """Gradients must reach both wrapper (modulator_mlp) and inner brain params."""
    B = 2
    wrapper = _wrapped(ActionSpec(discrete=True, n=4), n_modulators=4)
    state = wrapper.init_state(B, cpu_device)
    _, _, aux = wrapper.forward(_obs(B, cpu_device), state)
    # For discrete, gradients flow through gated logits, value, and modulators.
    loss = aux["logits"].sum() + aux["value"].sum() + aux["modulators"].sum()
    loss.backward()

    wrapper_params = list(wrapper.modulator_mlp.parameters())
    inner_params = list(wrapper.inner_brain.parameters())
    assert len(wrapper_params) > 0
    assert len(inner_params) > 0
    for p in wrapper_params:
        assert p.grad is not None, f"modulator param {p.shape} has no gradient"
        assert torch.isfinite(p.grad).all()
    for p in inner_params:
        assert p.grad is not None, f"inner-brain param {p.shape} has no gradient"
        assert torch.isfinite(p.grad).all()


def test_gradients_flow_continuous(cpu_device: torch.device) -> None:
    B = 2
    wrapper = _wrapped(ActionSpec(discrete=False, n=3), n_modulators=4)
    state = wrapper.init_state(B, cpu_device)
    action, _, aux = wrapper.forward(_obs(B, cpu_device), state)
    loss = action.sum() + aux["value"].sum() + aux["modulators"].sum()
    loss.backward()

    wrapper_params = list(wrapper.modulator_mlp.parameters())
    inner_params = list(wrapper.inner_brain.parameters())
    for p in wrapper_params:
        assert p.grad is not None, f"modulator param {p.shape} has no gradient"
        assert torch.isfinite(p.grad).all()
    for p in inner_params:
        assert p.grad is not None, f"inner-brain param {p.shape} has no gradient"
        assert torch.isfinite(p.grad).all()


# ── state recurrence ─────────────────────────────────────────────────────────


def test_state_recurrence_preserved(cpu_device: torch.device) -> None:
    """The wrapper must not break the inner brain's recurrent state updates."""
    B = 2
    wrapper = _wrapped(ActionSpec(discrete=True, n=4))
    s0 = wrapper.init_state(B, cpu_device)
    obs = _obs(B, cpu_device)
    _, s1, _ = wrapper.forward(obs, s0)
    _, s2, _ = wrapper.forward(obs, s1)
    # KV cache grows: same acceptance criterion as TransformerBrain.
    assert s1["k"].shape[2] == 1, "KV cache should have length 1 after first step"
    assert s2["k"].shape[2] == 2, "KV cache should have length 2 after second step"
    # Original state untouched.
    assert s0["k"].shape[2] == 0


# ── all modulators == 1 → identity ───────────────────────────────────────────


def test_unit_modulators_identity_continuous(cpu_device: torch.device) -> None:
    """When modulator gates are all 1, output must equal the raw inner-brain output."""
    B = 3
    inner = _inner_brain(ActionSpec(discrete=False, n=3))
    wrapper = NeuromodWrapper(
        inner, _obs_spec(), ActionSpec(discrete=False, n=3), n_modulators=4, hidden_dim=16
    )
    obs = _obs(B, cpu_device)
    state = wrapper.init_state(B, cpu_device)

    # Reference: raw inner brain output.
    raw_action, _, _ = inner.forward(obs, state)

    # Patch modulator_mlp to return all-ones.
    def _ones_forward(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], wrapper.n_modulators, device=x.device)

    with patch.object(wrapper.modulator_mlp, "forward", side_effect=_ones_forward):
        gated_action, _, aux = wrapper.forward(obs, state)

    assert torch.allclose(gated_action, raw_action, atol=1e-6), (
        "With all modulators=1 the gated action must equal the inner-brain action"
    )
    assert torch.allclose(aux["modulators"], torch.ones_like(aux["modulators"]))


def test_unit_modulators_identity_discrete(cpu_device: torch.device) -> None:
    """Discrete: gated action (argmax) must equal raw inner-brain action when gates=1."""
    B = 3
    inner = _inner_brain(ActionSpec(discrete=True, n=5))
    wrapper = NeuromodWrapper(
        inner, _obs_spec(), ActionSpec(discrete=True, n=5), n_modulators=4, hidden_dim=16
    )
    obs = _obs(B, cpu_device)
    state = wrapper.init_state(B, cpu_device)

    raw_action, _, _ = inner.forward(obs, state)

    def _ones_forward(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], wrapper.n_modulators, device=x.device)

    with patch.object(wrapper.modulator_mlp, "forward", side_effect=_ones_forward):
        gated_action, _, _ = wrapper.forward(obs, state)

    assert torch.equal(gated_action, raw_action), (
        "With all modulators=1 the discrete action must equal the inner-brain action"
    )


# ── construction guards ───────────────────────────────────────────────────────


def test_raises_for_non_module_inner_brain() -> None:
    class FakeBrain:
        obs_spec = _obs_spec()
        action_spec = ActionSpec(discrete=True, n=4)
        state_spec = None

    with pytest.raises(TypeError, match=r"torch\.nn\.Module"):
        NeuromodWrapper(FakeBrain(), _obs_spec(), ActionSpec(discrete=True, n=4))  # type: ignore[arg-type]


def test_raises_for_zero_modulators() -> None:
    with pytest.raises(ValueError, match="n_modulators"):
        _wrapped(ActionSpec(discrete=True, n=4), n_modulators=0)

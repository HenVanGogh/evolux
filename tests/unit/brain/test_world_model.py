"""Unit tests for evolux.brain.world_model.WorldModelBrain.

Covers all acceptance tests from src/evolux/brain/SPEC.md:

- Forward output shape matches action_spec.
- imagine(horizon=10) returns a Trajectory with T=10 and finite tensors.
- State recurrence works (h changes between steps).
- Gradients flow to all trainable parameters (policy, dynamics, reward, decoder).
- Reward head outputs shape (B,) or (B, 1).
- Registration in BRAIN_REGISTRY under key 'world_model'.
- Brain and Imaginer Protocol compliance.
"""

from __future__ import annotations

import pytest
import torch

from evolux.brain import BRAIN_REGISTRY
from evolux.brain.world_model import WorldModelBrain
from evolux.core.protocols import Brain, Imaginer
from evolux.core.types import ActionSpec, ObsSpec

# ── fixtures / helpers ────────────────────────────────────────────────────


def _obs_spec() -> ObsSpec:
    return ObsSpec(
        fields={
            "proprio": ((6,), torch.float32),
            "image": ((1, 4, 4), torch.float32),
        }
    )


def _make(action: ActionSpec, **kwargs) -> WorldModelBrain:
    return WorldModelBrain(
        _obs_spec(),
        action,
        det_dim=16,
        n_cats=4,
        cat_dim=4,
        hidden_dim=16,
        **kwargs,
    )


def _obs(b: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "proprio": torch.randn(b, 6, device=device),
        "image": torch.randn(b, 1, 4, 4, device=device),
    }


# ── registry & protocol ───────────────────────────────────────────────────


def test_registered_in_registry() -> None:
    assert "world_model" in BRAIN_REGISTRY
    assert BRAIN_REGISTRY.get("world_model") is WorldModelBrain


def test_brain_protocol_compliance(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    assert isinstance(brain, Brain)


def test_imaginer_protocol_compliance(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    assert isinstance(brain, Imaginer)


# ── forward shape tests ───────────────────────────────────────────────────


def test_forward_shape_discrete(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=5))
    state = brain.init_state(3, cpu_device)
    action, _, aux = brain.forward(_obs(3, cpu_device), state)
    assert action.shape == (3,)
    assert action.dtype == torch.long
    assert aux["logits"].shape == (3, 5)
    assert aux["value"].shape == (3,)


def test_forward_shape_continuous(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3))
    state = brain.init_state(2, cpu_device)
    action, _, aux = brain.forward(_obs(2, cpu_device), state)
    assert action.shape == (2, 3)
    assert aux["logits"].shape == (2, 3)


def test_forward_shape_continuous_with_bounds(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3, bounds=(-1.0, 1.0)))
    state = brain.init_state(2, cpu_device)
    action, _, _ = brain.forward(_obs(2, cpu_device), state)
    assert action.shape == (2, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


# ── reward head shape ─────────────────────────────────────────────────────


def test_reward_head_shape(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(4, cpu_device)
    _, _, aux = brain.forward(_obs(4, cpu_device), state)
    reward = aux["reward_pred"]
    assert reward.shape == (4,), f"reward head must output (B,), got {reward.shape}"


# ── state recurrence ──────────────────────────────────────────────────────


def test_state_recurrence_works(cpu_device) -> None:
    """Acceptance test: s2 != s1 — recurrent hidden state changes each step."""
    brain = _make(ActionSpec(discrete=True, n=4))
    s0 = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    _, s1, _ = brain.forward(obs, s0)
    _, s2, _ = brain.forward(obs, s1)
    # h must change between consecutive steps
    assert not torch.allclose(s1["h"], s2["h"]), "h should evolve between steps"


def test_init_state_zeros(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    s = brain.init_state(5, cpu_device)
    assert s["h"].shape == (5, brain.det_dim)
    assert s["z"].shape == (5, brain.stoch_dim)
    assert torch.all(s["h"] == 0)
    assert torch.all(s["z"] == 0)


def test_forward_does_not_mutate_input_state(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    s0 = brain.init_state(2, cpu_device)
    h_before = s0["h"].clone()
    obs = _obs(2, cpu_device)
    brain.forward(obs, s0)
    assert torch.allclose(s0["h"], h_before), "forward must not mutate input state"


# ── imagine ───────────────────────────────────────────────────────────────


def test_imagine_horizon_shape(cpu_device) -> None:
    """imagine(horizon=10) returns a Trajectory with T=10."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    traj = brain.imagine(obs, state, horizon=10)
    assert traj.horizon == 10, f"expected horizon=10, got {traj.horizon}"
    assert traj.batch_size == 2


def test_imagine_finite_tensors(cpu_device) -> None:
    """Imagined trajectory must contain only finite (non-NaN, non-Inf) tensors."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    traj = brain.imagine(obs, state, horizon=10)
    assert torch.isfinite(traj.rewards).all(), "rewards must be finite"
    assert torch.isfinite(traj.actions.float()).all(), "actions must be finite"
    for k, v in traj.obs.items():
        assert torch.isfinite(v).all(), f"imagined obs[{k!r}] must be finite"


def test_imagine_obs_shapes(cpu_device) -> None:
    """Imagined obs must match declared obs_spec shapes with leading (B, T)."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    traj = brain.imagine(obs, state, horizon=5)
    for field_name, (field_shape, _) in brain.obs_spec.fields.items():
        expected = (2, 5, *field_shape)
        actual = traj.obs[field_name].shape
        assert actual == expected, (
            f"imagined obs[{field_name!r}]: expected {expected}, got {actual}"
        )


def test_imagine_length_field(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(3, cpu_device)
    traj = brain.imagine(_obs(3, cpu_device), state, horizon=7)
    assert traj.length.shape == (3,)
    assert torch.all(traj.length == 7)


def test_imagine_continuous_actions(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3))
    state = brain.init_state(2, cpu_device)
    traj = brain.imagine(_obs(2, cpu_device), state, horizon=4)
    assert traj.actions.shape == (2, 4, 3)


# ── gradient flow ─────────────────────────────────────────────────────────


def test_gradients_flow_to_all_parameters_forward(cpu_device) -> None:
    """All trainable parameters must receive a gradient via the forward pass."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    _, _, aux = brain.forward(_obs(2, cpu_device), state)
    loss = (
        aux["logits"].sum()
        + aux["value"].sum()
        + aux["reward_pred"].sum()
        + aux["recon"].sum()
        + aux["prior_z_logits"].sum()  # ensures prior_net gets gradients (KL target)
    )
    loss.backward()
    params = list(brain.trainable_parameters())
    assert len(params) > 0
    for p in params:
        assert p.grad is not None, f"parameter {p.shape} must receive a gradient"
        assert torch.isfinite(p.grad).all(), f"gradient must be finite for {p.shape}"


def test_gradients_flow_through_imagine(cpu_device) -> None:
    """Gradients must flow back through the imagined trajectory (for world-model training)."""
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    traj = brain.imagine(obs, state, horizon=3)
    loss = traj.rewards.sum()
    loss.backward()
    params = list(brain.trainable_parameters())
    grad_params = [p for p in params if p.grad is not None]
    assert len(grad_params) > 0, "at least some parameters must get gradients via imagine"


# ── construction guards ───────────────────────────────────────────────────


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        WorldModelBrain(ObsSpec(fields={}), ActionSpec(discrete=True, n=3))
    with pytest.raises(ValueError):
        WorldModelBrain(_obs_spec(), ActionSpec(discrete=True, n=0))

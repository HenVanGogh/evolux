"""Unit tests for evolux.brain.transformer.TransformerBrain.

Covers the acceptance bullets from src/evolux/brain/SPEC.md:

- Forward output shape matches action_spec.
- State persistence: state evolves between successive forward calls (KV cache).
- Gradients flow to all trainable_parameters.
- Registration in BRAIN_REGISTRY under key 'transformer'.
- Brain Protocol compliance.
"""

from __future__ import annotations

import pytest
import torch

from evolux.brain import BRAIN_REGISTRY
from evolux.brain.transformer import TransformerBrain
from evolux.core.protocols import Brain
from evolux.core.types import ActionSpec, ObsSpec


def _obs_spec() -> ObsSpec:
    return ObsSpec(
        fields={
            "proprio": ((6,), torch.float32),
            "image": ((1, 4, 4), torch.float32),
        }
    )


def _make(action: ActionSpec, **kwargs) -> TransformerBrain:
    return TransformerBrain(
        _obs_spec(),
        action,
        hidden_dim=16,
        n_layers=2,
        n_heads=2,
        mlp_ratio=2,
        max_seq_len=8,
        **kwargs,
    )


def _obs(b: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "proprio": torch.randn(b, 6, device=device),
        "image": torch.randn(b, 1, 4, 4, device=device),
    }


def test_registered_in_registry() -> None:
    assert "transformer" in BRAIN_REGISTRY
    assert BRAIN_REGISTRY.get("transformer") is TransformerBrain


def test_protocol_compliance(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    assert isinstance(brain, Brain)


def test_forward_shape_discrete(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=5))
    state = brain.init_state(3, cpu_device)
    action, _, aux = brain.forward(_obs(3, cpu_device), state)
    assert action.shape == (3,)
    assert action.dtype == torch.long
    assert aux["logits"].shape == (3, 5)
    assert aux["value"].shape == (3,)


def test_forward_shape_continuous_with_bounds(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=False, n=3, bounds=(-1.0, 1.0)))
    state = brain.init_state(2, cpu_device)
    action, _, _ = brain.forward(_obs(2, cpu_device), state)
    assert action.shape == (2, 3)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)


def test_state_recurrence_kv_cache_grows(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    s0 = brain.init_state(2, cpu_device)
    assert s0["k"].shape[2] == 0  # empty cache
    obs = _obs(2, cpu_device)
    _, s1, _ = brain.forward(obs, s0)
    _, s2, _ = brain.forward(obs, s1)
    # Acceptance test: s2 != s1 — both shape and content differ.
    assert s1["k"].shape[2] == 1
    assert s2["k"].shape[2] == 2
    # Same prefix preserved (top-of-stack invariant).
    assert torch.allclose(s2["k"][:, :, :1], s1["k"])


def test_state_truncates_at_max_seq_len(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(1, cpu_device)
    obs = _obs(1, cpu_device)
    for _ in range(brain.max_seq_len + 3):
        _, state, _ = brain.forward(obs, state)
    assert state["k"].shape[2] == brain.max_seq_len
    assert state["v"].shape[2] == brain.max_seq_len


def test_gradients_flow_to_all_trainable_parameters(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    _, _, aux = brain.forward(_obs(2, cpu_device), state)
    loss = aux["logits"].sum() + aux["value"].sum()
    loss.backward()
    params = list(brain.trainable_parameters())
    assert len(params) > 0
    for p in params:
        assert p.grad is not None, "every trainable parameter must receive a gradient"
        assert torch.isfinite(p.grad).all()


def test_init_state_batch_and_device(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    s = brain.init_state(7, cpu_device)
    assert s["k"].shape == (brain.n_layers, 7, 0, brain.n_heads, brain.head_dim)
    assert s["length"].shape == (7,) and int(s["length"].sum()) == 0
    assert s["k"].device.type == cpu_device.type


def test_invalid_construction_raises() -> None:
    with pytest.raises(ValueError):
        TransformerBrain(_obs_spec(), ActionSpec(discrete=True, n=3), hidden_dim=15, n_heads=4)
    with pytest.raises(ValueError):
        TransformerBrain(ObsSpec(fields={}), ActionSpec(discrete=True, n=3))
    with pytest.raises(ValueError):
        TransformerBrain(_obs_spec(), ActionSpec(discrete=True, n=0))


def test_forward_does_not_mutate_input_state(cpu_device) -> None:
    brain = _make(ActionSpec(discrete=True, n=4))
    state = brain.init_state(2, cpu_device)
    obs = _obs(2, cpu_device)
    _, state1, _ = brain.forward(obs, state)
    # Step further from state1 then check the original state object is untouched.
    _, _, _ = brain.forward(obs, state1)
    assert state["k"].shape[2] == 0
    assert state1["k"].shape[2] == 1

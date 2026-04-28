"""Unit tests for the Phase-2 ModularMorphology.

Covers all acceptance tests from ``src/evolux/morphology/SPEC.md``:

- Sum of ``s.output_dim for s in sensors`` matches encoder feature dim.
- Sum of ``a.input_dim for a in actuators`` matches action dim.
- ``init_body_state(B=8, device=cpu)`` returns dict of (B, ...) tensors on cpu.
- ``from_dict(m.to_dict())`` produces an identical body (same segment count,
  same sensor/actuator names per segment).
- Empty body (single root segment, no sensors/actuators) initialises without
  crashing.
"""

from __future__ import annotations

import torch

from evolux.core.protocols import Morphology
from evolux.morphology import MORPHOLOGY_REGISTRY
from evolux.morphology.modular import ModularMorphology, Segment

# ── Helpers ───────────────────────────────────────────────────────────────────


def _two_segment_body() -> ModularMorphology:
    """Return a simple two-segment body with known sensors/actuators."""
    segments = [
        Segment(parent_idx=-1, length=1.0, sensors=["vision", "proprio"], actuators=["forward"]),
        Segment(parent_idx=0, length=0.5, sensors=[], actuators=["turn"]),
    ]
    return ModularMorphology(segments)


# ── Protocol compliance ───────────────────────────────────────────────────────


def test_modular_morphology_implements_morphology_protocol() -> None:
    morph = _two_segment_body()
    assert isinstance(morph, Morphology)


# ── Registry ──────────────────────────────────────────────────────────────────


def test_modular_morphology_registered() -> None:
    assert "modular" in MORPHOLOGY_REGISTRY
    assert MORPHOLOGY_REGISTRY.get("modular") is ModularMorphology


# ── Sensor / actuator dimension sums (SPEC acceptance tests) ─────────────────


def test_sensor_output_dim_sum_equals_obs_dim() -> None:
    morph = _two_segment_body()
    computed = sum(s.output_dim for s in morph.sensors)
    assert computed == morph.obs_dim


def test_actuator_input_dim_sum_equals_action_dim() -> None:
    morph = _two_segment_body()
    computed = sum(a.input_dim for a in morph.actuators)
    assert computed == morph.action_dim


def test_sensor_names_in_order() -> None:
    morph = _two_segment_body()
    names = [s.name for s in morph.sensors]
    # Segment 0 has vision + proprio; segment 1 has no sensors.
    assert names == ["vision", "proprio"]


def test_actuator_names_in_order() -> None:
    morph = _two_segment_body()
    names = [a.name for a in morph.actuators]
    # Segment 0: forward; segment 1: turn.
    assert names == ["forward", "turn"]


def test_obs_dim_equals_sum_of_sensor_dims() -> None:
    morph = _two_segment_body()
    # vision=100, proprio=3 → 103
    assert morph.obs_dim == 103


def test_action_dim_equals_sum_of_actuator_dims() -> None:
    morph = _two_segment_body()
    # forward=1, turn=1 → 2
    assert morph.action_dim == 2


# ── init_body_state (SPEC acceptance test) ────────────────────────────────────


def test_init_body_state_keys(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    state = morph.init_body_state(8, cpu_device)
    assert "pos" in state
    assert "joint_angle" in state
    assert "energy" in state


def test_init_body_state_shapes(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    B = 8
    n = morph.n_segments  # 2
    state = morph.init_body_state(B, cpu_device)

    assert state["pos"].shape == (B, n, 2)
    assert state["joint_angle"].shape == (B, n)
    assert state["energy"].shape == (B,)


def test_init_body_state_on_cpu(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    state = morph.init_body_state(4, cpu_device)
    for v in state.values():
        assert v.device.type == "cpu"


def test_init_body_state_energy_full(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    state = morph.init_body_state(8, cpu_device)
    assert torch.all(state["energy"] == 1.0)


def test_init_body_state_pos_zeros(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    state = morph.init_body_state(8, cpu_device)
    assert torch.all(state["pos"] == 0.0)


def test_init_body_state_joint_angle_zeros(cpu_device: torch.device) -> None:
    morph = _two_segment_body()
    state = morph.init_body_state(8, cpu_device)
    assert torch.all(state["joint_angle"] == 0.0)


# ── Serialise / deserialise round-trip (SPEC acceptance test) ─────────────────


def test_round_trip_segment_count() -> None:
    morph = _two_segment_body()
    morph2 = ModularMorphology.from_dict(morph.to_dict())
    assert morph2.n_segments == morph.n_segments


def test_round_trip_sensor_names_per_segment() -> None:
    morph = _two_segment_body()
    d = morph.to_dict()
    morph2 = ModularMorphology.from_dict(d)

    for seg_orig, seg_new in zip(morph._segments, morph2._segments, strict=True):
        assert seg_new.sensors == seg_orig.sensors


def test_round_trip_actuator_names_per_segment() -> None:
    morph = _two_segment_body()
    d = morph.to_dict()
    morph2 = ModularMorphology.from_dict(d)

    for seg_orig, seg_new in zip(morph._segments, morph2._segments, strict=True):
        assert seg_new.actuators == seg_orig.actuators


def test_round_trip_parent_idx_preserved() -> None:
    morph = _two_segment_body()
    d = morph.to_dict()
    morph2 = ModularMorphology.from_dict(d)

    for seg_orig, seg_new in zip(morph._segments, morph2._segments, strict=True):
        assert seg_new.parent_idx == seg_orig.parent_idx


def test_round_trip_length_preserved() -> None:
    morph = _two_segment_body()
    d = morph.to_dict()
    morph2 = ModularMorphology.from_dict(d)

    for seg_orig, seg_new in zip(morph._segments, morph2._segments, strict=True):
        assert seg_new.length == seg_orig.length


def test_to_dict_type_field() -> None:
    morph = _two_segment_body()
    assert morph.to_dict()["type"] == "modular"


def test_from_dict_returns_modular_morphology() -> None:
    morph = _two_segment_body()
    morph2 = ModularMorphology.from_dict(morph.to_dict())
    assert isinstance(morph2, ModularMorphology)


# ── Empty body (single root, no sensors/actuators) ────────────────────────────


def test_empty_body_initialises_without_crash(cpu_device: torch.device) -> None:
    """Single root segment with no sensors or actuators must not crash."""
    morph = ModularMorphology([Segment(parent_idx=-1, length=1.0)])
    state = morph.init_body_state(4, cpu_device)
    assert state["pos"].shape == (4, 1, 2)
    assert state["joint_angle"].shape == (4, 1)
    assert state["energy"].shape == (4,)


def test_empty_body_obs_dim_zero() -> None:
    morph = ModularMorphology([Segment(parent_idx=-1, length=1.0)])
    assert morph.obs_dim == 0
    assert morph.action_dim == 0


def test_empty_body_sensors_actuators_empty() -> None:
    morph = ModularMorphology([Segment(parent_idx=-1, length=1.0)])
    assert morph.sensors == []
    assert morph.actuators == []


# ── Multi-segment body ────────────────────────────────────────────────────────


def test_n_segments_matches_input() -> None:
    segments = [
        Segment(parent_idx=-1, length=1.0),
        Segment(parent_idx=0, length=0.5),
        Segment(parent_idx=0, length=0.5),
    ]
    morph = ModularMorphology(segments)
    assert morph.n_segments == 3


def test_sensors_accumulated_across_segments() -> None:
    segments = [
        Segment(parent_idx=-1, length=1.0, sensors=["vision"]),
        Segment(parent_idx=0, length=0.5, sensors=["proprio"]),
    ]
    morph = ModularMorphology(segments)
    assert len(morph.sensors) == 2
    assert morph.sensors[0].name == "vision"
    assert morph.sensors[1].name == "proprio"

"""Unit tests for the Phase-1 morphology module.

Covers all acceptance tests from ``src/evolux/morphology/SPEC.md``:

- Sum of ``s.output_dim for s in sensors`` matches encoder feature dim.
- Sum of ``a.input_dim for a in actuators`` matches action dim.
- ``init_body_state(B, device)`` returns a dict of (B, ...) tensors on device.
- Serialise/deserialise round-trip (dict).

Also verifies Protocol compliance and registry registration.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from evolux.core.protocols import Actuator, Morphology, Sensor
from evolux.morphology import MORPHOLOGY_REGISTRY
from evolux.morphology.simple import (
    MotorActuator,
    SimpleMorphology,
    SimpleProprioSensor,
    SimpleVisionSensor,
)

# ── Protocol compliance ───────────────────────────────────────────────────────


def test_vision_sensor_implements_sensor_protocol() -> None:
    sensor = SimpleVisionSensor()
    assert isinstance(sensor, Sensor)


def test_proprio_sensor_implements_sensor_protocol() -> None:
    sensor = SimpleProprioSensor()
    assert isinstance(sensor, Sensor)


def test_motor_actuator_implements_actuator_protocol() -> None:
    actuator = MotorActuator("forward")
    assert isinstance(actuator, Actuator)


def test_simple_morphology_implements_morphology_protocol() -> None:
    morph = SimpleMorphology()
    assert isinstance(morph, Morphology)


# ── Registry ──────────────────────────────────────────────────────────────────


def test_simple_morphology_registered() -> None:
    assert "simple_v1" in MORPHOLOGY_REGISTRY
    assert MORPHOLOGY_REGISTRY.get("simple_v1") is SimpleMorphology


# ── Sensor / actuator dimension sums (SPEC acceptance tests) ─────────────────


def test_sensor_output_dim_sum_equals_obs_dim() -> None:
    morph = SimpleMorphology()
    computed = sum(s.output_dim for s in morph.sensors)
    assert computed == morph.obs_dim


def test_actuator_input_dim_sum_equals_action_dim() -> None:
    morph = SimpleMorphology()
    computed = sum(a.input_dim for a in morph.actuators)
    assert computed == morph.action_dim


def test_sensor_count_and_names() -> None:
    morph = SimpleMorphology()
    names = [s.name for s in morph.sensors]
    assert names == ["vision", "proprio"]


def test_actuator_count_and_names() -> None:
    morph = SimpleMorphology()
    names = [a.name for a in morph.actuators]
    assert names == ["forward", "turn"]


def test_default_obs_dim() -> None:
    """Default sensor dims sum to 103 (100 vision + 3 proprio)."""
    morph = SimpleMorphology()
    assert morph.obs_dim == 100 + 3


def test_default_action_dim() -> None:
    """Two MotorActuators each with input_dim=1 → action_dim == 2."""
    morph = SimpleMorphology()
    assert morph.action_dim == 2


def test_custom_dims_reflected_in_obs_dim() -> None:
    morph = SimpleMorphology(vision_output_dim=64, proprio_output_dim=8)
    assert morph.obs_dim == 72


# ── init_body_state (SPEC acceptance test) ────────────────────────────────────


def test_init_body_state_shapes(cpu_device: torch.device) -> None:
    morph = SimpleMorphology()
    B = 8
    state = morph.init_body_state(B, cpu_device)

    assert "position" in state
    assert "heading" in state
    assert "energy" in state

    assert state["position"].shape == (B, 2)
    assert state["heading"].shape == (B,)
    assert state["energy"].shape == (B,)


def test_init_body_state_on_cpu_device(cpu_device: torch.device) -> None:
    morph = SimpleMorphology()
    state = morph.init_body_state(4, cpu_device)
    for v in state.values():
        assert v.device.type == "cpu"


def test_init_body_state_energy_full(cpu_device: torch.device) -> None:
    morph = SimpleMorphology()
    state = morph.init_body_state(16, cpu_device)
    assert torch.all(state["energy"] == 1.0)


def test_init_body_state_position_zero(cpu_device: torch.device) -> None:
    morph = SimpleMorphology()
    state = morph.init_body_state(4, cpu_device)
    assert torch.all(state["position"] == 0.0)


# ── Serialise / deserialise round-trip (SPEC acceptance test) ─────────────────


def test_serialize_deserialize_defaults() -> None:
    morph = SimpleMorphology()
    d = morph.to_dict()
    morph2 = SimpleMorphology.from_dict(d)

    assert morph2.obs_dim == morph.obs_dim
    assert morph2.action_dim == morph.action_dim


def test_serialize_deserialize_custom_dims() -> None:
    morph = SimpleMorphology(vision_output_dim=64, proprio_output_dim=5)
    d = morph.to_dict()
    morph2 = SimpleMorphology.from_dict(d)

    assert morph2.sensors[0].output_dim == 64
    assert morph2.sensors[1].output_dim == 5


def test_to_dict_type_field() -> None:
    morph = SimpleMorphology()
    d = morph.to_dict()
    assert d["type"] == "simple_v1"


def test_from_dict_returns_simple_morphology() -> None:
    morph = SimpleMorphology(vision_output_dim=32, proprio_output_dim=4)
    d = morph.to_dict()
    morph2 = SimpleMorphology.from_dict(d)
    assert isinstance(morph2, SimpleMorphology)


# ── sense() and act() smoke tests (via mock World) ───────────────────────────


def _make_mock_world(batch_size: int = 4) -> MagicMock:
    """Build a mock implementing the World Protocol for test purposes."""
    world = MagicMock()
    world.batch_size = batch_size
    world.device = torch.device("cpu")
    world.observe.return_value = {
        "vision": torch.ones(batch_size, 4, 5, 5),  # (B, C, H, W)
        "proprio": torch.ones(batch_size, 3),
    }
    return world


def test_vision_sensor_sense_shape() -> None:
    sensor = SimpleVisionSensor(output_dim=100)
    world = _make_mock_world(batch_size=4)
    out = sensor.sense(world, {})
    assert out.shape == (4, 100)


def test_proprio_sensor_sense_shape() -> None:
    sensor = SimpleProprioSensor(output_dim=3)
    world = _make_mock_world(batch_size=4)
    out = sensor.sense(world, {})
    assert out.shape == (4, 3)


def test_vision_sensor_fallback_when_key_missing() -> None:
    sensor = SimpleVisionSensor(output_dim=50)
    world = MagicMock()
    world.batch_size = 2
    world.device = torch.device("cpu")
    world.observe.return_value = {}  # no 'vision' key
    out = sensor.sense(world, {})
    assert out.shape == (2, 50)
    assert torch.all(out == 0.0)


def test_proprio_sensor_fallback_when_key_missing() -> None:
    sensor = SimpleProprioSensor(output_dim=3)
    world = MagicMock()
    world.batch_size = 2
    world.device = torch.device("cpu")
    world.observe.return_value = {}  # no 'proprio' key
    out = sensor.sense(world, {})
    assert out.shape == (2, 3)
    assert torch.all(out == 0.0)


def test_motor_actuator_act_returns_correct_key() -> None:
    actuator = MotorActuator("forward", input_dim=1)
    world = MagicMock()
    body_state: dict = {}
    command = torch.tensor([[0.8], [0.3]])
    result = actuator.act(world, body_state, command)
    assert "forward" in result
    assert torch.equal(result["forward"], command)


def test_motor_actuator_turn_key() -> None:
    actuator = MotorActuator("turn", input_dim=1)
    world = MagicMock()
    command = torch.tensor([[1.0]])
    result = actuator.act(world, {}, command)
    assert "turn" in result

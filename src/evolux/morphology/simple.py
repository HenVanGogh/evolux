"""Phase-1 fixed body — SimpleMorphology with two sensors and two motor actuators.

Implements the ``Morphology``, ``Sensor``, and ``Actuator`` Protocols from
``evolux.core.protocols`` with a minimal hard-coded body plan:

- 1 vision sensor  (``SimpleVisionSensor``)
- 1 proprio sensor (``SimpleProprioSensor``)
- 2 motor actuators — *forward* and *turn* (``MotorActuator``)

This is the default body for Phase-1 creatures.  Phase-2 modular bodies live
in ``modular.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from evolux.core.types import BodyState

if TYPE_CHECKING:
    from evolux.core.protocols import World

logger = logging.getLogger(__name__)

# ── Default sensor dimensions ────────────────────────────────────────────────
_VISION_OUTPUT_DIM: int = 100  # 4 channels x 5x5 local crop, flattened
_PROPRIO_OUTPUT_DIM: int = 3  # [energy, hunger, last_action_norm]


# ── Sensors ──────────────────────────────────────────────────────────────────


class SimpleVisionSensor:
    """Extracts a flattened local-vision crop from the world observation.

    Calls ``world.observe()`` and reads the ``"vision"`` key (expected shape
    ``(B, C, H, W)``).  Flattens spatial dims and slices to ``output_dim``.
    Falls back to zeros when the key is absent.
    """

    name: str = "vision"

    def __init__(self, output_dim: int = _VISION_OUTPUT_DIM) -> None:
        self.output_dim: int = output_dim

    def sense(self, world: World, body_state: BodyState) -> Tensor:
        """Return ``(B, output_dim)`` vision features."""
        obs = world.observe()
        if "vision" in obs:
            v: Tensor = obs["vision"]  # (B, C, H, W)
            flat = v.flatten(start_dim=1)  # (B, C*H*W)
            dim = min(self.output_dim, flat.shape[1])
            out = flat.new_zeros(flat.shape[0], self.output_dim)
            out[:, :dim] = flat[:, :dim]
            return out
        logger.warning("SimpleVisionSensor: 'vision' key missing from world.observe().")
        return torch.zeros(world.batch_size, self.output_dim, device=world.device)


class SimpleProprioSensor:
    """Extracts proprioception features from the world observation.

    Calls ``world.observe()`` and reads the ``"proprio"`` key (expected shape
    ``(B, P)``).  Slices or zero-pads to ``output_dim``.
    Falls back to zeros when the key is absent.
    """

    name: str = "proprio"

    def __init__(self, output_dim: int = _PROPRIO_OUTPUT_DIM) -> None:
        self.output_dim: int = output_dim

    def sense(self, world: World, body_state: BodyState) -> Tensor:
        """Return ``(B, output_dim)`` proprioception features."""
        obs = world.observe()
        if "proprio" in obs:
            p: Tensor = obs["proprio"]  # (B, P)
            dim = min(self.output_dim, p.shape[1])
            out = p.new_zeros(p.shape[0], self.output_dim)
            out[:, :dim] = p[:, :dim]
            return out
        logger.warning("SimpleProprioSensor: 'proprio' key missing from world.observe().")
        return torch.zeros(world.batch_size, self.output_dim, device=world.device)


# ── Actuators ────────────────────────────────────────────────────────────────


class MotorActuator:
    """Single-axis motor actuator (e.g. *forward* or *turn*).

    ``input_dim`` defaults to 1 (one scalar command per actuator).
    ``act`` returns the command tensor under the actuator's name so that
    downstream physics can consume it by key.
    """

    def __init__(self, name: str, input_dim: int = 1) -> None:
        self.name: str = name
        self.input_dim: int = input_dim

    def act(
        self,
        world: World,
        body_state: BodyState,
        command: Tensor,
    ) -> dict[str, Tensor]:
        """Return ``{name: command}`` for downstream physics."""
        return {self.name: command}


# ── Morphology ───────────────────────────────────────────────────────────────


class SimpleMorphology:
    """Fixed body for Phase-1 creatures.

    Body plan:
    - sensors  : [SimpleVisionSensor, SimpleProprioSensor]
    - actuators: [MotorActuator("forward"), MotorActuator("turn")]

    ``init_body_state`` returns a ``BodyState`` dict with keys
    ``"position"`` (B, 2), ``"heading"`` (B,), ``"energy"`` (B,),
    all on the requested device.

    Serialisation is a plain ``dict`` round-trip — sufficient for Phase 1.
    """

    def __init__(
        self,
        vision_output_dim: int = _VISION_OUTPUT_DIM,
        proprio_output_dim: int = _PROPRIO_OUTPUT_DIM,
    ) -> None:
        self.sensors: list[SimpleVisionSensor | SimpleProprioSensor] = [
            SimpleVisionSensor(vision_output_dim),
            SimpleProprioSensor(proprio_output_dim),
        ]
        self.actuators: list[MotorActuator] = [
            MotorActuator("forward"),
            MotorActuator("turn"),
        ]

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def obs_dim(self) -> int:
        """Total encoder input dim — sum of all sensor ``output_dim`` values."""
        return sum(s.output_dim for s in self.sensors)

    @property
    def action_dim(self) -> int:
        """Total action dim — sum of all actuator ``input_dim`` values."""
        return sum(a.input_dim for a in self.actuators)

    # ── Protocol method ───────────────────────────────────────────────────────

    def init_body_state(self, batch_size: int, device: torch.device) -> BodyState:
        """Initialise body state tensors on *device*.

        Returns
        -------
        dict with keys:
            ``"position"``  — ``(B, 2)`` float32, zeros (origin).
            ``"heading"``   — ``(B,)``   float32, zeros (facing right).
            ``"energy"``    — ``(B,)``   float32, ones  (full energy).
        """
        return {
            "position": torch.zeros(batch_size, 2, device=device),
            "heading": torch.zeros(batch_size, device=device),
            "energy": torch.ones(batch_size, device=device),
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict (Phase-1 round-trip)."""
        vision_sensor = next((s for s in self.sensors if s.name == "vision"), None)
        proprio_sensor = next((s for s in self.sensors if s.name == "proprio"), None)
        if vision_sensor is None or proprio_sensor is None:
            raise ValueError("SimpleMorphology: expected 'vision' and 'proprio' sensors.")
        return {
            "type": "simple_v1",
            "vision_output_dim": vision_sensor.output_dim,
            "proprio_output_dim": proprio_sensor.output_dim,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SimpleMorphology:
        """Deserialise from a plain dict produced by :meth:`to_dict`."""
        return cls(
            vision_output_dim=int(data["vision_output_dim"]),  # type: ignore[arg-type]
            proprio_output_dim=int(data["proprio_output_dim"]),  # type: ignore[arg-type]
        )

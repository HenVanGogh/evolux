"""Phase-2 modular morphology — tree of segments with per-segment sensors/actuators.

Implements the ``Morphology`` Protocol from ``evolux.core.protocols`` with a
body plan encoded as a **segment tree**: each segment can carry zero or more
named sensors and actuators drawn from the built-in catalogue below.

References
----------
Sims, *Evolving Virtual Creatures* (1994).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from evolux.core.protocols import Actuator, Sensor
from evolux.core.types import BodyState
from evolux.morphology.simple import MotorActuator, SimpleProprioSensor, SimpleVisionSensor

logger = logging.getLogger(__name__)

# ── Built-in sensor / actuator catalogue ─────────────────────────────────────
# Maps a string name → factory callable that returns a new instance.
# Sensor entries are classes (no required arguments).
# Actuator entries are lambdas that pass the catalogue key as the name.
# Extend here when new sensor/actuator types are added.

_SENSOR_CATALOGUE: dict[str, Any] = {
    "vision": SimpleVisionSensor,
    "proprio": SimpleProprioSensor,
}

_ACTUATOR_CATALOGUE: dict[str, Any] = {
    "forward": lambda: MotorActuator("forward"),
    "turn": lambda: MotorActuator("turn"),
}


def _make_sensor(name: str) -> Sensor:
    """Instantiate a sensor by catalogue name.

    Parameters
    ----------
    name:
        Catalogue key (e.g. ``"vision"``, ``"proprio"``).

    Raises
    ------
    KeyError
        If the name is not in the built-in catalogue.
    """
    if name not in _SENSOR_CATALOGUE:
        raise KeyError(f"Unknown sensor '{name}'. Available: {sorted(_SENSOR_CATALOGUE)}")
    cls = _SENSOR_CATALOGUE[name]
    return cls()


def _make_actuator(name: str) -> Actuator:
    """Instantiate an actuator by catalogue name.

    Parameters
    ----------
    name:
        Catalogue key (e.g. ``"forward"``, ``"turn"``).

    Raises
    ------
    KeyError
        If the name is not in the built-in catalogue.
    """
    if name not in _ACTUATOR_CATALOGUE:
        raise KeyError(f"Unknown actuator '{name}'. Available: {sorted(_ACTUATOR_CATALOGUE)}")
    factory = _ACTUATOR_CATALOGUE[name]
    return factory()


# ── Segment dataclass ─────────────────────────────────────────────────────────


@dataclass
class Segment:
    """A single body segment in a creature's morphology tree.

    Parameters
    ----------
    parent_idx:
        Index of the parent segment in the segment list.  Root segment uses
        ``-1`` (no parent).
    length:
        Physical length of the segment (arbitrary units; used for body-state
        initialisation and physics).
    sensors:
        Ordered list of sensor names (catalogue keys).  Each name maps to one
        concrete :class:`~evolux.core.protocols.Sensor` instance.
    actuators:
        Ordered list of actuator names (catalogue keys).  Each name maps to
        one concrete :class:`~evolux.core.protocols.Actuator` instance.
    """

    parent_idx: int
    length: float
    sensors: list[str] = field(default_factory=list)
    actuators: list[str] = field(default_factory=list)


# ── ModularMorphology ─────────────────────────────────────────────────────────


class ModularMorphology:
    """Modular morphology built from a tree of :class:`Segment` objects.

    Body plan
    ---------
    - The body is described by an ordered list of :class:`Segment` instances.
    - Segment 0 is always the **root** (``parent_idx == -1``).
    - Sensors are collected in segment order (then in-segment order).
    - Actuators are collected in the same traversal order.

    ``init_body_state`` returns a :class:`~evolux.core.types.BodyState` dict
    with keys:

    - ``"pos"``          — ``(B, n_segments, 2)``  float32, zeros (2-D plane).
    - ``"joint_angle"``  — ``(B, n_segments)``      float32, zeros.
    - ``"energy"``       — ``(B,)``                 float32, ones (full energy).

    Serialisation is a plain dict that fully reconstructs the segment tree.
    """

    def __init__(self, segments: list[Segment]) -> None:
        if not segments:
            raise ValueError("ModularMorphology requires at least one segment (root).")

        self._segments: list[Segment] = segments

        # Collect sensors and actuators across all segments (in order).
        self.sensors: list[Sensor] = []
        self.actuators: list[Actuator] = []

        for seg in segments:
            for sensor_name in seg.sensors:
                self.sensors.append(_make_sensor(sensor_name))
            for actuator_name in seg.actuators:
                self.actuators.append(_make_actuator(actuator_name))

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_segments(self) -> int:
        """Number of segments in the body tree."""
        return len(self._segments)

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
        """Initialise body-state tensors on *device*.

        Returns
        -------
        dict with keys:
            ``"pos"``         — ``(B, n_segments, 2)`` float32, zeros.
            ``"joint_angle"`` — ``(B, n_segments)``    float32, zeros.
            ``"energy"``      — ``(B,)``               float32, ones.
        """
        n = self.n_segments
        return {
            "pos": torch.zeros(batch_size, n, 2, device=device),
            "joint_angle": torch.zeros(batch_size, n, device=device),
            "energy": torch.ones(batch_size, device=device),
        }

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialise the morphology to a plain dict.

        The dict is fully self-describing and can be passed to
        :meth:`from_dict` to reconstruct an identical ``ModularMorphology``.
        """
        return {
            "type": "modular",
            "segments": [
                {
                    "parent_idx": seg.parent_idx,
                    "length": seg.length,
                    "sensors": list(seg.sensors),
                    "actuators": list(seg.actuators),
                }
                for seg in self._segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModularMorphology:
        """Reconstruct a :class:`ModularMorphology` from a serialised dict.

        Parameters
        ----------
        data:
            Dict produced by :meth:`to_dict`.
        """
        segments = [
            Segment(
                parent_idx=int(s["parent_idx"]),
                length=float(s["length"]),
                sensors=list(s["sensors"]),
                actuators=list(s["actuators"]),
            )
            for s in data["segments"]
        ]
        return cls(segments)

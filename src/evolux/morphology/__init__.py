"""morphology — evolved bodies with modular limbs, sensors, and actuators."""

from __future__ import annotations

from evolux.core.registry import Registry
from evolux.morphology.modular import ModularMorphology, Segment
from evolux.morphology.simple import (
    MotorActuator,
    SimpleMorphology,
    SimpleProprioSensor,
    SimpleVisionSensor,
)

MORPHOLOGY_REGISTRY: Registry = Registry("morphology")
MORPHOLOGY_REGISTRY.add("simple_v1", SimpleMorphology)
MORPHOLOGY_REGISTRY.add("modular", ModularMorphology)

__all__ = [
    "MORPHOLOGY_REGISTRY",
    "ModularMorphology",
    "MotorActuator",
    "Segment",
    "SimpleMorphology",
    "SimpleProprioSensor",
    "SimpleVisionSensor",
]

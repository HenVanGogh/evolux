"""Foundation layer: types, protocols, config, RNG, device, registry.

This module has zero internal dependencies. Everything else builds on it.
"""

from evolux.core.config import BaseConfig, ConfigError, load_config
from evolux.core.device import DeviceManager, get_device
from evolux.core.registry import Registry
from evolux.core.rng import RNG
from evolux.core.types import (
    Action,
    ActionSpec,
    AuxInfo,
    BodyState,
    BrainState,
    Done,
    MemoryState,
    Obs,
    ObsSpec,
    Reward,
    StateSpec,
    Tensor,
    Trajectory,
    WorldState,
)

__all__ = [
    "RNG",
    "Action",
    "ActionSpec",
    "AuxInfo",
    "BaseConfig",
    "BodyState",
    "BrainState",
    "ConfigError",
    "DeviceManager",
    "Done",
    "MemoryState",
    "Obs",
    "ObsSpec",
    "Registry",
    "Reward",
    "StateSpec",
    "Tensor",
    "Trajectory",
    "WorldState",
    "get_device",
    "load_config",
]

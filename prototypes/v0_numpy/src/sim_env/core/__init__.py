"""Core package: simulation loop, world, registry."""

from sim_env.core.simulation import Simulation
from sim_env.core.world import World
from sim_env.core.registry import ComponentRegistry

__all__ = ["Simulation", "World", "ComponentRegistry"]

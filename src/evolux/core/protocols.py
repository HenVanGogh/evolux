"""Cross-module Protocol definitions.

These are the **only** types other modules should depend on for
inter-module communication. See ``docs/INTERFACES.md`` for the full
human-readable contract.

Protocols are versioned: breaking changes follow the policy in
``docs/INTERFACES.md`` (rename + alias + CHANGELOG entry).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

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
    Trajectory,
)

PROTOCOL_VERSION = "1.0.0"


# ── Brain ──────────────────────────────────────────────────────────────────


@runtime_checkable
class Brain(Protocol):
    """Stateful policy mapping observations to actions.

    All inputs and outputs have leading batch dim ``B``.
    """

    obs_spec: ObsSpec
    action_spec: ActionSpec
    state_spec: StateSpec

    def init_state(self, batch_size: int, device: torch.device) -> BrainState: ...

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]: ...

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...


@runtime_checkable
class Imaginer(Protocol):
    """Optional capability: generate imagined trajectories (Dreamer-style)."""

    def imagine(self, obs: Obs, state: BrainState, horizon: int) -> Trajectory: ...


# ── Memory ─────────────────────────────────────────────────────────────────


@runtime_checkable
class Memory(Protocol):
    """Differentiable memory bank (working / episodic / semantic / Hebbian)."""

    capacity: int
    key_dim: int
    val_dim: int

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState: ...

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState: ...

    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor: ...

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState: ...


# ── Genome ─────────────────────────────────────────────────────────────────


@runtime_checkable
class Genome(Protocol):
    """Heritable specification of a creature's brain structure and parameters.

    A Genome encodes how to build a :class:`Brain` and :class:`Morphology` and
    can be serialised for checkpointing or cross-process transfer.
    """

    encoding: str

    def decode_brain(self, brain_factory: BrainFactory) -> Brain: ...

    def decode_morphology(self) -> Morphology: ...

    def serialize(self) -> bytes: ...


@runtime_checkable
class BrainFactory(Protocol):
    """Builds a Brain from architectural hyperparameters."""

    def build(self, obs_spec: ObsSpec, action_spec: ActionSpec, hparams: dict) -> Brain: ...


@runtime_checkable
class GenomeOperator(Protocol):
    """Genetic operators: mutation, crossover, and distance for a genome encoding."""

    def mutate(self, g: Genome, rng: torch.Generator) -> Genome: ...

    def crossover(self, a: Genome, b: Genome, rng: torch.Generator) -> Genome: ...

    def distance(self, a: Genome, b: Genome) -> float: ...


# ── Morphology / sensors / actuators ───────────────────────────────────────


@runtime_checkable
class Sensor(Protocol):
    """A single perception modality that reads raw signals from the world.

    Sensors are attached to a :class:`Morphology` and convert world state +
    body state into a fixed-size feature tensor consumed by the brain.
    """

    name: str
    output_dim: int

    def sense(self, world: World, body_state: BodyState) -> Tensor: ...


@runtime_checkable
class Actuator(Protocol):
    """A motor channel that translates brain commands into world effects.

    Actuators are attached to a :class:`Morphology` and consume a slice of the
    action tensor, producing a dict of state updates applied to the world.
    """

    name: str
    input_dim: int

    def act(self, world: World, body_state: BodyState, command: Tensor) -> dict[str, Tensor]: ...


@runtime_checkable
class Morphology(Protocol):
    """Physical body specification: sensors, actuators, and body-state layout.

    A Morphology owns the lists of :class:`Sensor` and :class:`Actuator` that
    define what a creature can perceive and do, and provides the initial body
    state for a batch of creatures.
    """

    sensors: list[Sensor]
    actuators: list[Actuator]

    def init_body_state(self, batch_size: int, device: torch.device) -> BodyState: ...


# ── World / environment ────────────────────────────────────────────────────


@runtime_checkable
class World(Protocol):
    """Batched simulation environment that creatures inhabit.

    A World manages the physical state of ``B`` parallel environments.  It is
    reset per generation, stepped each tick, and observed by creatures via
    their sensors.
    """

    batch_size: int
    device: torch.device

    def reset(self, mask: Tensor | None = None) -> None: ...

    def observe(self) -> Obs: ...

    def step(self, action: Action) -> tuple[Reward, Done, AuxInfo]: ...


@runtime_checkable
class Environment(Protocol):
    """Wraps a World with seasons / ecology / procedural dynamics."""

    world: World

    def step(self) -> None: ...

    def fields(self) -> dict[str, Tensor]: ...


# ── Fitness / objectives ───────────────────────────────────────────────────


@runtime_checkable
class Objective(Protocol):
    """A single fitness signal extracted from a creature's trajectory.

    Multiple Objectives are combined by a :class:`FitnessAggregator` into a
    scalar fitness value used by the evolutionary selector.
    """

    name: str
    weight: float
    higher_is_better: bool

    def evaluate(self, trajectory: Trajectory, world: World) -> Tensor:
        """Return a (B,) tensor of per-creature scores."""


@runtime_checkable
class FitnessAggregator(Protocol):
    """Combines per-objective scores into a scalar fitness for each creature."""

    objectives: list[Objective]

    def aggregate(self, scores: dict[str, Tensor]) -> Tensor: ...


# ── Evolution ──────────────────────────────────────────────────────────────


@runtime_checkable
class Selector(Protocol):
    """Chooses parent genomes from a population for the next generation."""

    def select_parents(self, pop: Population, n: int, rng: torch.Generator) -> list[Genome]: ...

    def get_elites(self, pop: Population) -> list[Genome]: ...


@runtime_checkable
class Population(Protocol):
    """Container for the current generation of genomes and their fitness scores."""

    genomes: list[Genome]
    fitness: Tensor  # (P,)
    behaviour: Tensor | None  # (P, D_b) for novelty / MAP-Elites


@runtime_checkable
class BehaviourDescriptor(Protocol):
    """Maps a trajectory to a behavioural feature vector for novelty / MAP-Elites."""

    dim: int

    def describe(self, trajectory: Trajectory) -> Tensor: ...


# ── Logging / rendering ────────────────────────────────────────────────────


@runtime_checkable
class StatsLogger(Protocol):
    """Structured logging sink for scalars, dicts, and images."""

    def log_scalar(self, key: str, value: float, step: int) -> None: ...

    def log_dict(self, data: dict, step: int) -> None: ...

    def log_image(self, key: str, image: Tensor, step: int) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class Renderer(Protocol):
    """Produces a visual frame tensor from the current world state."""

    def render(self, world: World, env_idx: int = 0) -> Tensor: ...

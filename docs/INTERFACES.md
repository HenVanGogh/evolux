# Cross-module Interfaces

All cross-module interaction goes through `typing.Protocol` definitions in `src/evolux/core/protocols.py`. Modules **must** depend on Protocols, never on concrete classes from sibling modules.

This document is the canonical reference. The Protocols themselves carry docstrings; this file is the human-friendly explanation.

---

## 1. `Brain`

```python
class Brain(Protocol):
    obs_spec:    ObsSpec
    action_spec: ActionSpec
    state_spec:  StateSpec

    def init_state(self, batch_size: int, device: torch.device) -> BrainState: ...
    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]: ...
    def imagine(self, obs: Obs, state: BrainState, horizon: int) -> Trajectory: ...
        # Optional. Default: raises NotImplementedError. Implemented by world-model brains.
    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]: ...
        # Empty iterator for pure-evolution brains.
```

**Invariants**
- `forward` is **pure** w.r.t. `state`: returns new state, does not mutate input.
- All inputs/outputs leading dim is `B`.
- `obs_spec.batch_invariant == True` ⇒ shapes do not depend on `B`.

---

## 2. `Memory`

```python
class Memory(Protocol):
    capacity: int
    key_dim:  int
    val_dim:  int

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState: ...
    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState: ...
    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor: ...
    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState: ...
```

`mask` of shape `(B,)` allows partial-batch episode resets (some envs done, others not).

---

## 3. `Genome`

```python
class Genome(Protocol):
    encoding: str   # "direct" | "hyperneat" | "indirect"

    def decode_brain(self, brain_factory: BrainFactory) -> Brain: ...
    def decode_morphology(self) -> Morphology: ...
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, blob: bytes) -> "Genome": ...

class GenomeOperator(Protocol):
    def mutate(self, g: Genome, rng: torch.Generator) -> Genome: ...
    def crossover(self, a: Genome, b: Genome, rng: torch.Generator) -> Genome: ...
    def distance(self, a: Genome, b: Genome) -> float: ...   # for speciation
```

---

## 4. `World` and `Environment`

```python
class World(Protocol):
    """Vectorised physical state for B parallel envs."""
    batch_size: int
    device: torch.device

    def reset(self, mask: Tensor | None = None) -> WorldState: ...
    def observe(self) -> Obs: ...
    def step(self, action: Action) -> tuple[Reward, Done, Info]: ...

class Environment(Protocol):
    """Wraps World with seasons / ecology / procedural dynamics."""
    world: World
    def step(self) -> None: ...
    def fields(self) -> dict[str, Tensor]: ...   # food, hazard, climate, etc.
```

`reset(mask)` allows per-env reset on done flags — critical for Dreamer-style continuous training.

---

## 5. `Fitness`

```python
class Objective(Protocol):
    name: str
    weight: float
    higher_is_better: bool
    def evaluate(self, trajectory: Trajectory, world: World) -> Tensor: ...   # (B,)

class FitnessAggregator(Protocol):
    objectives: list[Objective]
    def aggregate(self, scores: dict[str, Tensor]) -> Tensor: ...   # (B,)
```

`Trajectory` is the per-creature record of obs/act/reward over a generation.

---

## 6. `Selector` (evolution)

```python
class Selector(Protocol):
    def select_parents(self, pop: Population, n: int, rng: torch.Generator) -> list[Genome]: ...
    def get_elites(self, pop: Population) -> list[Genome]: ...
```

---

## 7. `Behaviour Descriptor` (novelty / MAP-Elites)

```python
class BehaviourDescriptor(Protocol):
    dim: int
    def describe(self, trajectory: Trajectory) -> Tensor: ...   # (B, dim)
```

---

## 8. `Sensor` / `Actuator` (morphology ↔ world plumbing)

```python
class Sensor(Protocol):
    name: str
    output_dim: int
    def sense(self, world: World, body_state: BodyState) -> Tensor: ...

class Actuator(Protocol):
    name: str
    input_dim: int
    def act(self, world: World, body_state: BodyState, command: Tensor) -> WorldDelta: ...
```

A `Morphology` exposes `sensors: list[Sensor]` and `actuators: list[Actuator]`. The brain's I/O dim is the sum of these. Evolved morphologies plug new sensors in dynamically.

---

## 9. `Renderer` / `StatsLogger`

```python
class StatsLogger(Protocol):
    def log_scalar(self, key: str, value: float, step: int) -> None: ...
    def log_dict(self,   data: dict, step: int) -> None: ...
    def log_image(self,  key: str, image: Tensor, step: int) -> None: ...
    def close(self) -> None: ...

class Renderer(Protocol):
    def render(self, world: World, env_idx: int = 0) -> Tensor: ...   # (H, W, 3)
```

---

## Stability policy

Protocols are **versioned**. Breaking changes require:
1. A `Protocol` rename with a deprecation alias for one minor release.
2. An entry in `docs/CHANGELOG.md`.
3. All in-tree implementations updated in the same PR.

Adding new optional methods (with default raising) is **non-breaking**.

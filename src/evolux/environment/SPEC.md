# environment — SPEC

| Field | Value |
| --- | --- |
| Layer | 4 |
| Phase | 1 (foraging) → 3 (procedural multi-task biomes) |
| Depends on | `core`, `tensors`, `world`, `physics` |
| Implements | `evolux.core.protocols.Environment` |

## Mission

Sit on top of a `World`, define **task semantics**: what spawns, what kills,
when an episode ends, what reward signals exist. Phase 3 introduces
procedural-content generation so the env is itself a sample from a task
distribution (curriculum / domain randomisation).

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `foraging.py` | Eat food, avoid hazards | 1 |
| `predator_prey.py` | Multi-population dynamics | 2 |
| `procedural.py` | Sample env params each episode | 3 |
| `curriculum.py` | Adaptive difficulty / POET-like task generation | Stretch |

## Acceptance tests

- Episode terminates when termination condition met (timeout, death, goal).
- `reset(mask)` re-spawns only masked envs.
- Reward is bounded and finite.
- Procedural envs produced by different seeds differ (regression on seed=0).

## Performance budget

- `step` at B=256: ≤ 2 ms incl. reward computation.

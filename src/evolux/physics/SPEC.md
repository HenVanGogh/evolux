# physics — SPEC

| Field | Value |
| --- | --- |
| Layer | 3 |
| Phase | 1 (discrete grid) → 3 (continuous, optionally Brax) |
| Depends on | `core`, `tensors`, `morphology` |

## Mission

Per-step physical updates for B parallel envs:
- Movement integration (discrete or continuous)
- Collision detection / resolution
- Energy bookkeeping (metabolism, action cost)
- Optional differentiable forward dynamics for world-model training

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `discrete.py`   | Grid-cell movement, occupancy, eat-at-cell | 1 |
| `continuous.py` | Continuous 2-D physics (positions, velocities, simple collisions) | 3 |
| `energy.py`     | Metabolism, action-cost computation | 1 |
| `brax_adapter.py` | Optional Brax-backed 3-D physics (`evolux[physics-brax]` extra) | Stretch |

## Acceptance tests

- Movement is bounded by world dims (with or without wrap).
- Collisions resolved deterministically per RNG seed.
- Energy is conservative within rounding for closed-system tests.
- Continuous integrator passes a circular-orbit unit test.

## Performance budget

- Discrete physics step at B=256, world 64×64: ≤ 1 ms on GPU.
- Continuous physics step at B=256, 1k bodies each: ≤ 5 ms.

# world — SPEC

| Field | Value |
| --- | --- |
| Layer | 3 |
| Phase | 1 (grid) → 3 (continuous biome) |
| Depends on | `core`, `tensors`, `physics`, `perception` |
| Implements | `evolux.core.protocols.World` |

## Mission

State for B parallel worlds, all packed into batched tensors. The world owns
the *physical* substrate (terrain, food, hazards, occupancy); the env layer
owns dynamics on top.

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `grid.py`       | `(B, H, W, C)` tensor world | 1 |
| `continuous.py` | Position-array world for continuous physics | 3 |
| `observation.py`| Builds Obs dicts from world state for the brain | 1 |

## Acceptance tests

- `reset(mask)` resets only masked envs, leaves others unchanged.
- `observe()` returns dict matching declared `ObsSpec`.
- `step(action)` returns `(reward, done, info)` with leading dim B.
- Determinism: same seed + same action sequence → bit-equal trajectory.

## Performance budget

- `step` at B=256, grid 64×64×8: ≤ 1 ms on GPU.
- `observe` at B=256: ≤ 0.5 ms.

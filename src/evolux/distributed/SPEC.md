# distributed — SPEC

| Field | Value |
| --- | --- |
| Layer | 6 |
| Phase | 4 |
| Depends on | `core`, `tensors`, `evolution`, `world`, `environment`, `brain`, `genome` |
| Optional dep | `ray[default]`, `pyarrow` (`evolux[distributed]` extra) |

## Mission

Scale rollouts and evaluation across cores / GPUs / nodes without changing
single-node code. The orchestrator submits jobs to a `RolloutBackend`; the
default backend is in-process, the Ray backend distributes.

## Submodules

| File | Purpose |
| --- | --- |
| `backend.py`        | `RolloutBackend` Protocol + `LocalBackend` |
| `ray_backend.py`    | Ray actor pool implementation |
| `genome_pickle.py`  | pyarrow-based zero-copy genome batch serialisation |
| `sharding.py`       | Population sharding strategies |

## Acceptance tests

- `LocalBackend` and `RayBackend` produce identical results for fixed seed.
- Ray backend graceful-degrades to local when Ray is not installed.
- Pickled genome batch round-trips bit-exact.

## Performance budget

- Throughput: ≥ 4× speedup on 4 worker actors vs single-process baseline
  (small overhead acceptable for population size < 64).

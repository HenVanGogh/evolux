# orchestrator — SPEC

| Field | Value |
| --- | --- |
| Layer | 7 |
| Phase | 0 (CLI skeleton) → 1 (training loop) → 4 (resume / checkpoint / distributed) |
| Depends on | every other module |
| Implements | top-level `EvolutionLoop` class + Typer CLI |

## Mission

The single place that wires Genome → Brain → Morphology → World →
Environment → Fitness → Selector together using only the Protocols. No
domain logic lives here — only orchestration.

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `cli.py`           | Typer entry point: `validate`, `run`, `info`, `resume` | 0 |
| `loop.py`          | `EvolutionLoop` class | 1 |
| `checkpoint.py`    | Save/restore population + RNG + step counter | 1 |
| `assemble.py`      | Build modules from config registry keys | 1 |

## Acceptance tests

- `evolux validate configs/experiments/smoke.yaml` exits 0.
- `evolux run` produces a `runs/<id>/` directory with config copy + at least one log entry.
- `checkpoint.save → load` is bit-exact (same RNG state continues identical trajectory).

## Notes

- The CLI is a *thin* layer over `EvolutionLoop`; programmatic users should
  be able to skip the CLI entirely.

# core — SPEC

| Field | Value |
| --- | --- |
| Layer | 0 |
| Phase | 0 |
| Depends on | (none) |
| Exposes | `BaseConfig`, `EvoluxConfig`, `load_config`, `RNG`, `DeviceManager`, `get_device`, `Registry`, `Trajectory`, all type aliases, all Protocols |

## Mission

Provide the foundation layer: shared types, runtime services (RNG, device, config), and the **Protocol definitions** that govern every cross-module contract. This module has zero internal dependencies and is imported by everything else.

## Files in scope

```
src/evolux/core/
  __init__.py
  types.py
  protocols.py
  config.py
  device.py
  rng.py
  registry.py
  SPEC.md          (this file)
tests/unit/core/
  test_*.py
```

## Files out of scope

Anything outside `src/evolux/core/` and `tests/unit/core/`.

## Acceptance tests

- `tests/unit/core/test_rng.py` — `RNG.split(name)` is deterministic; same name → same generator state; different names → independent streams.
- `tests/unit/core/test_config.py` — round-trip YAML load + validation; rejects unknown keys; deep-merge with `base`.
- `tests/unit/core/test_device.py` — `get_device("auto")` returns CPU when no GPU is available; respects explicit specs.
- `tests/unit/core/test_registry.py` — registration / lookup / duplicate-key error.
- `tests/unit/core/test_protocols.py` — `runtime_checkable` Protocols accept conforming dummies.

## Performance budget

N/A — no hot path here.

## Notes

- **Do not** add domain logic (env/brain/etc.) here. If a helper feels domain-specific, it belongs in that module.
- Adding a new optional method to a Protocol is non-breaking. Renaming a method is breaking — see `docs/INTERFACES.md` § Stability.

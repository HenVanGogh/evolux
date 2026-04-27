# morphology — SPEC

| Field | Value |
| --- | --- |
| Layer | 2 |
| Phase | 1 (fixed body) → 2 (modular evolved body) |
| Depends on | `core`, `tensors`, `perception` |
| Implements | `evolux.core.protocols.Morphology` (and `Sensor`, `Actuator`) |

## Mission

Body plans for creatures. Phase 1 ships a single fixed body
(`SimpleAgent`). Phase 2 introduces **modular morphology** — bodies built
from segments connected by joints, each segment carrying optional sensors
and actuators. Genome encodes the tree of segments + their parameters.

## Submodules

| File | Purpose |
| --- | --- |
| `simple.py`  | Fixed default body (Phase 1) |
| `modular.py` | Tree of segments, joints, sensors, actuators (Phase 2) |
| `sensors/`   | Sensor implementations (vision_eye, touch_pad, chemo_sniffer, audio_ear) |
| `actuators/` | Actuator implementations (motor, jet, secrete) |

## Common contract

```python
class Morphology:
    sensors:   list[Sensor]
    actuators: list[Actuator]
    def init_body_state(B, device) -> BodyState: ...
```

Sensors expose `output_dim`; the brain's input is the concatenation of all
sensor outputs. Actuators expose `input_dim`; the brain's output is split
across actuators.

## Acceptance tests

- Sum of `s.output_dim for s in sensors` matches encoder feature dim.
- Sum of `a.input_dim for a in actuators` matches action dim.
- `init_body_state(B, device)` returns a dict of (B, ...) tensors on `device`.
- Modular: serialise/deserialise the body tree round-trips.

## Performance budget

- `init_body_state(B=256)`: ≤ 1 ms.
- All sensor `sense` calls combined for B=256: ≤ 2 ms.

## References

- Sims, *Evolving Virtual Creatures* (1994) — modular body trees.
- Soft-robot evolution: Cheney et al., *Unshackling Evolution* (2014).

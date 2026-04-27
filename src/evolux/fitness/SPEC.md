# fitness — SPEC

| Field | Value |
| --- | --- |
| Layer | 4 |
| Phase | 1 (single objective) → 3 (multi-obj + descriptors for MAP-Elites) |
| Depends on | `core`, `tensors` |
| Implements | `Objective`, `FitnessAggregator`, `BehaviourDescriptor` |

## Mission

Decouple "what is success" from selection algorithms. Each objective is its
own class, registered, configurable in YAML. Aggregators combine them
(weighted sum, Pareto, lexicographic). Behaviour descriptors enable Quality-
Diversity (MAP-Elites, Novelty Search).

## Submodules

| File | Purpose | Phase |
| --- | --- | --- |
| `objectives.py`   | survival_time, food_collected, distance_travelled, novelty | 1-3 |
| `aggregators.py`  | WeightedSum, Pareto, Lexicographic | 1-2 |
| `descriptors.py`  | trajectory_endpoints, action_entropy, body_morphology_hash | 3 |

## Acceptance tests

- Each `Objective.evaluate(trajectory)` returns scalar (or per-batch tensor).
- WeightedSum recovers single-objective when only one objective registered.
- Pareto front is non-dominated by definition.
- Novelty descriptor distance triangle-inequality holds.

## Performance budget

- All objectives evaluated for B=256: ≤ 1 ms.

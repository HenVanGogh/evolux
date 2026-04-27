# evolution — SPEC

| Field | Value |
| --- | --- |
| Layer | 5 |
| Phase | 1 (tournament) → 3 (NEAT, MAP-Elites, CMA-ES, PBT, Novelty, Directed) |
| Depends on | `core`, `tensors`, `genome`, `fitness` |
| Implements | `Selector`, `Population` |

## Mission

A pluggable family of evolutionary strategies, all sharing a common
`Selector + Population` Protocol pair so the orchestrator does not change
when you swap algorithms.

## Submodules

| File | Class | Phase | Notes |
| --- | --- | --- | --- |
| `tournament.py` | `Tournament` | 1 | k-tournament + elitism baseline |
| `neat.py`       | `NEAT`       | 3 | Speciation via genetic distance |
| `map_elites.py` | `MAPElites`  | 3 | Behaviour-grid QD |
| `cma_es.py`     | `CMAES`      | 3 | Black-box continuous opt |
| `pbt.py`        | `PBT`        | 3 | Hyperparam evolution alongside weights |
| `novelty.py`    | `NoveltySearch` | 3 | Behaviour archive + k-NN distance |
| `directed.py`   | `DirectedGA` | 3 | Human-in-the-loop pressure |
| `operators.py`  | mutation / crossover utilities | 1+ | Shared by all strategies |

## Acceptance tests

- Selecting from a population of size N produces N offspring (or specified count).
- MAP-Elites grid never loses a cell once filled by a higher-fitness elite.
- CMA-ES converges on a 10-D sphere benchmark in < 200 iters.
- PBT exploit/explore swap leaves population size unchanged.
- Novelty distance is symmetric and non-negative.

## Performance budget

- One generation of size 256, tournament selection: ≤ 5 ms CPU.
- MAP-Elites archive insertion: ≤ 1 ms per individual.

## References

- NEAT: Stanley & Miikkulainen (2002).
- MAP-Elites: Mouret & Clune, *Illuminating search spaces by mapping elites* (2015).
- CMA-ES: Hansen, *The CMA Evolution Strategy: A Tutorial* (2016).
- PBT: Jaderberg et al., *Population Based Training of Neural Networks* (2017).
- Novelty Search: Lehman & Stanley, *Abandoning Objectives* (2008).

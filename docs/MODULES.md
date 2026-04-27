# Modules

Index of all modules and their dependency edges. Each module's authoritative spec lives in `src/evolux/<module>/SPEC.md`.

## Dependency graph

```
                   orchestrator ─┐
                                 ├─► all
                       cli ──────┘

         distributed ──────► world, evolution, viz
              viz   ──────► world, evolution

         evolution ─► fitness, genome, brain, world
           fitness ─► world, brain, memory
       environment ─► world, physics
            world  ─► physics, perception
            brain  ─► memory, perception, neuromod, meta
       morphology  ─► perception
           genome  ─► (none beyond core)
           memory  ─► (none beyond core)
        perception ─► (none beyond core)
           tensors ─► (none beyond core)
              core ─► (foundation)
```

## Module catalogue

| # | Module | Layer | Phase | Owner agent | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | [core](../src/evolux/core/SPEC.md) | 0 | 0 | unassigned | scaffold |
| 2 | [tensors](../src/evolux/tensors/SPEC.md) | 0 | 0 | unassigned | scaffold |
| 3 | [perception](../src/evolux/perception/SPEC.md) | 1 | 1 | unassigned | scaffold |
| 4 | [memory.working](../src/evolux/memory/SPEC.md) | 1 | 1 | unassigned | scaffold |
| 5 | [memory.episodic](../src/evolux/memory/SPEC.md) | 1 | 2 | unassigned | scaffold |
| 6 | [memory.semantic](../src/evolux/memory/SPEC.md) | 1 | 2 | unassigned | scaffold |
| 7 | [memory.hebbian](../src/evolux/memory/SPEC.md) | 1 | 2 | unassigned | scaffold |
| 8 | [genome.direct](../src/evolux/genome/SPEC.md) | 1 | 1 | unassigned | scaffold |
| 9 | [genome.hyperneat](../src/evolux/genome/SPEC.md) | 1 | 3 | unassigned | scaffold |
| 10 | [genome.indirect](../src/evolux/genome/SPEC.md) | 1 | 3 | unassigned | scaffold |
| 11 | [brain.transformer](../src/evolux/brain/SPEC.md) | 2 | 1 | unassigned | scaffold |
| 12 | [brain.ssm](../src/evolux/brain/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 13 | [brain.hybrid](../src/evolux/brain/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 14 | [brain.world_model](../src/evolux/brain/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 15 | [brain.neuromod](../src/evolux/brain/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 16 | [brain.meta](../src/evolux/brain/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 17 | [morphology](../src/evolux/morphology/SPEC.md) | 2 | 2 | unassigned | scaffold |
| 18 | [physics](../src/evolux/physics/SPEC.md) | 3 | 1 | unassigned | scaffold |
| 19 | [world](../src/evolux/world/SPEC.md) | 3 | 1 | unassigned | scaffold |
| 20 | [environment.procedural](../src/evolux/environment/SPEC.md) | 4 | 3 | unassigned | scaffold |
| 21 | [environment.climate](../src/evolux/environment/SPEC.md) | 4 | 3 | unassigned | scaffold |
| 22 | [environment.ecology](../src/evolux/environment/SPEC.md) | 4 | 3 | unassigned | scaffold |
| 23 | [fitness.objectives](../src/evolux/fitness/SPEC.md) | 4 | 1 | unassigned | scaffold |
| 24 | [fitness.intrinsic](../src/evolux/fitness/SPEC.md) | 4 | 3 | unassigned | scaffold |
| 25 | [fitness.pareto](../src/evolux/fitness/SPEC.md) | 4 | 3 | unassigned | scaffold |
| 26 | [evolution.neat](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 27 | [evolution.map_elites](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 28 | [evolution.cma_es](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 29 | [evolution.pbt](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 30 | [evolution.novelty](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 31 | [evolution.directed](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 32 | [evolution.operators](../src/evolux/evolution/SPEC.md) | 5 | 3 | unassigned | scaffold |
| 33 | [distributed](../src/evolux/distributed/SPEC.md) | 6 | 4 | unassigned | scaffold |
| 34 | [viz.stats](../src/evolux/viz/SPEC.md) | 6 | 1 | unassigned | scaffold |
| 35 | [viz.replay](../src/evolux/viz/SPEC.md) | 6 | 4 | unassigned | scaffold |
| 36 | [viz.web](../src/evolux/viz/SPEC.md) | 6 | 4 | unassigned | scaffold |
| 37 | [orchestrator](../src/evolux/orchestrator/SPEC.md) | 7 | 1 | unassigned | scaffold |

Update **Owner agent** and **Status** columns as work progresses.

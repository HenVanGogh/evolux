# evolux

> **GPU-native, modular framework for the simulated evolution of intelligent creatures.**

`evolux` is a research-grade testbed for evolving creatures with **modern brain architectures** (Transformer + State-Space Model hybrids, world models, neuromodulation), **differentiable hierarchical memory**, and **directed-evolution** algorithms (Quality-Diversity, Novelty Search, gradient-augmented selection) — all running batched on GPU and scalable to a Ray cluster.

The project is engineered for **parallel multi-agent development**: every module ships with a self-contained `SPEC.md` (interface + invariants + acceptance tests), so independent cloud agents can pick up modules and ship them without coordination overhead.

---

## Headline features (target v1)

| Domain | Feature |
| --- | --- |
| Brain | Transformer + Mamba/SSM hybrid policy |
| Brain | Dreamer-V3-style world model with imagination rollouts |
| Brain | Neuromodulator gating (dopamine / serotonin / surprise) |
| Brain | MAML / PEARL meta-learning adapters for lifetime adaptation |
| Memory | DNC-style differentiable episodic memory |
| Memory | Retrieval-augmented semantic memory (FAISS / torch index) |
| Memory | Slow Hebbian "procedural" weights persisting across episodes |
| Body | Evolved modular morphology (limbs, sensors, actuators) |
| Genome | Direct (NEAT) + indirect (HyperNEAT/CPPN) + diffusion-decoded encodings |
| Evolution | NEAT, MAP-Elites, CMA-ES, PBT, Novelty Search, Directed-GA |
| Fitness | Multi-objective (Pareto, lexicase) + intrinsic rewards (curiosity, empowerment) |
| Environment | Procedural terrain, climate cycles, ecology, predator-prey |
| Compute | Vectorised GPU world (B parallel envs in one tensor) |
| Compute | Ray-based distributed training + sharded checkpoints |
| Tooling | wandb / TensorBoard logging |
| Tooling | Live web dashboard (FastAPI + Three.js) |

See [docs/ROADMAP.md](docs/ROADMAP.md) for the phased delivery plan and [docs/MODULES.md](docs/MODULES.md) for the module dependency graph.

---

## Architecture at a glance

```
                ┌───────────────────────────────────────────────┐
                │             Orchestrator / CLI                │
                └───────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   ┌──────────┐                  ┌──────────┐                  ┌──────────┐
   │  World   │ ◄──── obs ─────  │ Creature │ ───── act ────► │   Env    │
   │ (B envs) │                  │ batch    │                  │ dynamics │
   └────┬─────┘                  └────┬─────┘                  └────┬─────┘
        │                             │                             │
        │              ┌──────────────┼──────────────┐              │
        │              ▼              ▼              ▼              │
        │        ┌──────────┐  ┌──────────┐  ┌──────────┐           │
        │        │  Brain   │  │  Memory  │  │ Morph.   │           │
        │        └────┬─────┘  └────┬─────┘  └──────────┘           │
        │             ▼             ▼                               │
        │        ┌──────────────────────┐                           │
        │        │   Neuromod / Meta    │                           │
        │        └──────────────────────┘                           │
        │                                                           │
        └─────────────► Fitness ◄──── Genome ◄──── Evolution ◄──────┘
```

All inter-module data flows are **fully-batched torch tensors** of shape `(B, ...)` where `B` is the number of parallel envs/agents. No per-creature Python loops in the hot path.

---

## Quickstart

```bash
git clone https://github.com/HenVanGogh/evolux.git
cd evolux
./scripts/bootstrap_dev.sh           # creates venv, installs deps + pre-commit
source .venv/bin/activate
pytest -q                            # all module unit tests should pass
evolux validate --config configs/experiments/smoke.yaml
evolux run      --config configs/experiments/foraging.yaml
```

---

## Repository layout

```
.github/             CI workflows, issue templates, agent instructions
docs/                Architecture, roadmap, module specs
src/evolux/          The framework
  ├── core/          Config, types, device, RNG, registry, protocols
  ├── tensors/       Batched/struct-of-arrays utilities
  ├── world/         Vectorised world tensors
  ├── physics/       Movement, collision, energy
  ├── perception/    Vision/proprio/chemo encoders
  ├── morphology/    Evolved bodies
  ├── brain/         Transformer · SSM · world model · neuromod · meta
  ├── memory/        Working · episodic · semantic · Hebbian
  ├── genome/        Direct · HyperNEAT · indirect encodings
  ├── evolution/     NEAT · MAP-Elites · CMA-ES · PBT · novelty · directed
  ├── fitness/       Objectives · Pareto · intrinsic motivation
  ├── environment/   Procedural · climate · ecology
  ├── distributed/   Ray runner, checkpointing, sharding
  ├── viz/           Stats, replay, web dashboard
  └── orchestrator/  Experiments, scheduler, CLI
tests/               unit/ · integration/ · benchmarks/
configs/             YAML experiment configs
scripts/             Dev bootstrap, issue creation, benchmarks
prototypes/          Reference prototypes (v0_numpy/)
```

---

## For cloud agents

If you are an agent assigned to a module, read **AGENTS.md** first and then your module's `src/evolux/<module>/SPEC.md`. Each spec defines:

1. The Protocol(s) you must implement
2. The acceptance tests your code must pass
3. The performance budget (latency / VRAM)
4. The dependencies you may import (no upward dependencies allowed)

---

## License

MIT — see [LICENSE](LICENSE).

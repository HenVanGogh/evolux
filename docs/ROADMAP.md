# Roadmap

The project is split into four phases. Each phase has a **demo target** — something runnable end-to-end at the end of the phase. Each phase's modules are designed so that several can proceed in parallel.

---

## Phase 0 — Foundation (parallelism: high)

**Demo target:** `pytest -q` green on empty stubs; CI matrix runs per-module test jobs.

- [ ] `core/` — config, types, device, RNG, registry, protocols
- [ ] `tensors/` — batched utilities, struct-of-arrays helpers, `assert_finite`
- [ ] Repo tooling — pyproject, ruff, pytest, pre-commit, CI workflow
- [ ] Layering check script
- [ ] Per-module SPEC.md scaffolds

Owner-style: **1–2 agents**, sequential to avoid merge churn on shared interfaces.

---

## Phase 1 — Single-creature, single-env smoke loop (parallelism: medium)

**Demo target:** one creature, one env, random brain, runs 1k steps without NaN, logs to stdout.

- [ ] `world/` — minimal grid-tensor world (B, H, W, C)
- [ ] `physics/` — discrete movement, energy bookkeeping
- [ ] `perception/` — basic CNN encoder + proprio MLP
- [ ] `brain/transformer.py` — minimal causal Transformer policy
- [ ] `memory/working.py` — KV cache wrapper
- [ ] `morphology/` — fixed default body
- [ ] `genome/direct.py` — direct encoding of brain weights
- [ ] `fitness/objectives.py` — survival + energy
- [ ] `orchestrator/cli.py` — `evolux validate` command
- [ ] `viz/stats.py` — TensorBoard sink

Owner-style: **6–8 agents**, one per module, brain-first then world-first chains can run independently.

---

## Phase 2 — Modern brain & memory (parallelism: high)

**Demo target:** an agent with a Transformer+SSM hybrid brain, world model, and DNC memory, learns a foraging task in batched envs.

- [ ] `brain/ssm.py` — Mamba/S4 selective SSM block
- [ ] `brain/hybrid.py` — Transformer+SSM hybrid policy
- [ ] `brain/world_model.py` — Dreamer-V3-lite (RSSM + decoder + reward head)
- [ ] `brain/neuromod.py` — neuromodulator gating layers
- [ ] `brain/meta.py` — MAML/PEARL inner-loop adapter
- [ ] `memory/episodic.py` — DNC-style differentiable memory
- [ ] `memory/semantic.py` — FAISS-backed retrieval
- [ ] `memory/hebbian.py` — slow-weight associative matrix
- [ ] `perception/` extensions — chemo, audio
- [ ] `morphology/` — modular limbs (segments × joints)
- [ ] Brain training loop (Dreamer rollouts) + checkpointing

Owner-style: **8–12 agents** in parallel; brain submodules independent given Phase 1 protocols.

---

## Phase 3 — Evolution & complex environment (parallelism: high)

**Demo target:** a full population evolves modular bodies + brains for a procedurally-generated environment with seasons and predators; MAP-Elites archive populates.

- [ ] `genome/hyperneat.py` — CPPN encoding
- [ ] `genome/indirect.py` — diffusion-decoded weight tensors
- [ ] `evolution/neat.py` — NEAT speciation
- [ ] `evolution/map_elites.py` — quality-diversity archive
- [ ] `evolution/cma_es.py` — covariance-matrix adaptation
- [ ] `evolution/pbt.py` — population-based training
- [ ] `evolution/novelty.py` — behaviour-space novelty
- [ ] `evolution/directed.py` — gradient-augmented selection
- [ ] `evolution/operators.py` — mutation / crossover
- [ ] `fitness/intrinsic.py` — surprise, empowerment, curiosity
- [ ] `fitness/pareto.py` — multi-objective ranking
- [ ] `environment/procedural.py` — terrain noise, biomes
- [ ] `environment/climate.py` — weather + seasons
- [ ] `environment/ecology.py` — plants, predators, food chains

Owner-style: **10+ agents**; evolution algorithms each have their own SPEC and can be developed independently.

---

## Phase 4 — Distributed + visualisation + polish (parallelism: medium)

**Demo target:** A run on a small Ray cluster, with a live web dashboard showing the best creature playing, lineage trees, and the MAP-Elites grid.

- [ ] `distributed/ray_runner.py` — Ray actor sharding
- [ ] `distributed/checkpoint.py` — sharded async checkpoints
- [ ] `distributed/shard.py` — genome / archive shard primitives
- [ ] `viz/replay.py` — generate GIF/video of a creature episode
- [ ] `viz/web/` — FastAPI backend + Three.js frontend
- [ ] Documentation pass — tutorials, design docs, examples
- [ ] Reference experiments + baselines

Owner-style: **4–6 agents**; web frontend separable from Ray work.

---

## Stretch / research

- Embodied language ("creatures with proto-words")
- Multi-agent communication / cooperation
- Open-ended evolution (POET-style co-evolved environments)
- Differentiable physics (Brax integration)

---

## Rough sizing

| Phase | Modules | Suggested agents | Target wall-clock if 4 agents in parallel |
| --- | --- | --- | --- |
| 0 | 5 | 1–2 | 1 day |
| 1 | 10 | 6–8 | 3–5 days |
| 2 | 11 | 8–12 | 1–2 weeks |
| 3 | 14 | 10+ | 2–3 weeks |
| 4 | 7 | 4–6 | 1 week |

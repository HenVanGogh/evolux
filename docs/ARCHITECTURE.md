# Architecture

## Design principles

1. **Batched-tensor everywhere.** Hot-path data is `(B, ...)` torch tensors on GPU. No per-creature Python loops in the inner loop. Use `torch.vmap` or explicit batched ops.
2. **Modules talk through Protocols, not classes.** Every cross-module boundary is a `typing.Protocol` defined in `core.protocols`. Concrete implementations are swappable without changing callers.
3. **Pure stateless components where possible.** `forward(state, input) -> (state', output)` style. State lives in tensors, not Python attributes. This is what makes vectorisation, checkpointing, and Ray sharding cheap.
4. **No upward dependencies.** A module may import from `core/`, `tensors/`, and modules listed in its SPEC's `depends_on` — never from a higher-level module.
5. **Every module is independently testable.** A module's unit tests must run in isolation with mocked Protocols, no full simulation needed.
6. **Configs are typed.** YAML loads into pydantic dataclasses defined in `core.config`. Runtime config errors are surfaced at start, not in step 50,000.
7. **Determinism is opt-in but supported.** A single `RNG` service (split per module) makes any run reproducible given a seed.

## Layered structure

```
Layer 0  core, tensors                 (no internal deps)
Layer 1  perception, memory, genome    (deps: layer 0)
Layer 2  brain, morphology             (deps: layer 0, 1)
Layer 3  physics, world                (deps: layer 0, 2)
Layer 4  environment, fitness          (deps: layer 0..3)
Layer 5  evolution                     (deps: layer 0..4)
Layer 6  distributed, viz              (deps: layer 0..5)
Layer 7  orchestrator, cli             (deps: all)
```

A module lint check (`scripts/check_layering.py`) enforces this graph in CI.

## Hot-path tensor shapes

Conventions used everywhere:

| Symbol | Meaning | Typical |
| --- | --- | --- |
| `B`  | batch (parallel envs/agents) | 256–8192 |
| `T`  | time / sequence length | 1 (per step) or rollout length |
| `Hₐ` | brain hidden / action width | 64–512 |
| `M`  | memory slots | 32–1024 |
| `Pₓ` | proprioception dim | varies per body |
| `Ix` | image side | 32 / 64 |

Brain `forward`:
```
obs:    {"image": (B, C, Ix, Ix), "proprio": (B, Pₓ), "tokens": (B, T, D)}
state:  {"recurrent": (B, L, Hₐ), "memory": (B, M, D)}
→ action_logits: (B, A), value: (B,), aux: dict
```

## State / memory / time

Each creature has three state tiers:

| Tier | Lifetime | Mechanism |
| --- | --- | --- |
| Activation | one step | Transformer KV cache, SSM hidden state |
| Episodic   | one episode (generation) | DNC-style differentiable memory |
| Procedural | many episodes / lifetime | Slow Hebbian weights, retrieval-augmented vector store |

The genome encodes only **structure + initial parameters**. Hebbian and episodic state are reset/derived per episode unless `lifetime_learning=True`.

## Evolution loop

```
for generation g:
    1. Decode genomes → (brains, bodies)            [genome → brain, morphology]
    2. Reset world & place creatures                [world]
    3. Roll out T_env steps:                        [world × brain × physics × env]
         obs = world.observe()
         act = brain(obs, state)
         world, env = step(world, env, act)
    4. Compute fitness (multi-objective + intrinsic)[fitness]
    5. Update behaviour archive                      [evolution.novelty / map_elites]
    6. Augment fitness with novelty / gradient       [evolution.directed]
    7. Select parents, reproduce, mutate             [evolution.operators]
    8. Checkpoint, log                               [distributed, viz]
```

Steps 1–3 are GPU-batched. Step 7 is mostly CPU — genome operations on Python objects — and runs concurrently with checkpointing.

## Distributed model

* **Single-node:** one process, one GPU; `B` envs co-located.
* **Multi-GPU:** `torch.distributed` data-parallel, each rank owns `B/world_size` envs; gradients (for differentiable components like world model) all-reduced.
* **Multi-node (Ray):** `RayRunner` sharding generations across worker actors, central `Orchestrator` coordinates evolution. Genomes serialise via Arrow.

## Differentiability boundaries

Some components are **differentiable**:
- World model (Dreamer-V3) — trained by gradient descent on imagined trajectories.
- Episodic memory write/read — soft attention gradients.

Others are **non-differentiable** and trained by evolution:
- Genome → brain decoding (NEAT, HyperNEAT).
- Morphology.
- Behavioural policies (when running pure-evolution mode).

Hybrid mode (recommended): differentiable components train via Dreamer; evolution mutates topology + meta-parameters around the differentiable kernels.

## Performance budgets (target, GTX 1050 Ti class GPU)

| Path | Budget |
| --- | --- |
| Single env step (B=256) | ≤ 2 ms |
| Brain forward (B=256, hybrid Tx+SSM) | ≤ 5 ms |
| Generation of 256 agents × 500 steps | ≤ 5 min |

A100-class targets are documented per module SPEC.

## Failure modes & invariants

- **NaN guard.** `core.tensors.assert_finite` runs in dev/test mode after every brain forward.
- **Genome bloat.** Mutations capped by `max_nodes`, `max_connections`; structural mutation rates anneal as size grows.
- **Memory drift.** Hebbian weights clipped to `[-w_max, w_max]`; retrieval index garbage-collects stale entries.
- **Selection collapse.** Speciation + novelty pressure prevent loss of diversity.

# KICKOFF — Phase 0/1 work for all agents

> **Read order**: this file → [`AGENTS.md`](../AGENTS.md) → your module's `src/evolux/<module>/SPEC.md` → `docs/ARCHITECTURE.md`.
>
> All work below is **Phase 0 or Phase 1** — i.e. the "first useful loop" milestone. After this round, the framework should run a smoke evolution loop end-to-end with a simple Transformer brain on a foraging grid world.

---

## Ground rules (recap — non-negotiable)

1. **Stay in scope.** Only edit files listed in your module's `SPEC.md` "Files in scope" + your `tests/unit/<module>/`.
2. **No cross-module imports of concrete classes.** If you need `Brain`, import the Protocol from `evolux.core.protocols`, never the class.
3. **Layering.** A module may import only from strictly lower layers. `python scripts/check_layering.py` must pass.
4. **Determinism.** All randomness via `evolux.core.rng.RNG`. No `torch.manual_seed`, no bare `random`, no unseeded `torch.rand*` in module code.
5. **No `print`.** Use `logging` (or `rich` for CLI output).
6. **Type hints on every public symbol.** Ruff `ANN` rules will fail you otherwise.
7. **Tests.** New code goes with new tests under `tests/unit/<module>/`. CPU-only by default; mark GPU-only tests with `@pytest.mark.gpu`.
8. **PR per module.** One module per PR. Reference the issue with `Closes #N`.
9. **Branch name**: `feat/<module>-<issue-number>` or `fix/<module>-<issue-number>`.
10. **Conventional commits**: `feat(brain): add TransformerBrain`, `test(memory): cover episodic write/read`.

---

## Workflow checklist (every PR)

```bash
git checkout -b feat/<module>-<N>
# ... edit only files in scope ...
ruff check . --fix && ruff format .
python scripts/check_layering.py
pytest tests/unit/<module>/ -q
# Push, open PR with Closes #N, fill PR template.
```

A PR is mergeable when:
- [ ] All acceptance tests in your SPEC pass
- [ ] No layering violation
- [ ] No new pip dependency unless declared & justified
- [ ] Coverage of new code ≥ 80%
- [ ] Ruff clean
- [ ] No regressions in existing tests

---

## Per-module Phase-0/1 starter task list

Each section below corresponds to one open issue. Pick one issue, do the tasks, open one PR.

---

### `core` (Issue #13, Phase 0)

**Status**: skeleton already implemented. This issue is for *hardening*.

Tasks:
1. Add `tests/unit/core/test_protocols.py` that asserts every Protocol in `evolux.core.protocols` is `runtime_checkable` and has `__doc__`.
2. Add a `Trajectory.concat(others)` classmethod (concatenate along time dim) + tests.
3. Add `RNG.fork(n)` returning `n` independent child RNGs (deterministic given seed) + tests.
4. Add docstring `Notes` blocks to `RNG`, `DeviceManager`, `load_config` referencing the design rationale in `docs/ARCHITECTURE.md`.
5. Bump `PROTOCOL_VERSION` if you add any new Protocol method (you shouldn't need to).

Acceptance: 100% line coverage on `core/`; existing 31 tests still pass.

---

### `tensors` (Issue #6, Phase 0)

**Status**: `assert_finite`, `flatten/unflatten_batch`, `one_hot`, `soft_attention` already exist.

Tasks:
1. Add `masked_mean(x, mask, dim) -> Tensor` and `masked_softmax(x, mask, dim) -> Tensor` (used everywhere by partial-batch resets).
2. Add `where_done(state, done_mask, init_fn)` — replaces rows of a state dict where `done_mask` is True with freshly-initialised values from `init_fn`. This is the canonical "reset only finished envs" primitive.
3. Add `gather_indices(x, idx)` — vectorised batched gather (B, T, D) along T using (B,) indices.
4. Tests for shape, gradient flow, dtype preservation, and a 1k-iter benchmark to ensure no allocation explosion.

Reference test pattern: see `tests/unit/tensors/test_tensors.py`.

Out of scope: do **not** introduce einops dependency for these (use `torch` only); einops is reserved for `brain/`.

---

### `perception` (Issue #15, Phase 1)

Tasks:
1. Implement `vision.py::VisionCNN` — NatureCNN-lite: 3 conv layers (32→64→64) + flatten + linear head. Input `(B, C, H, W)`, output `(B, output_dim)`. Register as `"vision_cnn_v1"`.
2. Implement `proprio.py::ProprioMLP` — 2-layer MLP `(B, P) -> (B, output_dim)`. Register as `"proprio_mlp_v1"`.
3. Implement `multimodal.py::ConcatEncoder` — wraps a dict `{name: encoder}` and concats outputs. Output dim = sum of children. Register as `"concat"`.
4. Tests per encoder: output shape; gradient flow (`requires_grad` propagation); deterministic given RNG.
5. Add `Encoder` Protocol to `core.protocols` only if reviewers agree it belongs there — otherwise keep it as a local module convention.

Perf budget: VisionCNN B=64, 32×32 RGB ≤ 5 ms CPU.

Out of scope: `chemo.py` and `audio.py` are Phase 2.

---

### `memory` (Issue #5, Phase 1)

Tasks:
1. Implement `working.py::WorkingMemory` — sliding-window KV cache. State = `{"keys": (B, M, D), "values": (B, M, V), "ptr": (B,)}`. `write` appends with circular-buffer semantics. `read` does soft attention via `tensors.soft_attention`. Register as `"working_v1"`.
2. Verify it implements the `Memory` Protocol (`isinstance(m, Memory)` is True at runtime).
3. Tests: `init_state` shapes/device; `write` is pure (returns new dict, doesn't mutate); written-then-read recovers value with cosine ≥ 0.95 (single-write); `reset_episode(state, mask)` zeroes only masked rows.
4. Bench: B=256, M=128, D=64, V=64 — combined write+read ≤ 1 ms on CPU.

Out of scope: `episodic.py`, `semantic.py`, `hebbian.py` — Phase 2 (separate issues will be filed for them).

---

### `genome` (Issue #2, Phase 1)

Tasks:
1. Implement `direct.py::DirectGenome` — flat float32 vector of length `n_params`. Methods: `decode_brain(brain_factory)` (calls factory with the vector reshaped per the factory's `parameter_shapes`), `decode_morphology()` (returns a default `SimpleMorphology` from `morphology` module — but ONLY via the Protocol, never the class), `serialize() -> bytes` (use `numpy.tobytes`).
2. Implement `direct.py::DirectOperator` — `mutate(g, rng)` adds Gaussian noise (σ configurable); `crossover(a, b, rng)` does uniform crossover; `distance(a, b)` returns L2.
3. Register both with `GENOME_REGISTRY` and `GENOME_OPERATOR_REGISTRY` under key `"direct"`.
4. Tests: serialize→deserialize round-trip bit-exact; `distance(g, g) == 0`; mutation σ statistically correct over 10k samples; no in-place mutation of inputs.

Out of scope: `neat.py`, `hyperneat.py`, `indirect.py` — Phase 3.

---

### `brain` (Issue #14, Phase 1)

Tasks:
1. Implement `transformer.py::TransformerBrain` — small causal Transformer:
   - Pre-LN, RMSNorm, RoPE positional encoding, GELU MLP.
   - Configurable: `n_layers`, `n_heads`, `d_model`, `d_ff`, `max_context`.
   - State = KV cache `{"k": (B, L, n_heads, T, head_dim), "v": ..., "pos": (B,)}`.
   - `forward(obs, state)` returns `(action, state', aux)`. Action is sampled from a Gaussian head (mean, log_std) for continuous specs OR from a Categorical head for discrete.
2. Register as `"transformer_v1"` in `BRAIN_REGISTRY`.
3. Implement `factory.py::TransformerBrainFactory` (implements `BrainFactory`): exposes `parameter_shapes` so a `DirectGenome` can decode a flat vector into model weights.
4. Tests: forward output shape matches `action_spec`; recurrence (state changes step-to-step); gradients flow to all `trainable_parameters()`; deterministic given RNG.
5. Use `einops` for any non-trivial reshape (`pyproject.toml` already has it).

Perf budget: B=64, d_model=128, ctx=64, 4 layers, forward ≤ 5 ms CPU.

Out of scope: `ssm.py`, `hybrid.py`, `world_model.py`, `neuromod.py`, `meta.py` — Phase 2.

---

### `morphology` (Issue #4, Phase 1)

Tasks:
1. Implement `simple.py::SimpleMorphology` — fixed body with: 1 vision sensor, 1 proprio sensor, 2 motor actuators (forward, turn).
2. Implement `simple.py::SimpleVisionSensor`, `SimpleProprioSensor`, `MotorActuator` matching the `Sensor` / `Actuator` Protocols.
3. `init_body_state(B, device)` returns `{"position": (B, 2), "heading": (B,), "energy": (B,)}` — all initialised to sane defaults.
4. Tests: sensor output dims sum correctly; actuator input dims sum correctly; body state on requested device; serialise/deserialise (just dict round-trip is fine for SimpleMorphology).

Out of scope: `modular.py`, `sensors/`, `actuators/` — Phase 2.

---

### `physics` (Issue #12, Phase 1)

Tasks:
1. Implement `discrete.py::DiscretePhysics` — wraps a `(B, H, W, C)` world tensor + `(B, 2)` agent positions. Methods:
   - `step(world, positions, headings, actions) -> (positions', headings', collision_mask, energy_cost)`.
   - Movement = forward 1 cell if `actions[:, 0] > 0.5`; turn ±90° if `actions[:, 1]` non-zero.
   - Collisions = bumping into wall channel of world; agent doesn't move on collision.
2. Implement `energy.py::compute_action_cost(actions) -> Tensor` — simple sum-of-squares cost.
3. Tests: bounded movement (positions stay within world dims); collision blocks movement; energy cost ≥ 0; deterministic.

Perf budget: B=64, world 32×32 — step ≤ 1 ms CPU.

Out of scope: `continuous.py`, `brax_adapter.py` — Phase 3 / stretch.

---

### `world` (Issue #8, Phase 1)

Tasks:
1. Implement `grid.py::GridWorld` (implements `World` Protocol) — `(B, H, W, C)` tensor with channels: `[wall, food, agent_occupancy, hazard]`.
2. `reset(mask)`: re-spawns walls (random maze or empty), food (Poisson), agent positions for masked envs only.
3. `observe(positions, headings)`: returns dict `{"vision": (B, C, k, k), "proprio": (B, P)}` — `vision` is a k×k local crop centred on agent, rotated to agent frame; `proprio` is `[energy, hunger, last_action]`.
4. `step(actions)`: calls `physics.discrete.DiscretePhysics`, updates food (eat-at-cell), returns `(reward, done, info)`. `done` triggers when energy ≤ 0 OR step count ≥ max_steps.
5. Tests: deterministic given seed; partial reset works; observe shape matches declared `ObsSpec`.

Out of scope: `continuous.py` — Phase 3.

---

### `environment` (Issue #9, Phase 1)

Tasks:
1. Implement `foraging.py::ForagingEnv` (implements `Environment` Protocol) — wraps a `GridWorld` (via the `World` Protocol, never the concrete class). Defines:
   - Episode termination = `done` from world.
   - Reward shaping: +1 per food eaten, −0.01 per step (idle penalty), −10 on death.
2. Register as `"foraging_v1"`.
3. Tests: episode terminates; reward bounded; partial reset.

Out of scope: `predator_prey.py`, `procedural.py`, `curriculum.py` — Phase 2/3.

---

### `fitness` (Issue #1, Phase 1)

Tasks:
1. Implement `objectives.py`:
   - `SurvivalTime` — sum of `~done` over the trajectory.
   - `FoodCollected` — sum of positive rewards.
   - `DistanceTravelled` — L2 path length from positions in `aux`.
   Each implements the `Objective` Protocol; register all in `OBJECTIVE_REGISTRY`.
2. Implement `aggregators.py::WeightedSum` (implements `FitnessAggregator`).
3. Tests: each objective returns scalar tensor of shape `(B,)`; `WeightedSum` recovers single-objective when only one weight nonzero.

Out of scope: `descriptors.py`, Pareto/Lexicographic aggregators — Phase 2/3.

---

### `evolution` (Issue #10, Phase 1)

Tasks:
1. Implement `tournament.py::Tournament` (implements `Selector`) — k-tournament selection with configurable elitism. Operates on a list/batch of `(genome, fitness)` pairs and returns N offspring genomes (using the genome's `GenomeOperator` for mutation/crossover).
2. Implement `population.py::Population` (implements `Population` Protocol) — holds a list of genomes + their cached fitness; supports `add`, `select`, `step_generation(selector, operator)`.
3. Tests: deterministic given RNG; population size constant; elitism preserves top-k; selection pressure increases with k.
4. Register as `"tournament"` in `EVOLUTION_REGISTRY`.

Out of scope: `neat.py`, `map_elites.py`, `cma_es.py`, `pbt.py`, `novelty.py`, `directed.py` — Phase 3.

---

### `viz` (Issue #7, Phase 1)

Tasks:
1. Implement `loggers.py::JsonlLogger` — append-only JSONL writer for scalars/dicts. Always available (no extras).
2. Implement `loggers.py::TensorboardLogger` — guarded by `try/except ImportError`; falls back to `JsonlLogger` if `tensorboard` not installed (warn once via `logging`).
3. Implement `loggers.py::WandbLogger` — same guard pattern for `wandb`.
4. Common Protocol: `log_scalar(tag, value, step)`, `log_hist(tag, values, step)`, `close()`.
5. Tests: JsonlLogger writes valid JSONL; missing-extra path gracefully degrades (no raise); `close()` is idempotent.

Out of scope: `renderers.py`, `qd_plots.py`, `lineage.py`, `web/` — Phase 3/4.

---

### `orchestrator` (Issue #3, Phase 1)

Tasks:
1. Implement `assemble.py::assemble_from_config(cfg)` — reads `cfg.modules.{world, brain, genome, evolution, ...}` keys, looks them up in their respective registries, instantiates them, returns an `AssembledRun` namedtuple/dataclass.
2. Implement `loop.py::EvolutionLoop` — the canonical training loop:
   ```
   for gen in range(max_generations):
       for env_step in range(steps_per_generation):
           obs = world.observe(...)
           action, brain_state = brain.forward(obs, brain_state)
           reward, done, info = world.step(action)
           # collect trajectory
       fitness = aggregator.evaluate_population(trajectories)
       population.step_generation(selector, operator)
       logger.log_scalar("fitness/mean", fitness.mean(), gen)
   ```
3. Implement `checkpoint.py::save(run, path)` and `load(path) -> run` — pickle population genomes (via their `serialize`) + RNG state + step counter.
4. Wire `evolux run <config.yaml>` in `cli.py` to call `EvolutionLoop`.
5. Tests: smoke run on `configs/experiments/smoke.yaml` completes in ≤ 5 s on CPU with no errors.

This is the **integration** module. **You depend on every other Phase-1 PR being merged first.** Until then, you can scaffold the loop with mocked Protocols (`unittest.mock.Mock(spec=Brain)` etc.) and verify wiring.

---

### `distributed` (Issue #11, Phase 4)

**Defer.** This is Phase 4. Do not start until Phase 1 is merged and the in-process loop is stable. The issue stays open as a placeholder.

If you want to do prep work now: add `backend.py::RolloutBackend` Protocol + `LocalBackend` (in-process implementation that wraps the existing loop). No Ray imports yet.

---

## Coordination

- **Two agents pick the same issue?** Second one yields and picks the next unblocked module from the same Phase. Comment on the issue: "yielding to @other-agent, picking up #N instead".
- **Need a Protocol change?** STOP. Open a PR titled `protocols: <reason>` first. Get one human review. Only after that PR merges can dependent module PRs use the new contract. See `AGENTS.md` §9.
- **Need a new pip dependency?** Open an issue tagged `dependency-bump` with justification. Don't add it in your module PR.
- **Cross-module question?** Open a GitHub Discussion under "Architecture" rather than guessing at another module's API.
- **Acceptance test unclear?** Comment on your issue. Don't relax it — clarify it.

## Recommended pickup order (parallelism map)

```
Wave 1 (no deps on Phase-1 work, can start in parallel):
  #6 tensors      #13 core (hardening)    #2 genome    #4 morphology
  #15 perception  #5 memory (working)     #1 fitness   #7 viz

Wave 2 (need wave 1 modules to land first):
  #12 physics  → needs morphology stubs
  #14 brain    → needs perception, memory, tensors
  #8 world     → needs physics, perception
  #10 evolution → needs genome, fitness

Wave 3 (integration):
  #9 environment → needs world
  #3 orchestrator → needs everything

Phase 4 (deferred):
  #11 distributed
```

Pick a Wave 1 issue if you're starting fresh.

---

## Definition of "Phase 1 complete"

- `evolux validate configs/experiments/smoke.yaml` exits 0 (already true).
- `evolux run configs/experiments/smoke.yaml` runs 2 generations × 16 env steps × 8 agents on a foraging grid with a TransformerBrain + DirectGenome + Tournament selection, writes a `runs/<id>/` dir with JSONL log of fitness, and exits 0.
- All Phase-1 acceptance tests in every SPEC.md pass.
- CI green on `main`.

That's the milestone. Ship it.

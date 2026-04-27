# brain — SPEC

| Field | Value |
| --- | --- |
| Layer | 2 |
| Phase | 1 (transformer) → 2 (ssm, hybrid, world_model, neuromod, meta) |
| Depends on | `core`, `tensors`, `memory`, `perception` |
| Implements | `evolux.core.protocols.Brain` (and `Imaginer` for world_model) |

## Mission

Production-grade implementations of modern policy architectures, all sharing
the `Brain` Protocol so the orchestrator and evolution code never know which
architecture is in use.

## Submodules

| File | Class | Phase | Algorithm | Notes |
| --- | --- | --- | --- | --- |
| `transformer.py` | `TransformerBrain` | 1 | Causal Transformer | Pre-LN, RMSNorm, RoPE |
| `ssm.py`         | `SSMBrain`         | 2 | Mamba selective SSM | Linear-time recurrence |
| `hybrid.py`      | `HybridBrain`      | 2 | Tx ⊕ SSM blocks     | Layer-interleaved |
| `world_model.py` | `WorldModelBrain`  | 2 | Dreamer-V3-lite     | RSSM + reconstruction + reward |
| `neuromod.py`    | `NeuromodWrapper`  | 2 | Modulator gating    | Wraps another brain |
| `meta.py`        | `MetaAdapter`      | 2 | MAML/PEARL inner-loop | Wraps another brain |

Each registered as e.g. `@BRAIN_REGISTRY.register("transformer")`.

## Common contract

See `core.protocols.Brain` and `Imaginer`. Key methods:

```python
init_state(B, device)                                  -> BrainState
forward(obs, state) -> (action, state', aux)
imagine(obs, state, horizon) -> Trajectory             # (Imaginer only)
trainable_parameters() -> Iterator[Parameter]
```

## Acceptance tests (per submodule)

- Forward output shape matches `action_spec`.
- State persistence: `s1 = forward(o, s0)[1]; s2 = forward(o, s1)[1]; s2 != s1` (recurrence works).
- Gradients flow to all `trainable_parameters` (when applicable).
- For `WorldModelBrain`: `imagine(horizon=10)` returns a `Trajectory` with `T=10`.
- For `NeuromodWrapper`: gating tensor in `aux` has shape `(B, n_modulators)`.

## Performance budget (B=256, hidden=128, GTX-1050-Ti class)

- TransformerBrain forward: ≤ 4 ms
- SSMBrain forward: ≤ 2 ms (linear time advantage)
- HybridBrain forward: ≤ 5 ms
- WorldModelBrain imagine(horizon=15): ≤ 30 ms

## References

- Mamba: Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023).
- Dreamer-V3: Hafner et al., *Mastering Diverse Domains through World Models* (2023).
- MAML: Finn et al., *Model-Agnostic Meta-Learning* (2017).
- PEARL: Rakelly et al., *Efficient Off-Policy Meta-RL via Probabilistic Context Variables* (2019).
- Neuromodulation: Soltoggio et al., *Born to Learn* (2018).

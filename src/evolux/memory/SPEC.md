# memory — SPEC

| Field | Value |
| --- | --- |
| Layer | 1 |
| Phase | 1 (working) → 2 (episodic, semantic, hebbian) |
| Depends on | `core`, `tensors` |
| Exposes | `MEMORY_REGISTRY`, concrete `Memory` classes |
| Implements | `evolux.core.protocols.Memory` |

## Mission

Provide all four memory tiers as drop-in implementations of the `Memory`
Protocol. Stateless API: `forward(state, ...) -> state'`. State lives in
batched tensors, not Python attributes.

## Submodules

| File | Class | Phase | Mechanism |
| --- | --- | --- | --- |
| `working.py`  | `WorkingMemory`  | 1 | sliding-window KV cache |
| `episodic.py` | `EpisodicMemory` | 2 | DNC-style content-addressable, soft attention |
| `semantic.py` | `SemanticMemory` | 2 | FAISS / torch index, cross-episode |
| `hebbian.py`  | `HebbianMemory`  | 2 | slow weights, generalised Hebb rule |

## Acceptance tests (per submodule)

- `init_state(B, device)` returns a state dict whose tensors all have leading
  dim B and live on `device`.
- `write(state, k, v)` is pure (returns new state, doesn't mutate).
- For episodic: writing then reading the same key recovers the value with
  cosine-similarity > 0.95 (single-write case).
- `reset_episode(state, mask)` zeroes only the masked envs.
- Hebbian weights are bounded by `w_max` after 10k updates.

## Performance budget

- Episodic write/read at B=256, M=128, D=64: ≤ 1 ms on GPU.
- Semantic retrieval at index_size=10k: ≤ 5 ms.
- Hebbian update B=256, hidden=128, out=128: ≤ 0.5 ms.

## Notes

- DNC reference: Graves et al., *Hybrid computing using a neural network with
  dynamic external memory* (Nature 2016).
- Use FAISS only via `evolux[retrieval]` extra; gracefully fall back to a
  torch-only KNN when FAISS is unavailable.

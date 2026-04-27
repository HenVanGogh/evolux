# genome — SPEC

| Field | Value |
| --- | --- |
| Layer | 1 |
| Phase | 1 (direct) → 3 (hyperneat, indirect) |
| Depends on | `core`, `tensors` |
| Implements | `evolux.core.protocols.Genome`, `GenomeOperator` |

## Mission

Encodings that map a compact "DNA" to a `Brain` + `Morphology`. Each encoding
ships with a matching `GenomeOperator` (mutation / crossover / distance).

## Submodules

| File | Class | Phase | Notes |
| --- | --- | --- | --- |
| `direct.py`    | `DirectGenome`, `DirectOperator`     | 1 | Flat weight vector |
| `neat.py`      | `NEATGenome`, `NEATOperator`         | 3 | Topology + innovation numbers |
| `hyperneat.py` | `HyperNEATGenome`, `HyperNEATOperator` | 3 | CPPN-decoded |
| `indirect.py`  | `IndirectGenome`, `IndirectOperator` | 3 | Latent → diffusion-decoded |

## Common contract

```python
class MyGenome:
    encoding: str
    def decode_brain(self, brain_factory: BrainFactory) -> Brain: ...
    def decode_morphology(self) -> Morphology: ...
    def serialize(self) -> bytes: ...
    @classmethod
    def deserialize(cls, blob: bytes) -> "MyGenome": ...

class MyOperator:
    def mutate(self, g, rng): ...
    def crossover(self, a, b, rng): ...
    def distance(self, a, b) -> float: ...   # for speciation
```

## Acceptance tests

- `serialize → deserialize` round-trips bit-exact.
- Mutation never produces invalid genomes (validator passes).
- `distance(g, g) == 0`; symmetric; non-negative.
- Decoded brains pass `Brain.forward` smoke test on dummy obs.

## Performance budget

- Mutate batch of 256 direct genomes: ≤ 50 ms CPU.
- HyperNEAT decode (CPPN with 100 nodes, output W of 128×128): ≤ 100 ms.

## References

- NEAT: Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies* (2002).
- HyperNEAT: Stanley, D'Ambrosio, Gauci, *A Hypercube-Based Encoding* (2009).
- Indirect via diffusion: Wang et al., *Diffusion as a Generative Model for Neural Network Weights* (2023).

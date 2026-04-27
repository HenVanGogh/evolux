# perception — SPEC

| Field | Value |
| --- | --- |
| Layer | 1 |
| Phase | 1 (vision/proprio) → 2 (chemo, audio) |
| Depends on | `core`, `tensors` |
| Exposes | `PERCEPTION_REGISTRY`, concrete encoder classes |

## Mission

Modality-specific neural encoders that turn raw observations into fixed-size
feature vectors a brain can consume. Each encoder is independently testable.

## Submodules to deliver

- `vision.py` — small CNN (e.g. NatureCNN-lite) for `(B, C, H, W)` images.
- `proprio.py` — MLP for proprioceptive vectors `(B, P)`.
- `chemo.py` — radial-basis encoder for chemical gradient sensing.
- `audio.py` — 1-D conv stack for `(B, C, T)` audio windows.

## Common contract

```python
class Encoder:
    output_dim: int
    def forward(self, x: Tensor) -> Tensor: ...   # (B, ...) → (B, output_dim)
```

Each encoder registers itself: ``@PERCEPTION_REGISTRY.register("vision_cnn_v1")``.

## Acceptance tests

- Output shape matches `output_dim`.
- Gradients flow back to inputs (autograd check).
- CPU forward time meets per-encoder budget in SPEC.

## Performance budget

- VisionCNN B=256, 64×64 RGB: ≤ 3 ms on GPU.
- Proprio MLP B=256, P=64: ≤ 0.2 ms on GPU.

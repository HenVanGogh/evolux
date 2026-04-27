# tensors — SPEC

| Field | Value |
| --- | --- |
| Layer | 0 |
| Phase | 0 |
| Depends on | `core` (types only) |
| Exposes | `assert_finite`, `flatten_batch`, `unflatten_batch`, `one_hot`, `soft_attention` |

## Mission

Pure-function tensor utilities reused by every higher layer: shape gymnastics, attention primitives, NaN/Inf guards. Anything that ends up as a one-liner in three or more modules belongs here.

## Files in scope

```
src/evolux/tensors/
  __init__.py
  batched.py
  SPEC.md
tests/unit/tensors/
  test_*.py
```

## Acceptance tests

- `assert_finite` raises on NaN/Inf, no-op when `EVOLUX_FAST=1`.
- `flatten_batch / unflatten_batch` are exact inverses.
- `soft_attention` matches reference `softmax(QK/τ)V` to 1e-6.
- `one_hot` is differentiable.

## Performance budget

- `soft_attention(B=256, M=128, D=64)` ≤ 0.5 ms on CPU.
- All ops zero-copy where possible.

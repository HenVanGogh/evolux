"""tensors module tests."""

from __future__ import annotations

import math

import pytest
import torch

from evolux.tensors import assert_finite, flatten_batch, one_hot, soft_attention, unflatten_batch


def test_assert_finite_passes() -> None:
    assert_finite(torch.zeros(3))


def test_assert_finite_raises_on_nan() -> None:
    t = torch.tensor([1.0, float("nan")])
    with pytest.raises(ValueError):
        assert_finite(t)


def test_flatten_unflatten_round_trip() -> None:
    t = torch.randn(4, 5, 7)
    flat, orig = flatten_batch(t, n_batch_dims=2)
    assert flat.shape == (20, 7)
    back = unflatten_batch(flat, orig)
    assert torch.equal(back, t)


def test_one_hot_shape_dtype() -> None:
    idx = torch.tensor([[0, 1], [2, 0]])
    h = one_hot(idx, n=3)
    assert h.shape == (2, 2, 3)
    assert h.dtype == torch.float32


def test_soft_attention_recovers_value() -> None:
    # When query == key[0], attention should mostly route to values[0].
    B, M, D, V = 2, 4, 8, 5
    keys = torch.randn(B, M, D)
    values = torch.randn(B, M, V)
    query = keys[:, 0]
    out = soft_attention(query, keys, values, temperature=0.01)
    # With low temperature, output should be close to values[:, 0]
    diff = (out - values[:, 0]).abs().max().item()
    assert diff < 0.5, f"expected near-match, got max-diff={diff}"


def test_soft_attention_uniform_when_keys_equal() -> None:
    B, M, D, V = 1, 3, 4, 2
    keys = torch.ones(B, M, D)
    values = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]])
    query = torch.ones(B, D)
    out = soft_attention(query, keys, values, temperature=1.0)
    expected = torch.tensor([[3.0, 0.0]])  # mean of 1,3,5
    assert math.isclose(out[0, 0].item(), expected[0, 0].item(), abs_tol=1e-5)

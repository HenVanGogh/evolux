"""Tests for Phase-0 tensor ops: masked_mean, masked_softmax, where_done, gather_indices.

Also covers the Phase-0 SPEC acceptance tests:
- one_hot is differentiable
- soft_attention matches reference softmax(QK/τ)V to 1e-6 (normalised inputs)
"""

from __future__ import annotations

import time

import pytest
import torch

from evolux.tensors import (
    gather_indices,
    masked_mean,
    masked_softmax,
    one_hot,
    soft_attention,
    where_done,
)

# ── one_hot acceptance test ──────────────────────────────────────────────────


def test_one_hot_differentiable() -> None:
    """one_hot output should allow gradients to flow from downstream ops."""
    idx = torch.tensor([0, 2, 1])
    h = one_hot(idx, n=4)  # float32 by default
    # Multiply by learnable weights so we can call .backward().
    w = torch.randn(4, requires_grad=True)
    loss = (h * w).sum()
    loss.backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape


# ── soft_attention acceptance test ───────────────────────────────────────────


def test_soft_attention_matches_reference() -> None:
    """With normalised Q and K, cosine-sim attention equals softmax(QK^T/τ)V to 1e-6."""
    torch.manual_seed(0)
    B, M, D, V = 4, 8, 16, 12
    tau = 0.5

    # Use already-normalised inputs so both formulas agree.
    query = torch.nn.functional.normalize(torch.randn(B, D), dim=-1)
    keys = torch.nn.functional.normalize(torch.randn(B, M, D), dim=-1)
    values = torch.randn(B, M, V)

    # Reference: softmax(QK^T / τ) V
    q_exp = query.unsqueeze(1)  # (B, 1, D)
    raw_sims = (q_exp * keys).sum(-1) / tau  # (B, M)
    ref_weights = torch.nn.functional.softmax(raw_sims, dim=-1).unsqueeze(-1)  # (B, M, 1)
    ref_out = (ref_weights * values).sum(1)  # (B, V)

    out = soft_attention(query, keys, values, temperature=tau)
    assert torch.allclose(out, ref_out, atol=1e-6), f"max diff: {(out - ref_out).abs().max()}"


# ── masked_mean ───────────────────────────────────────────────────────────────


def test_masked_mean_shape() -> None:
    x = torch.randn(3, 5, 7)
    mask = torch.ones(3, 5, 7, dtype=torch.bool)
    out = masked_mean(x, mask, dim=1)
    assert out.shape == (3, 7)


def test_masked_mean_correct_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[True, True, False], [True, False, True]])
    out = masked_mean(x, mask, dim=1)
    assert torch.allclose(out[0], torch.tensor(1.5))  # (1+2)/2
    assert torch.allclose(out[1], torch.tensor(5.0))  # (4+6)/2


def test_masked_mean_all_masked_returns_zero() -> None:
    """All-False mask: denominator clamped to 1 → result is zero."""
    x = torch.ones(2, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    out = masked_mean(x, mask, dim=1)
    assert torch.all(out == 0.0)


def test_masked_mean_dtype_preserved() -> None:
    x = torch.randn(4, 6, dtype=torch.float64)
    mask = torch.ones(4, 6, dtype=torch.bool)
    out = masked_mean(x, mask, dim=1)
    assert out.dtype == torch.float64


def test_masked_mean_gradient_flow() -> None:
    x = torch.randn(3, 5, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    out = masked_mean(x, mask, dim=1)
    out.sum().backward()
    assert x.grad is not None


# ── masked_softmax ────────────────────────────────────────────────────────────


def test_masked_softmax_shape() -> None:
    x = torch.randn(4, 8)
    mask = torch.ones(4, 8, dtype=torch.bool)
    out = masked_softmax(x, mask, dim=-1)
    assert out.shape == (4, 8)


def test_masked_softmax_masked_positions_zero() -> None:
    x = torch.randn(2, 5)
    mask = torch.tensor([[True, True, True, False, False], [True, False, True, False, True]])
    out = masked_softmax(x, mask, dim=-1)
    # Masked-out positions should be (near) zero.
    assert out[0, 3].item() == pytest.approx(0.0, abs=1e-6)
    assert out[0, 4].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1, 1].item() == pytest.approx(0.0, abs=1e-6)
    assert out[1, 3].item() == pytest.approx(0.0, abs=1e-6)


def test_masked_softmax_valid_sums_to_one() -> None:
    x = torch.randn(3, 6)
    mask = torch.ones(3, 6, dtype=torch.bool)
    out = masked_softmax(x, mask, dim=-1)
    assert torch.allclose(out.sum(dim=-1), torch.ones(3), atol=1e-6)


def test_masked_softmax_dtype_preserved() -> None:
    x = torch.randn(4, 6, dtype=torch.float32)
    mask = torch.ones(4, 6, dtype=torch.bool)
    out = masked_softmax(x, mask, dim=-1)
    assert out.dtype == torch.float32


def test_masked_softmax_gradient_flow() -> None:
    x = torch.randn(3, 5, requires_grad=True)
    mask = torch.ones(3, 5, dtype=torch.bool)
    out = masked_softmax(x, mask, dim=-1)
    out.sum().backward()
    assert x.grad is not None


# ── where_done ────────────────────────────────────────────────────────────────


def _make_state(B: int = 4) -> dict[str, torch.Tensor]:
    return {
        "h": torch.ones(B, 8),
        "c": torch.ones(B, 8) * 2.0,
    }


def _fresh_state(B: int = 4) -> dict[str, torch.Tensor]:
    return {
        "h": torch.zeros(B, 8),
        "c": torch.zeros(B, 8),
    }


def test_where_done_shape_preserved() -> None:
    B = 4
    state = _make_state(B)
    done = torch.tensor([True, False, True, False])
    out = where_done(state, done, lambda: _fresh_state(B))
    assert out["h"].shape == state["h"].shape
    assert out["c"].shape == state["c"].shape


def test_where_done_done_rows_replaced() -> None:
    B = 4
    state = _make_state(B)
    done = torch.tensor([True, False, True, False])
    out = where_done(state, done, lambda: _fresh_state(B))
    # Rows 0 and 2 should be zeroed (fresh state).
    assert torch.all(out["h"][0] == 0.0)
    assert torch.all(out["h"][2] == 0.0)
    # Rows 1 and 3 should be unchanged.
    assert torch.all(out["h"][1] == 1.0)
    assert torch.all(out["h"][3] == 1.0)


def test_where_done_no_done_returns_same_object() -> None:
    """When no episode is done, the original state dict is returned unchanged."""
    B = 4
    state = _make_state(B)
    done = torch.zeros(B, dtype=torch.bool)
    out = where_done(state, done, lambda: _fresh_state(B))
    assert out is state


def test_where_done_does_not_mutate_input() -> None:
    B = 4
    state = _make_state(B)
    h_clone = state["h"].clone()
    done = torch.tensor([True, False, True, False])
    where_done(state, done, lambda: _fresh_state(B))
    assert torch.equal(state["h"], h_clone)


def test_where_done_gradient_flow() -> None:
    B = 4
    h = torch.ones(B, 8, requires_grad=True)
    state = {"h": h}
    done = torch.tensor([False, False, False, False])
    out = where_done(state, done, lambda: {"h": torch.zeros(B, 8)})
    out["h"].sum().backward()
    assert h.grad is not None


def test_where_done_multidim_state() -> None:
    """where_done must handle state values with more than 2 dims."""
    B = 4
    state = {"k": torch.ones(B, 10, 16)}
    done = torch.tensor([True, False, False, True])
    fresh = {"k": torch.zeros(B, 10, 16)}
    out = where_done(state, done, lambda: fresh)
    assert torch.all(out["k"][0] == 0.0)
    assert torch.all(out["k"][3] == 0.0)
    assert torch.all(out["k"][1] == 1.0)
    assert torch.all(out["k"][2] == 1.0)


# ── gather_indices ────────────────────────────────────────────────────────────


def test_gather_indices_shape() -> None:
    B, T, D = 8, 16, 32
    x = torch.randn(B, T, D)
    idx = torch.randint(0, T, (B,))
    out = gather_indices(x, idx)
    assert out.shape == (B, D)


def test_gather_indices_correct_values() -> None:
    B, T, D = 3, 5, 4
    x = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    idx = torch.tensor([0, 2, 4])
    out = gather_indices(x, idx)
    for b in range(B):
        assert torch.equal(out[b], x[b, idx[b]])


def test_gather_indices_dtype_preserved() -> None:
    B, T, D = 4, 8, 16
    x = torch.randn(B, T, D, dtype=torch.float64)
    idx = torch.zeros(B, dtype=torch.long)
    out = gather_indices(x, idx)
    assert out.dtype == torch.float64


def test_gather_indices_gradient_flow() -> None:
    B, T, D = 4, 6, 8
    x = torch.randn(B, T, D, requires_grad=True)
    idx = torch.tensor([0, 3, 1, 5])
    out = gather_indices(x, idx)
    out.sum().backward()
    assert x.grad is not None
    # Only the gathered positions should have non-zero gradient.
    for b in range(B):
        assert x.grad[b, idx[b]].abs().sum() > 0
        for t in range(T):
            if t != idx[b].item():
                assert torch.all(x.grad[b, t] == 0.0)


# ── 1 000-iteration allocation benchmark ─────────────────────────────────────


@pytest.mark.slow
def test_no_allocation_explosion_1k_iters() -> None:
    """Run 1 000 iterations of all new ops; wall-time must stay reasonable."""
    B, T, D, M, V = 64, 32, 64, 32, 64
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    idx = torch.randint(0, T, (B,))
    q = torch.randn(B, D)
    k = torch.randn(B, M, D)
    v = torch.randn(B, M, V)
    state = {"h": torch.ones(B, D)}
    done = torch.zeros(B, dtype=torch.bool)

    start = time.perf_counter()
    for _ in range(1_000):
        masked_mean(x, mask.unsqueeze(-1).expand_as(x), dim=1)
        masked_softmax(x[:, :, 0], mask, dim=-1)
        gather_indices(x, idx)
        where_done(state, done, lambda: {"h": torch.zeros(B, D)})
        soft_attention(q, k, v)
    elapsed = time.perf_counter() - start
    # 1 000 iterations of light ops should complete well under 10 s on CPU.
    assert elapsed < 10.0, f"1k iters took {elapsed:.2f}s — possible allocation explosion"

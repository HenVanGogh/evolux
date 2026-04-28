"""Unit tests for evolux.memory.hebbian — HebbianMemory (Phase 2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from evolux.core.protocols import Memory
from evolux.core.rng import RNG
from evolux.memory import MEMORY_REGISTRY
from evolux.memory.hebbian import HebbianMemory

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def dim() -> int:
    return 16


@pytest.fixture
def mem(dim: int) -> HebbianMemory:
    return HebbianMemory(dim=dim)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_isinstance_memory_protocol(mem: HebbianMemory) -> None:
    """HebbianMemory must satisfy the Memory Protocol at runtime."""
    assert isinstance(mem, Memory)


def test_registered_in_memory_registry() -> None:
    """'hebbian' key must be present in MEMORY_REGISTRY."""
    assert "hebbian" in MEMORY_REGISTRY
    assert MEMORY_REGISTRY.get("hebbian") is HebbianMemory


# ── init_state ────────────────────────────────────────────────────────────────


def test_init_state_shape(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    B = 4
    state = mem.init_state(B, cpu)
    assert state["W"].shape == (B, dim, dim)


def test_init_state_device(mem: HebbianMemory, cpu: torch.device) -> None:
    state = mem.init_state(3, cpu)
    assert state["W"].device.type == "cpu"


def test_init_state_zeros(mem: HebbianMemory, cpu: torch.device) -> None:
    state = mem.init_state(2, cpu)
    assert state["W"].sum().item() == 0.0


# ── write ─────────────────────────────────────────────────────────────────────


def test_write_returns_new_state(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """write must be pure: returned state is a new dict."""
    B = 2
    state = mem.init_state(B, cpu)
    key = torch.randn(B, dim)
    val = torch.randn(B, dim)

    new_state = mem.write(state, key, val)

    assert new_state is not state
    assert new_state["W"] is not state["W"]


def test_write_does_not_mutate_original(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """Original state tensors must be unchanged after write."""
    B = 2
    state = mem.init_state(B, cpu)
    W_before = state["W"].clone()

    key = torch.randn(B, dim)
    val = torch.randn(B, dim)
    mem.write(state, key, val)

    assert torch.equal(state["W"], W_before)


def test_write_accumulates_association(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """After a write the weight matrix must be non-zero."""
    B = 3
    state = mem.init_state(B, cpu)
    state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))
    assert state["W"].abs().sum().item() > 0.0


def test_write_clamps_weights(cpu: torch.device, dim: int) -> None:
    """Weights must remain bounded by w_max after many writes."""
    B = 4
    w_max = 2.0
    mem = HebbianMemory(dim=dim, w_max=w_max)
    # Override learnable params with large values to stress-test clamping.
    with torch.no_grad():
        mem.log_lr.fill_(10.0)
        mem.log_decay.fill_(-10.0)  # decay ≈ 0 → no forgetting

    state = mem.init_state(B, cpu)
    for _ in range(100):
        state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))

    assert state["W"].abs().max().item() <= w_max + 1e-6


# ── read ──────────────────────────────────────────────────────────────────────


def test_read_output_shape(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    B = 5
    state = mem.init_state(B, cpu)
    query = torch.randn(B, dim)
    out = mem.read(state, query)
    assert out.shape == (B, dim)


def test_single_write_then_read_cosine(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """Reading with the same key that was written should recover the value direction.

    With a single write and zero initial W:
        W = lr * value ⊗ key
        W @ key = lr * value * ‖key‖²

    The output shares the same direction as ``value`` → cosine similarity ≈ 1.0.
    """
    B = 4
    state = mem.init_state(B, cpu)

    key = torch.randn(B, dim)
    val = torch.randn(B, dim)
    state = mem.write(state, key, val)

    out = mem.read(state, key)  # (B, D)

    cos = F.cosine_similarity(out, val, dim=-1)  # (B,)
    assert cos.min().item() >= 0.95, f"min cosine {cos.min().item():.4f} < 0.95"


# ── decay ─────────────────────────────────────────────────────────────────────


def test_decay_reduces_old_writes(cpu: torch.device, dim: int) -> None:
    """After the initial write is followed by zero-key writes, the W norm shrinks."""
    B = 2
    # Use high decay so the effect is clearly visible.
    mem = HebbianMemory(dim=dim)
    with torch.no_grad():
        # decay = sigmoid(large) ≈ 0.99, lr = softplus(small) ≈ 0.01
        mem.log_decay.fill_(4.6)  # sigmoid(4.6) ≈ 0.99
        mem.log_lr.fill_(-4.6)  # softplus(-4.6) ≈ 0.01

    state = mem.init_state(B, cpu)
    state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))
    norm_after_write = state["W"].norm().item()

    # Subsequent writes with zero key/value — only decay applies.
    zeros = torch.zeros(B, dim)
    for _ in range(20):
        state = mem.write(state, zeros, zeros)

    norm_after_decay = state["W"].norm().item()
    assert norm_after_decay < norm_after_write, (
        f"W norm did not decrease: {norm_after_decay:.4f} >= {norm_after_write:.4f}"
    )


# ── gradients ─────────────────────────────────────────────────────────────────


def test_gradients_flow_to_learnable_params(
    mem: HebbianMemory, cpu: torch.device, dim: int
) -> None:
    """A backward pass must propagate gradients to log_lr and log_decay.

    Two writes are required:
    - log_lr gradient comes from the lr * outer term (present after the first
      write).
    - log_decay gradient comes from the (1 - decay) * W term, which is non-zero
      only when W has already been updated (i.e., after the second write).
    """
    B = 4
    state = mem.init_state(B, cpu)

    key1 = torch.randn(B, dim)
    val1 = torch.randn(B, dim)
    state = mem.write(state, key1, val1)  # builds non-zero W

    key2 = torch.randn(B, dim)
    val2 = torch.randn(B, dim)
    state = mem.write(state, key2, val2)  # decay acts on non-zero W → grad flows

    out = mem.read(state, key2)
    loss = out.sum()
    loss.backward()

    assert mem.log_lr.grad is not None, "no gradient for log_lr"
    assert mem.log_decay.grad is not None, "no gradient for log_decay"
    assert mem.log_lr.grad.abs().item() > 0.0, "zero gradient for log_lr"
    assert mem.log_decay.grad.abs().item() > 0.0, "zero gradient for log_decay"


# ── determinism ───────────────────────────────────────────────────────────────


def test_determinism_with_fixed_rng(cpu: torch.device, dim: int) -> None:
    """Same RNG seed must produce identical outputs."""
    B = 3

    def _run(seed: int) -> torch.Tensor:
        rng = RNG(seed=seed)
        gen = rng.split("hebbian_test")
        mem = HebbianMemory(dim=dim)
        state = mem.init_state(B, cpu)
        key = torch.zeros(B, dim).normal_(generator=gen)
        val = torch.zeros(B, dim).normal_(generator=gen)
        state = mem.write(state, key, val)
        query = torch.zeros(B, dim).normal_(generator=gen)
        return mem.read(state, query)

    out_a = _run(42)
    out_b = _run(42)
    assert torch.equal(out_a, out_b), "outputs differ despite identical RNG seed"


# ── batched shapes ────────────────────────────────────────────────────────────


def test_shapes_preserved_batched(cpu: torch.device, dim: int) -> None:
    """Shapes must be correct for various batch sizes."""
    for B in (1, 4, 16, 64):
        mem = HebbianMemory(dim=dim)
        state = mem.init_state(B, cpu)
        assert state["W"].shape == (B, dim, dim)

        key = torch.randn(B, dim)
        val = torch.randn(B, dim)
        state = mem.write(state, key, val)
        assert state["W"].shape == (B, dim, dim)

        out = mem.read(state, key)
        assert out.shape == (B, dim)


# ── reset_episode ─────────────────────────────────────────────────────────────


def test_reset_episode_all(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """reset_episode(state, mask=None) should zero all W."""
    B = 3
    state = mem.init_state(B, cpu)
    for _ in range(3):
        state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))

    reset = mem.reset_episode(state)
    assert reset["W"].sum().item() == 0.0


def test_reset_episode_partial(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """reset_episode should only zero rows where mask is True."""
    B = 4
    state = mem.init_state(B, cpu)
    state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))

    mask = torch.tensor([True, False, True, False])
    reset = mem.reset_episode(state, mask)

    # Masked rows (0, 2) should be zeroed.
    assert reset["W"][0].sum().item() == 0.0
    assert reset["W"][2].sum().item() == 0.0

    # Unmasked rows (1, 3) should still hold the written data.
    assert torch.allclose(reset["W"][1], state["W"][1])
    assert torch.allclose(reset["W"][3], state["W"][3])


def test_reset_episode_does_not_mutate(mem: HebbianMemory, cpu: torch.device, dim: int) -> None:
    """reset_episode must be pure."""
    B = 2
    state = mem.init_state(B, cpu)
    state = mem.write(state, torch.randn(B, dim), torch.randn(B, dim))

    W_before = state["W"].clone()
    mask = torch.tensor([True, False])
    mem.reset_episode(state, mask)

    assert torch.equal(state["W"], W_before)

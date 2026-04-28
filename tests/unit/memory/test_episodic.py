"""Unit tests for evolux.memory.episodic — EpisodicMemory (Phase 2 DNC-style)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from evolux.core.protocols import Memory
from evolux.core.rng import RNG
from evolux.memory import MEMORY_REGISTRY
from evolux.memory.episodic import EpisodicMemory

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum cosine similarity expected after a single write then read with the same key.
_SINGLE_WRITE_COS_THRESHOLD = 0.95
# Minimum cosine similarity expected after writing a target + several unrelated entries.
# Lower than the single-write case because content is partially overwritten.
_MULTI_WRITE_COS_THRESHOLD = 0.3

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mem() -> EpisodicMemory:
    return EpisodicMemory(capacity=16, key_dim=32, val_dim=32)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_isinstance_memory_protocol(mem: EpisodicMemory) -> None:
    """EpisodicMemory must satisfy the Memory Protocol at runtime."""
    assert isinstance(mem, Memory)


def test_registered_in_memory_registry() -> None:
    """'episodic' key must be present in MEMORY_REGISTRY."""
    assert "episodic" in MEMORY_REGISTRY
    assert MEMORY_REGISTRY.get("episodic") is EpisodicMemory


# ── init_state ────────────────────────────────────────────────────────────────


def test_init_state_shapes(mem: EpisodicMemory, cpu: torch.device) -> None:
    B, N, D, V = 4, mem.capacity, mem.key_dim, mem.val_dim
    state = mem.init_state(B, cpu)

    assert state["M"].shape == (B, N, V)
    assert state["keys"].shape == (B, N, D)
    assert state["usage"].shape == (B, N)


def test_init_state_zeros(mem: EpisodicMemory, cpu: torch.device) -> None:
    state = mem.init_state(3, cpu)
    assert state["M"].sum().item() == 0.0
    assert state["keys"].sum().item() == 0.0
    assert state["usage"].sum().item() == 0.0


def test_init_state_device(mem: EpisodicMemory, cpu: torch.device) -> None:
    state = mem.init_state(2, cpu)
    for v in state.values():
        assert v.device.type == "cpu"


# ── write ─────────────────────────────────────────────────────────────────────


def test_write_returns_new_state(mem: EpisodicMemory, cpu: torch.device) -> None:
    """write must be pure: returned state is a new dict."""
    B = 2
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)

    new_state = mem.write(state, key, val)

    assert new_state is not state
    assert new_state["M"] is not state["M"]
    assert new_state["keys"] is not state["keys"]
    assert new_state["usage"] is not state["usage"]


def test_write_does_not_mutate_original(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Original state tensors must be unchanged after write."""
    B = 2
    state = mem.init_state(B, cpu)
    M_before = state["M"].clone()
    keys_before = state["keys"].clone()
    usage_before = state["usage"].clone()

    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    mem.write(state, key, val)

    assert torch.equal(state["M"], M_before)
    assert torch.equal(state["keys"], keys_before)
    assert torch.equal(state["usage"], usage_before)


# ── write → read round-trip ───────────────────────────────────────────────────


def test_write_read_roundtrip(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Reading with the exact write key should recover the written value (cosine ≥ 0.95)."""
    B = 4
    state = mem.init_state(B, cpu)

    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    out = mem.read(state, key)

    cos = F.cosine_similarity(out, val, dim=-1)  # (B,)
    assert cos.min().item() >= _SINGLE_WRITE_COS_THRESHOLD, (
        f"min cosine {cos.min().item():.4f} < {_SINGLE_WRITE_COS_THRESHOLD}"
    )


def test_write_read_roundtrip_multiple_writes(mem: EpisodicMemory, cpu: torch.device) -> None:
    """After multiple writes the target key should still be retrievable."""
    B = 2
    state = mem.init_state(B, cpu)

    target_key = torch.randn(B, mem.key_dim)
    target_val = torch.randn(B, mem.val_dim)
    state = mem.write(state, target_key, target_val)

    # Write a handful of unrelated entries.
    for _ in range(4):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    out = mem.read(state, target_key)
    cos = F.cosine_similarity(out, target_val, dim=-1)
    assert cos.min().item() >= _MULTI_WRITE_COS_THRESHOLD, (
        f"min cosine after extra writes: {cos.min().item():.4f} < {_MULTI_WRITE_COS_THRESHOLD}"
    )


# ── Gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_flow_through_write(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Gradients must propagate back through the write operation."""
    B = 2
    state = mem.init_state(B, cpu)

    key = torch.randn(B, mem.key_dim, requires_grad=True)
    val = torch.randn(B, mem.val_dim, requires_grad=True)

    new_state = mem.write(state, key, val)
    loss = new_state["M"].sum() + new_state["keys"].sum()
    loss.backward()

    assert key.grad is not None, "No gradient on write key"
    assert val.grad is not None, "No gradient on write value"
    assert key.grad.abs().sum().item() > 0.0
    assert val.grad.abs().sum().item() > 0.0


def test_gradients_flow_through_read(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Gradients must propagate back through the read query (non-uniform memory)."""
    B = 2
    N = mem.capacity
    # Manually populate state with non-uniform content so that the read output
    # depends on the query direction (uniform slots would make it invariant).
    state = {
        "M": torch.randn(B, N, mem.val_dim),
        "keys": torch.randn(B, N, mem.key_dim),
        "usage": torch.rand(B, N),
    }

    query = torch.randn(B, mem.key_dim, requires_grad=True)
    out = mem.read(state, query)
    out.sum().backward()

    assert query.grad is not None, "No gradient on read query"
    assert query.grad.abs().sum().item() > 0.0


def test_gradients_flow_through_state_m(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Gradients must propagate through both write and read in a full pass."""
    B = 2
    state = mem.init_state(B, cpu)

    key = torch.randn(B, mem.key_dim, requires_grad=True)
    val = torch.randn(B, mem.val_dim, requires_grad=True)
    new_state = mem.write(state, key, val)

    query = torch.randn(B, mem.key_dim)
    out = mem.read(new_state, query)
    out.sum().backward()

    assert val.grad is not None, "Gradient did not flow from read back through write"
    assert val.grad.abs().sum().item() > 0.0


# ── Usage vector ──────────────────────────────────────────────────────────────


def test_usage_grows_monotonically(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Usage sum must be non-decreasing across sequential writes."""
    B = 1
    state = mem.init_state(B, cpu)
    prev_total = state["usage"].sum().item()

    for _ in range(mem.capacity):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))
        total = state["usage"].sum().item()
        assert total >= prev_total - 1e-6, f"Usage decreased: {total:.6f} < {prev_total:.6f}"
        prev_total = total


def test_usage_stays_in_bounds(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Usage values must stay in [0, 1] regardless of the number of writes."""
    B = 2
    state = mem.init_state(B, cpu)

    for _ in range(mem.capacity * 3):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    assert state["usage"].min().item() >= 0.0
    assert state["usage"].max().item() <= 1.0 + 1e-6


# ── Determinism ───────────────────────────────────────────────────────────────


def test_determinism_with_fixed_rng(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Two runs with the same RNG seed must produce identical outputs."""

    def _run(seed: int) -> Tensor:
        rng = RNG(seed=seed)
        gen = rng.split("episodic_test")
        B = 3
        state = mem.init_state(B, cpu)
        for _ in range(4):
            k = torch.zeros(B, mem.key_dim).normal_(generator=gen)
            v = torch.zeros(B, mem.val_dim).normal_(generator=gen)
            state = mem.write(state, k, v)
        q = torch.zeros(B, mem.key_dim).normal_(generator=gen)
        return mem.read(state, q)

    out1 = _run(42)
    out2 = _run(42)
    assert torch.allclose(out1, out2), "Outputs differ between identical runs"


def test_different_seeds_produce_different_output(mem: EpisodicMemory, cpu: torch.device) -> None:
    """Different seeds must (overwhelmingly) produce different outputs."""

    def _run(seed: int) -> Tensor:
        rng = RNG(seed=seed)
        gen = rng.split("episodic_test")
        B = 1
        state = mem.init_state(B, cpu)
        k = torch.zeros(B, mem.key_dim).normal_(generator=gen)
        v = torch.zeros(B, mem.val_dim).normal_(generator=gen)
        state = mem.write(state, k, v)
        return mem.read(state, k)

    assert not torch.allclose(_run(0), _run(1))


# ── Shape preservation (batched) ──────────────────────────────────────────────


def test_batched_shapes_preserved(cpu: torch.device) -> None:
    """Output shapes must match spec across various batch sizes."""
    for B in (1, 4, 16):
        mem = EpisodicMemory(capacity=8, key_dim=24, val_dim=48)
        state = mem.init_state(B, cpu)

        key = torch.randn(B, mem.key_dim)
        val = torch.randn(B, mem.val_dim)
        new_state = mem.write(state, key, val)

        assert new_state["M"].shape == (B, mem.capacity, mem.val_dim)
        assert new_state["keys"].shape == (B, mem.capacity, mem.key_dim)
        assert new_state["usage"].shape == (B, mem.capacity)

        out = mem.read(new_state, key)
        assert out.shape == (B, mem.val_dim)


# ── reset_episode ─────────────────────────────────────────────────────────────


def test_reset_episode_all(mem: EpisodicMemory, cpu: torch.device) -> None:
    """reset_episode(state, mask=None) should zero all environments."""
    B = 3
    state = mem.init_state(B, cpu)
    for _ in range(3):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    reset = mem.reset_episode(state)
    assert reset["M"].sum().item() == 0.0
    assert reset["keys"].sum().item() == 0.0
    assert reset["usage"].sum().item() == 0.0


def test_reset_episode_partial(mem: EpisodicMemory, cpu: torch.device) -> None:
    """reset_episode should only zero rows where mask is True."""
    B = 4
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    mask = torch.tensor([True, False, True, False])
    reset = mem.reset_episode(state, mask)

    # Masked rows (0, 2) should be zeroed.
    assert reset["M"][0].sum().item() == 0.0
    assert reset["M"][2].sum().item() == 0.0
    assert reset["usage"][0].sum().item() == 0.0
    assert reset["usage"][2].sum().item() == 0.0

    # Unmasked rows (1, 3) should retain their data.
    assert torch.allclose(reset["M"][1], state["M"][1])
    assert torch.allclose(reset["M"][3], state["M"][3])


def test_reset_episode_does_not_mutate(mem: EpisodicMemory, cpu: torch.device) -> None:
    """reset_episode must be pure (does not mutate input state)."""
    B = 2
    state = mem.init_state(B, cpu)
    state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    M_before = state["M"].clone()
    mask = torch.tensor([True, False])
    mem.reset_episode(state, mask)

    assert torch.equal(state["M"], M_before)


# ── top_k read ────────────────────────────────────────────────────────────────


def test_read_top_k_shape(mem: EpisodicMemory, cpu: torch.device) -> None:
    """read with top_k > 1 must still return (B, D_val)."""
    B = 3
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    out = mem.read(state, key, top_k=4)
    assert out.shape == (B, mem.val_dim)

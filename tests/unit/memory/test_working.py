"""Unit tests for evolux.memory.working — WorkingMemory (Phase 1)."""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from evolux.core.protocols import Memory
from evolux.memory import MEMORY_REGISTRY
from evolux.memory.working import WorkingMemory

# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def mem() -> WorkingMemory:
    return WorkingMemory(capacity=8, key_dim=16, val_dim=16)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


# ── Protocol conformance ─────────────────────────────────────────────────────


def test_isinstance_memory_protocol(mem: WorkingMemory) -> None:
    """WorkingMemory must satisfy the Memory Protocol at runtime."""
    assert isinstance(mem, Memory)


def test_registered_in_memory_registry() -> None:
    """'working_v1' key must be present in MEMORY_REGISTRY."""
    assert "working_v1" in MEMORY_REGISTRY
    assert MEMORY_REGISTRY.get("working_v1") is WorkingMemory


# ── init_state ───────────────────────────────────────────────────────────────


def test_init_state_shapes(mem: WorkingMemory, cpu: torch.device) -> None:
    B, M, D, V = 4, mem.capacity, mem.key_dim, mem.val_dim
    state = mem.init_state(B, cpu)

    assert state["keys"].shape == (B, M, D)
    assert state["values"].shape == (B, M, V)
    assert state["ptr"].shape == (B,)


def test_init_state_device(mem: WorkingMemory, cpu: torch.device) -> None:
    state = mem.init_state(3, cpu)
    for v in state.values():
        assert v.device.type == "cpu"


def test_init_state_zeros(mem: WorkingMemory, cpu: torch.device) -> None:
    state = mem.init_state(2, cpu)
    assert state["keys"].sum().item() == 0.0
    assert state["values"].sum().item() == 0.0
    assert state["ptr"].sum().item() == 0


# ── write ────────────────────────────────────────────────────────────────────


def test_write_returns_new_state(mem: WorkingMemory, cpu: torch.device) -> None:
    """write must be pure: returned state is a new dict, not the original."""
    B = 2
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)

    new_state = mem.write(state, key, val)

    assert new_state is not state
    assert new_state["keys"] is not state["keys"]
    assert new_state["values"] is not state["values"]


def test_write_does_not_mutate_original(mem: WorkingMemory, cpu: torch.device) -> None:
    """Original state tensors must be unchanged after write."""
    B = 2
    state = mem.init_state(B, cpu)
    keys_before = state["keys"].clone()
    values_before = state["values"].clone()
    ptr_before = state["ptr"].clone()

    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    mem.write(state, key, val)

    assert torch.equal(state["keys"], keys_before)
    assert torch.equal(state["values"], values_before)
    assert torch.equal(state["ptr"], ptr_before)


def test_write_advances_pointer(mem: WorkingMemory, cpu: torch.device) -> None:
    B = 3
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    new_state = mem.write(state, key, val)

    assert torch.all(new_state["ptr"] == 1)


def test_write_pointer_wraps(mem: WorkingMemory, cpu: torch.device) -> None:
    """Pointer wraps around after filling the buffer."""
    B = 1
    state = mem.init_state(B, cpu)
    for _ in range(mem.capacity):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    # After M writes, ptr should be back at 0.
    assert torch.all(state["ptr"] == 0)


# ── read ─────────────────────────────────────────────────────────────────────


def test_read_output_shape(mem: WorkingMemory, cpu: torch.device) -> None:
    B = 5
    state = mem.init_state(B, cpu)
    query = torch.randn(B, mem.key_dim)
    out = mem.read(state, query)

    assert out.shape == (B, mem.val_dim)


def test_single_write_then_read_cosine(mem: WorkingMemory, cpu: torch.device) -> None:
    """Reading with the same key that was written should recover value with cosine ≥ 0.95."""
    B = 4
    state = mem.init_state(B, cpu)

    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    out = mem.read(state, key)

    # Compute per-batch cosine similarity between out and val.
    cos = F.cosine_similarity(out, val, dim=-1)  # (B,)
    assert cos.min().item() >= 0.95, f"min cosine similarity {cos.min().item():.4f} < 0.95"


# ── reset_episode ─────────────────────────────────────────────────────────────


def test_reset_episode_all(mem: WorkingMemory, cpu: torch.device) -> None:
    """reset_episode(state, mask=None) should zero all envs."""
    B = 3
    state = mem.init_state(B, cpu)
    for _ in range(3):
        state = mem.write(state, torch.randn(B, mem.key_dim), torch.randn(B, mem.val_dim))

    reset = mem.reset_episode(state)
    assert reset["keys"].sum().item() == 0.0
    assert reset["values"].sum().item() == 0.0
    assert reset["ptr"].sum().item() == 0


def test_reset_episode_partial(mem: WorkingMemory, cpu: torch.device) -> None:
    """reset_episode should only zero rows where mask is True."""
    B = 4
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    mask = torch.tensor([True, False, True, False])
    reset = mem.reset_episode(state, mask)

    # Masked rows (0, 2) should be zeroed.
    assert reset["keys"][0].sum().item() == 0.0
    assert reset["keys"][2].sum().item() == 0.0
    assert reset["ptr"][0].item() == 0
    assert reset["ptr"][2].item() == 0

    # Unmasked rows (1, 3) should still hold the written data.
    assert torch.allclose(reset["keys"][1], state["keys"][1])
    assert torch.allclose(reset["keys"][3], state["keys"][3])


def test_reset_episode_does_not_mutate(mem: WorkingMemory, cpu: torch.device) -> None:
    """reset_episode must be pure."""
    B = 2
    state = mem.init_state(B, cpu)
    key = torch.randn(B, mem.key_dim)
    val = torch.randn(B, mem.val_dim)
    state = mem.write(state, key, val)

    keys_before = state["keys"].clone()
    mask = torch.tensor([True, False])
    mem.reset_episode(state, mask)

    assert torch.equal(state["keys"], keys_before)


# ── Benchmark ────────────────────────────────────────────────────────────────


@pytest.mark.benchmark
def test_bench_write_read_cpu() -> None:
    """Report write+read latency at B=256, M=128, D=64, V=64 on CPU.

    The SPEC.md performance budget covers Phase-2 episodic memory on GPU.
    No explicit CPU timing target is specified for Phase-1 WorkingMemory; we
    enforce a 20 ms ceiling to catch accidental regressions without
    over-constraining CI hardware.
    """
    B, M, D, V = 256, 128, 64, 64
    mem = WorkingMemory(capacity=M, key_dim=D, val_dim=V)
    cpu = torch.device("cpu")
    state = mem.init_state(B, cpu)

    key = torch.randn(B, D)
    val = torch.randn(B, V)
    query = torch.randn(B, D)

    # Warm-up.
    for _ in range(5):
        s = mem.write(state, key, val)
        mem.read(s, query)

    N = 20
    start = time.perf_counter()
    for _ in range(N):
        s = mem.write(state, key, val)
        mem.read(s, query)
    elapsed_ms = (time.perf_counter() - start) / N * 1000

    assert elapsed_ms <= 20.0, f"write+read took {elapsed_ms:.3f} ms (ceiling: 20 ms)"

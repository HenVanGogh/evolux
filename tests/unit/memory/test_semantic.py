"""Unit tests for evolux.memory.semantic — SemanticMemory (Phase 2)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

faiss = pytest.importorskip("faiss")  # skip entire module if faiss-cpu is absent

from evolux.core.protocols import Memory  # noqa: E402
from evolux.core.rng import RNG  # noqa: E402
from evolux.memory import MEMORY_REGISTRY  # noqa: E402
from evolux.memory.semantic import SemanticMemory  # noqa: E402

# ── Fixtures ─────────────────────────────────────────────────────────────────

KEY_DIM = 32
VAL_DIM = 16


@pytest.fixture
def mem() -> SemanticMemory:
    return SemanticMemory(capacity=64, key_dim=KEY_DIM, val_dim=VAL_DIM)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_isinstance_memory_protocol(mem: SemanticMemory) -> None:
    """SemanticMemory must satisfy the Memory Protocol at runtime."""
    assert isinstance(mem, Memory)


def test_registered_in_memory_registry() -> None:
    """'semantic' key must be present in MEMORY_REGISTRY."""
    assert "semantic" in MEMORY_REGISTRY
    assert MEMORY_REGISTRY.get("semantic") is SemanticMemory


# ── init_state ────────────────────────────────────────────────────────────────


def test_init_state_device(mem: SemanticMemory, cpu: torch.device) -> None:
    state = mem.init_state(4, cpu)
    for v in state.values():
        assert v.device.type == "cpu"


def test_init_state_zeros(mem: SemanticMemory, cpu: torch.device) -> None:
    state = mem.init_state(2, cpu)
    assert state["n"].item() == 0


# ── Acceptance test 1: write 8 pairs → read → cosine-similar ─────────────────


def test_write_read_cosine_similarity(mem: SemanticMemory, cpu: torch.device) -> None:
    """Writing 8 distinct (key, value) pairs, then reading with one of the keys,
    should retrieve a value that is cosine-similar (> 0.5) to the stored value.
    """
    N = 8
    state = mem.init_state(1, cpu)

    keys = F.normalize(torch.randn(N, KEY_DIM), dim=-1)  # (N, D)
    values = torch.randn(N, VAL_DIM)  # (N, V)

    for i in range(N):
        state = mem.write(state, keys[i].unsqueeze(0), values[i].unsqueeze(0))

    # Read using key[3] — the exact stored key
    query = keys[3].unsqueeze(0)  # (1, D)
    out = mem.read(state, query, top_k=4)  # (1, V)

    cos = F.cosine_similarity(out, values[3].unsqueeze(0), dim=-1)
    assert cos.item() > 0.5, f"cosine similarity {cos.item():.4f} ≤ 0.5"


# ── Acceptance test 2: gradient flow ─────────────────────────────────────────


def test_gradient_flow_through_soft_attention(mem: SemanticMemory, cpu: torch.device) -> None:
    """Gradients must flow from the read output back through the soft-attention
    weights (i.e. through the differentiable score → softmax → weighted-sum path).
    """
    state = mem.init_state(1, cpu)

    # Write a few entries
    for _ in range(4):
        k = torch.randn(1, KEY_DIM)
        v = torch.randn(1, VAL_DIM)
        state = mem.write(state, k, v)

    query = torch.randn(1, KEY_DIM, requires_grad=True)
    out = mem.read(state, query, top_k=4)
    out.sum().backward()

    assert query.grad is not None, "query.grad is None — no gradient flowed"
    assert not torch.all(query.grad == 0), "query.grad is all-zeros"


# ── Acceptance test 3: determinism with fixed RNG ────────────────────────────


def test_determinism_with_fixed_rng(cpu: torch.device) -> None:
    """Two SemanticMemory instances with the same deterministic data must produce
    identical outputs.
    """
    rng = RNG(seed=42)

    def _build_and_read() -> torch.Tensor:
        m = SemanticMemory(capacity=32, key_dim=KEY_DIM, val_dim=VAL_DIM)
        state = m.init_state(1, cpu)
        gen = rng.split("semantic_test")
        for _ in range(6):
            k = torch.randn(1, KEY_DIM, generator=gen)
            v = torch.randn(1, VAL_DIM, generator=gen)
            state = m.write(state, k, v)
        query = torch.zeros(1, KEY_DIM)
        query[0, 0] = 1.0
        return m.read(state, query, top_k=4)

    out1 = _build_and_read()
    out2 = _build_and_read()

    assert torch.allclose(out1, out2), "Outputs differ despite identical inputs"


# ── Acceptance test 4: shapes preserved across batched call ──────────────────


def test_batched_read_shapes(mem: SemanticMemory, cpu: torch.device) -> None:
    """B simultaneous reads must return shape (B, val_dim)."""
    B = 5
    state = mem.init_state(1, cpu)

    # Populate the store
    for _ in range(8):
        state = mem.write(state, torch.randn(1, KEY_DIM), torch.randn(1, VAL_DIM))

    queries = torch.randn(B, KEY_DIM)  # (B, D)
    out = mem.read(state, queries, top_k=4)

    assert out.shape == (B, VAL_DIM), f"Expected ({B}, {VAL_DIM}), got {out.shape}"


def test_batched_write_then_read(mem: SemanticMemory, cpu: torch.device) -> None:
    """Writing a batch of B pairs at once and reading with matching queries
    should return sensible results (no crash, correct shape).
    """
    B = 4
    state = mem.init_state(1, cpu)

    keys = F.normalize(torch.randn(B, KEY_DIM), dim=-1)
    values = torch.randn(B, VAL_DIM)
    state = mem.write(state, keys, values)  # single batched write

    out = mem.read(state, keys, top_k=4)
    assert out.shape == (B, VAL_DIM)


# ── Acceptance test 5: empty store + read returns zeros ──────────────────────


def test_empty_store_read_returns_zeros(mem: SemanticMemory, cpu: torch.device) -> None:
    """Reading from an empty store must return zero tensor without crashing."""
    state = mem.init_state(1, cpu)
    query = torch.randn(1, KEY_DIM)
    out = mem.read(state, query)

    assert out.shape == (1, VAL_DIM)
    assert torch.allclose(out, torch.zeros_like(out))


def test_empty_store_batched_read_returns_zeros(mem: SemanticMemory, cpu: torch.device) -> None:
    B = 3
    state = mem.init_state(1, cpu)
    queries = torch.randn(B, KEY_DIM)
    out = mem.read(state, queries)

    assert out.shape == (B, VAL_DIM)
    assert torch.allclose(out, torch.zeros_like(out))


# ── Additional correctness tests ─────────────────────────────────────────────


def test_write_increments_n(mem: SemanticMemory, cpu: torch.device) -> None:
    """state['n'] should track the number of written pairs."""
    state = mem.init_state(1, cpu)
    assert state["n"].item() == 0

    state = mem.write(state, torch.randn(1, KEY_DIM), torch.randn(1, VAL_DIM))
    assert state["n"].item() == 1

    state = mem.write(state, torch.randn(3, KEY_DIM), torch.randn(3, VAL_DIM))
    assert state["n"].item() == 4


def test_write_is_functional(mem: SemanticMemory, cpu: torch.device) -> None:
    """write must return a new dict, not mutate the input state."""
    state = mem.init_state(1, cpu)
    n_before = state["n"].clone()

    new_state = mem.write(state, torch.randn(1, KEY_DIM), torch.randn(1, VAL_DIM))

    assert new_state is not state
    assert torch.equal(state["n"], n_before), "write mutated the input state"


def test_reset_episode_returns_new_state(mem: SemanticMemory, cpu: torch.device) -> None:
    """reset_episode must return a new dict without mutating the input."""
    state = mem.init_state(1, cpu)
    state = mem.write(state, torch.randn(1, KEY_DIM), torch.randn(1, VAL_DIM))
    n_before = state["n"].clone()

    reset = mem.reset_episode(state)

    assert reset is not state
    assert torch.equal(state["n"], n_before), "reset_episode mutated the input state"


def test_top_k_clamped_to_store_size(mem: SemanticMemory, cpu: torch.device) -> None:
    """Requesting top_k > store size must not crash."""
    state = mem.init_state(1, cpu)
    state = mem.write(state, torch.randn(2, KEY_DIM), torch.randn(2, VAL_DIM))

    out = mem.read(state, torch.randn(1, KEY_DIM), top_k=100)
    assert out.shape == (1, VAL_DIM)

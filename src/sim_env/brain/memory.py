"""Memory systems available to the brain.

Three tiers
-----------
WorkingMemory    — fixed-size ring buffer of float vectors; O(1) read/write
EpisodicMemory   — differentiable key-value store (content-addressed)
HebbianMemory    — slow-weight associative matrix updated by Hebbian rule
"""

from __future__ import annotations

import numpy as np


# ── Working memory ───────────────────────────────────────────────────────────

class WorkingMemory:
    """Fixed-size circular buffer of scalar slots.

    The brain can read slot *i* and write slot *i* each step, giving it an
    explicit short-term store that persists across forward passes within an
    episode.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._buf: np.ndarray = np.zeros(size, dtype=np.float32)

    def read(self) -> np.ndarray:
        return self._buf.copy()

    def write(self, values: np.ndarray) -> None:
        if values.shape[0] != self.size:
            raise ValueError(f"Expected {self.size} values, got {values.shape[0]}")
        np.copyto(self._buf, values.astype(np.float32))

    def write_slot(self, idx: int, value: float) -> None:
        self._buf[idx % self.size] = value

    def read_slot(self, idx: int) -> float:
        return float(self._buf[idx % self.size])

    def reset(self) -> None:
        self._buf[:] = 0.0

    def __repr__(self) -> str:
        return f"WorkingMemory(size={self.size})"


# ── Episodic memory ──────────────────────────────────────────────────────────

class EpisodicMemory:
    """Content-addressable key-value memory (soft attention retrieval).

    Keys and values are float vectors. A query retrieves a weighted average
    of all stored values proportional to cosine similarity with their keys.

    This is a simplified Neural Turing Machine / DNC-style external memory.
    """

    def __init__(self, capacity: int, key_dim: int, val_dim: int) -> None:
        self.capacity = capacity
        self.key_dim = key_dim
        self.val_dim = val_dim

        self._keys: np.ndarray = np.zeros((capacity, key_dim), dtype=np.float32)
        self._values: np.ndarray = np.zeros((capacity, val_dim), dtype=np.float32)
        self._write_ptr: int = 0
        self._usage: int = 0

    # Write (sequential, circular)
    def write(self, key: np.ndarray, value: np.ndarray) -> None:
        self._keys[self._write_ptr] = key.astype(np.float32)
        self._values[self._write_ptr] = value.astype(np.float32)
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._usage = min(self._usage + 1, self.capacity)

    # Read (soft attention)
    def read(self, query: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        if self._usage == 0:
            return np.zeros(self.val_dim, dtype=np.float32)

        keys = self._keys[: self._usage]
        q = query.astype(np.float32)
        # Cosine similarity
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        k_norms = keys / (np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8)
        sims = k_norms @ q_norm
        weights = _softmax(sims / max(temperature, 1e-6))
        return (weights[:, None] * self._values[: self._usage]).sum(axis=0)

    def reset(self) -> None:
        self._keys[:] = 0.0
        self._values[:] = 0.0
        self._write_ptr = 0
        self._usage = 0

    def __repr__(self) -> str:
        return (
            f"EpisodicMemory(capacity={self.capacity}, "
            f"key_dim={self.key_dim}, val_dim={self.val_dim}, "
            f"usage={self._usage})"
        )


# ── Hebbian memory ────────────────────────────────────────────────────────────

class HebbianMemory:
    """Slow associative weight matrix updated by generalised Hebbian rule.

    ``W[i,j] += lr * pre[i] * post[j] - decay * W[i,j]``

    Persists across episodes (represents lifetime learning / structural change).
    """

    def __init__(self, n_pre: int, n_post: int, lr: float = 0.01, decay: float = 0.001) -> None:
        self.n_pre = n_pre
        self.n_post = n_post
        self.lr = lr
        self.decay = decay
        self.W: np.ndarray = np.zeros((n_pre, n_post), dtype=np.float32)

    def update(self, pre: np.ndarray, post: np.ndarray) -> None:
        self.W += self.lr * np.outer(pre, post) - self.decay * self.W
        # Clip to prevent runaway
        np.clip(self.W, -5.0, 5.0, out=self.W)

    def apply(self, pre: np.ndarray) -> np.ndarray:
        return self.W.T @ pre.astype(np.float32)

    def reset_episode(self) -> None:
        """Do NOT reset Hebbian weights — they persist across episodes."""

    def reset_lifetime(self) -> None:
        self.W[:] = 0.0

    def __repr__(self) -> str:
        return f"HebbianMemory(pre={self.n_pre}, post={self.n_post}, lr={self.lr})"


# ── Composite memory bank ────────────────────────────────────────────────────

class MemoryBank:
    """Aggregates all memory types for a single creature's brain."""

    def __init__(self, cfg: dict) -> None:
        mc = cfg["brain"]["memory"]
        self.working = WorkingMemory(mc.get("working_size", 8))
        self.episodic = EpisodicMemory(
            capacity=mc.get("episodic_capacity", 32),
            key_dim=mc.get("episodic_key_dim", 4),
            val_dim=mc.get("episodic_val_dim", 8),
        )
        self.hebbian_enabled = mc.get("hebbian_enabled", False)
        self.hebbian: HebbianMemory | None = None  # initialised by brain after size is known

    def init_hebbian(self, n_pre: int, n_post: int) -> None:
        if self.hebbian_enabled:
            lr = 0.01
            self.hebbian = HebbianMemory(n_pre, n_post, lr=lr)

    def reset_episode(self) -> None:
        self.working.reset()
        self.episodic.reset()
        # Hebbian persists

    def __repr__(self) -> str:
        return f"MemoryBank(working={self.working}, episodic={self.episodic})"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

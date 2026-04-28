"""Semantic memory — FAISS-backed cross-episode retrieval store.

State layout
------------
n : ()  int64  — total number of key-value pairs currently in the store.

The FAISS index and the parallel value tensor live on the *instance* (not in
the state dict) because semantic memory is intentionally cross-episode: its
content accumulates across ``reset_episode`` calls.  The state dict exists
solely for Protocol compliance and to let callers track the store size without
peeking at private attributes.

Read differentiability
----------------------
FAISS inner-product search is not differentiable.  After retrieval we
*recompute* scores in PyTorch (``F.normalize(query) · retrieved_keys``) so
that ``autograd`` can trace gradients from the output back through the
softmax weights to the query tensor.

References
----------
Johnson et al., *Billion-scale similarity search with GPUs* (TPAMI 2021).
https://arxiv.org/abs/1702.08734
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from evolux.core.types import MemoryState
from evolux.memory import MEMORY_REGISTRY

if TYPE_CHECKING:
    import faiss as _faiss_t

logger = logging.getLogger(__name__)


def _get_faiss() -> _faiss_t:
    """Lazily import faiss to avoid module-level segfaults in some environments."""
    import faiss

    return faiss


@MEMORY_REGISTRY.register("semantic")
class SemanticMemory:
    """FAISS-backed retrieval memory implementing the Memory Protocol.

    Maintains a growing ``faiss.IndexFlatIP`` of L2-normalised keys and a
    parallel CPU value tensor.  Reads perform inner-product search (equivalent
    to cosine similarity on normalised vectors) followed by a differentiable
    softmax-weighted sum.

    Parameters
    ----------
    capacity:
        Soft upper bound on the number of stored entries.  When exceeded, the
        oldest entries are evicted (FIFO) to keep the index at most ``capacity``
        entries.  Pass ``capacity=0`` (or any non-positive value) to disable
        eviction and allow unbounded growth.
    key_dim:
        Dimensionality of stored keys ``D``.
    val_dim:
        Dimensionality of stored values ``V``.

    Notes
    -----
    The FAISS index and value store are instance-level (not in the state dict)
    so they persist across ``reset_episode`` calls — matching the cross-episode
    semantics of semantic memory.
    """

    def __init__(self, capacity: int, key_dim: int, val_dim: int) -> None:
        self.capacity: int = capacity
        self.key_dim: int = key_dim
        self.val_dim: int = val_dim

        # Instance-level storage (grows with write calls).
        self._stored_keys: Tensor = torch.empty(0, key_dim, dtype=torch.float32)
        self._stored_values: Tensor = torch.empty(0, val_dim, dtype=torch.float32)
        self._index: object | None = None  # faiss.IndexFlatIP when populated

    # ── Private helpers ──────────────────────────────────────────────────────

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from ``_stored_keys``."""
        faiss = _get_faiss()
        n = self._stored_keys.shape[0]
        if n == 0:
            self._index = None
            return
        index = faiss.IndexFlatIP(self.key_dim)
        index.add(self._stored_keys.numpy())  # keys are already L2-normalised
        self._index = index

    def _evict_if_needed(self) -> None:
        """Drop the oldest entries when ``capacity > 0`` and the store is full."""
        if self.capacity <= 0:
            return
        n = self._stored_keys.shape[0]
        if n > self.capacity:
            self._stored_keys = self._stored_keys[-self.capacity :].contiguous()
            self._stored_values = self._stored_values[-self.capacity :].contiguous()

    # ── Protocol methods ─────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Return a minimal state dict for *batch_size* environments.

        Returns
        -------
        dict with:
            ``n`` : ``()`` int64 — number of entries currently in the store.
        """
        # A single scalar is sufficient: the store is shared across the batch.
        return {"n": torch.tensor(0, dtype=torch.long, device=device)}

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState:
        """Add ``(key, value)`` pairs to the FAISS index.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        key:
            ``(B, D)`` key tensor.  Keys are L2-normalised before storage so
            that ``IndexFlatIP`` computes cosine similarity.
        value:
            ``(B, V)`` value tensor.

        Returns
        -------
        New state with ``n`` incremented by ``B`` (the batch size).
        """
        B = key.shape[0]
        # Normalise keys → cosine similarity via inner product
        norm_key = F.normalize(key.detach().float().cpu(), dim=-1)  # (B, D)
        stored_val = value.detach().float().cpu()  # (B, V)

        self._stored_keys = torch.cat([self._stored_keys, norm_key], dim=0)
        self._stored_values = torch.cat([self._stored_values, stored_val], dim=0)
        self._evict_if_needed()
        self._rebuild_index()

        new_n = state["n"] + B
        return {"n": new_n}

    def read(self, state: MemoryState, query: Tensor, top_k: int = 4) -> Tensor:
        """Differentiable soft-attention read over the top-*k* retrieved values.

        Parameters
        ----------
        state:
            Current memory state.
        query:
            ``(B, D)`` query tensor.
        top_k:
            Number of nearest neighbours to retrieve from FAISS.  Clamped to
            the number of stored entries when the store is smaller than
            ``top_k``.

        Returns
        -------
        ``(B, V)`` softmax-weighted sum of the top-*k* retrieved values.

        Notes
        -----
        The FAISS search is non-differentiable; however, the inner-product
        scores used for the softmax are *recomputed in PyTorch* (using the
        L2-normalised query and the retrieved stored keys), so gradients flow
        from the output through the softmax weights back to ``query``.
        """
        B = query.shape[0]
        n = self._stored_keys.shape[0]

        if n == 0 or self._index is None:
            return torch.zeros(B, self.val_dim, device=query.device, dtype=query.dtype)

        k = min(top_k, n)

        # ── Step 1: FAISS search (non-diff) ─────────────────────────────────
        norm_q_cpu = F.normalize(query.detach().float().cpu(), dim=-1)  # (B, D)
        _scores_np, indices_np = self._index.search(norm_q_cpu.numpy(), k)  # (B, k)

        flat_idx = indices_np.flatten()  # (B*k,)
        # Retrieved keys and values — move to query device/dtype
        ret_keys = self._stored_keys[flat_idx].reshape(B, k, self.key_dim)
        ret_keys = ret_keys.to(device=query.device, dtype=query.dtype)
        ret_values = self._stored_values[flat_idx].reshape(B, k, self.val_dim)
        ret_values = ret_values.to(device=query.device, dtype=query.dtype)

        # ── Step 2: Differentiable score → softmax → weighted sum ────────────
        # Recompute scores in PyTorch so that autograd can trace through them.
        norm_q = F.normalize(query, dim=-1).unsqueeze(1)  # (B, 1, D)
        scores = (norm_q * ret_keys).sum(-1)  # (B, k) — inner product = cosine sim

        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # (B, k, 1)
        return (weights * ret_values).sum(dim=1)  # (B, V)

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState:
        """Return a new state dict; the cross-episode index is *not* cleared.

        Semantic memory accumulates knowledge across episodes by design.  This
        method exists to fulfil the ``Memory`` Protocol; it resets only the
        in-state counter for masked batch elements.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        mask:
            Unused for semantic memory (present for Protocol compatibility).

        Returns
        -------
        New state with ``n`` unchanged (the store itself is persistent).
        """
        return {"n": state["n"].clone()}

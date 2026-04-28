"""Working memory — sliding-window KV cache with circular-buffer semantics.

State layout
------------
keys   : (B, M, D)  — stored keys
values : (B, M, V)  — stored values
ptr    : (B,)  int  — next write position (circular)
"""

from __future__ import annotations

import torch
from torch import Tensor

from evolux.core.types import MemoryState
from evolux.memory import MEMORY_REGISTRY
from evolux.tensors import soft_attention


@MEMORY_REGISTRY.register("working_v1")
class WorkingMemory:
    """Sliding-window KV cache implementing the :class:`evolux.core.protocols.Memory` Protocol.

    Writes use circular-buffer semantics: when the buffer is full, the oldest
    slot is overwritten. Reads perform cosine-similarity soft attention over all
    stored slots.

    Parameters
    ----------
    capacity:
        Number of slots (M).
    key_dim:
        Dimensionality of keys (D).
    val_dim:
        Dimensionality of values (V).
    """

    def __init__(self, capacity: int, key_dim: int, val_dim: int) -> None:
        self.capacity: int = capacity
        self.key_dim: int = key_dim
        self.val_dim: int = val_dim

    # ── Protocol methods ────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Return a zero-initialised state dict for *batch_size* environments.

        Returns
        -------
        dict with:
            ``keys``   : ``(B, M, D)`` float32 zeros
            ``values`` : ``(B, M, V)`` float32 zeros
            ``ptr``    : ``(B,)`` int64 zeros
        """
        return {
            "keys": torch.zeros(batch_size, self.capacity, self.key_dim, device=device),
            "values": torch.zeros(batch_size, self.capacity, self.val_dim, device=device),
            "ptr": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState:
        """Pure write: insert ``(key, value)`` at the current pointer and advance it.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        key:
            ``(B, D)`` key tensor.
        value:
            ``(B, V)`` value tensor.

        Returns
        -------
        New state with the entry written and pointer advanced (circular).
        """
        keys = state["keys"].clone()  # (B, M, D)
        values = state["values"].clone()  # (B, M, V)
        ptr = state["ptr"]  # (B,)  — read-only, not cloned

        B = key.shape[0]
        batch_idx = torch.arange(B, device=key.device)  # (B,)
        keys[batch_idx, ptr] = key
        values[batch_idx, ptr] = value

        new_ptr = (ptr + 1) % self.capacity  # (B,) new tensor, not in-place

        return {"keys": keys, "values": values, "ptr": new_ptr}

    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor:
        """Cosine-similarity soft-attention read over all stored slots.

        Parameters
        ----------
        state:
            Current memory state.
        query:
            ``(B, D)`` query tensor.
        top_k:
            Only ``top_k=1`` is supported in this implementation (soft attention
            over all slots). Passing a value other than 1 raises
            ``NotImplementedError``; selective top-k retrieval is deferred to
            Phase-2 :class:`EpisodicMemory`.

        Returns
        -------
        ``(B, V)`` weighted sum of value slots.
        """
        if top_k != 1:
            raise NotImplementedError(
                "WorkingMemory.read: top_k > 1 is not supported. "
                "Use EpisodicMemory (Phase 2) for selective retrieval."
            )
        return soft_attention(query, state["keys"], state["values"])

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState:
        """Zero out the memory for environments indicated by *mask*.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        mask:
            ``(B,)`` bool tensor. If ``None``, all environments are reset.

        Returns
        -------
        New state with masked environments zeroed.
        """
        if mask is None:
            return self.init_state(state["keys"].shape[0], state["keys"].device)

        new_keys = state["keys"].clone()
        new_values = state["values"].clone()
        new_ptr = state["ptr"].clone()

        new_keys[mask] = 0.0
        new_values[mask] = 0.0
        new_ptr[mask] = 0

        return {"keys": new_keys, "values": new_values, "ptr": new_ptr}

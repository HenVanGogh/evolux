"""Episodic memory — DNC-style differentiable external memory.

State layout
------------
M       : (B, N, D_val)  — external memory matrix (N slots, D_val-dim values)
keys    : (B, N, D_key)  — key matrix used for content-based addressing
usage   : (B, N)         — cumulative usage per slot (float, in [0, 1])

Write mechanism (from DNC, Graves et al. 2016)
-----------------------------------------------
1. Content addressing: cosine similarity between write key and each stored key
   → write weight ``w`` (B, N), softmax.
2. Allocation: least-used slot used to modulate write weight when memory is full.
3. Soft erase + add:
       M_new = M * (1 - w[..., None] * e[..., None])   # erase
               + w[..., None] * a[..., None]            # add
   where ``e`` is a learned erase vector (ones here → full overwrite).
4. Usage update: usage += w * (1 - usage).

Read mechanism
--------------
Content-based: cosine similarity between read query and all stored keys
→ read weight, soft-attended value.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from evolux.core.types import MemoryState
from evolux.memory import MEMORY_REGISTRY


@MEMORY_REGISTRY.register("episodic")
class EpisodicMemory:
    """DNC-style differentiable external memory implementing the Memory Protocol.

    External memory matrix ``M: (B, N, D_val)`` with content-based
    read/write addressing, soft erase + add, usage tracking, and
    allocation by least-used slot.

    Parameters
    ----------
    capacity:
        Number of memory slots (``N``).
    key_dim:
        Dimensionality of keys used for content addressing (``D_key``).
    val_dim:
        Dimensionality of stored values (``D_val``).
    alloc_mix:
        Interpolation coefficient between content-based write weight and
        least-used-slot allocation weight (``0`` = pure content, ``1`` = pure
        allocation).  The DNC paper calls this the *allocation gate*.
        Default ``0.5``.

    Notes
    -----
    Reference: Graves et al., *Hybrid computing using a neural network with
    dynamic external memory* (DNC, Nature 2016).

    State keys (from :meth:`init_state`)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``M``     : ``(B, N, D_val)`` — value memory matrix
    ``keys``  : ``(B, N, D_key)`` — key memory matrix
    ``usage`` : ``(B, N)``        — per-slot usage in ``[0, 1]``
    """

    def __init__(
        self,
        capacity: int,
        key_dim: int,
        val_dim: int,
        alloc_mix: float = 0.5,
    ) -> None:
        self.capacity: int = capacity
        self.key_dim: int = key_dim
        self.val_dim: int = val_dim
        self.alloc_mix: float = float(alloc_mix)

    # ── Protocol methods ─────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> MemoryState:
        """Return zero-initialised state for *batch_size* independent environments.

        Returns
        -------
        dict with:
            ``M``     : ``(B, N, D_val)`` float32 zeros — value memory matrix
            ``keys``  : ``(B, N, D_key)`` float32 zeros — key memory matrix
            ``usage`` : ``(B, N)``        float32 zeros — per-slot usage
        """
        N = self.capacity
        return {
            "M": torch.zeros(batch_size, N, self.val_dim, device=device),
            "keys": torch.zeros(batch_size, N, self.key_dim, device=device),
            "usage": torch.zeros(batch_size, N, device=device),
        }

    def write(self, state: MemoryState, key: Tensor, value: Tensor) -> MemoryState:
        """DNC-style differentiable write.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        key:
            ``(B, D_key)`` write key for content addressing.
        value:
            ``(B, D_val)`` value to store.

        Returns
        -------
        New state with updated ``M``, ``keys``, and ``usage`` tensors.

        Algorithm
        ---------
        1. Content-based write weight via cosine similarity.
        2. Allocation weight (least-used slot).
        3. Blend content + allocation using ``alloc_mix``.
        4. Soft erase + add update to memory matrix.
        5. Soft update to key matrix.
        6. Usage update: ``u_new = u + w * (1 - u)`` (clipped to [0, 1]).
        """
        M = state["M"]  # (B, N, D_val)
        K = state["keys"]  # (B, N, D_key)
        usage = state["usage"]  # (B, N)

        # 1. Content-based write weight (cosine similarity).
        q = F.normalize(key, dim=-1).unsqueeze(1)  # (B, 1, D_key)
        k_norm = F.normalize(K, dim=-1)  # (B, N, D_key)
        content_score = (q * k_norm).sum(-1)  # (B, N)
        w_content = F.softmax(content_score, dim=-1)  # (B, N)

        # 2. Allocation weight: focus on the least-used slot.
        # Sort slots by usage ascending; assign allocation weight to the
        # least-used slot (soft approximation: inverse-usage softmax).
        alloc_score = -usage  # (B, N)  — lower usage → higher score
        w_alloc = F.softmax(alloc_score, dim=-1)  # (B, N)

        # 3. Blend content + allocation.
        w = (1.0 - self.alloc_mix) * w_content + self.alloc_mix * w_alloc  # (B, N)

        # 4. Soft erase + add to value memory (erase vector = ones → full rewrite).
        #    M_new[b, n] = M[b, n] * (1 - w[b, n]) + w[b, n] * value[b]
        w3 = w.unsqueeze(-1)  # (B, N, 1)
        M_new = M * (1.0 - w3) + w3 * value.unsqueeze(1)  # (B, N, D_val)

        # 5. Update key matrix with same addressing weight.
        K_new = K * (1.0 - w3) + w3 * key.unsqueeze(1)  # (B, N, D_key)

        # 6. Usage update: u_new = u + w * (1 - u), clamped to [0, 1].
        usage_new = (usage + w * (1.0 - usage)).clamp(0.0, 1.0)  # (B, N)

        return {"M": M_new, "keys": K_new, "usage": usage_new}

    def read(self, state: MemoryState, query: Tensor, top_k: int = 1) -> Tensor:
        """Content-addressed soft-attention read.

        Parameters
        ----------
        state:
            Current memory state.
        query:
            ``(B, D_key)`` read query.
        top_k:
            Number of top-scoring slots used in a hard-masked read.  When
            ``top_k == 1`` (default) a standard soft-attention read over all
            slots is performed.  When ``top_k > 1`` only the top-k slots
            (by cosine similarity) contribute to the weighted sum.

        Returns
        -------
        ``(B, D_val)`` retrieved value tensor.
        """
        K = state["keys"]  # (B, N, D_key)
        M = state["M"]  # (B, N, D_val)

        q = F.normalize(query, dim=-1).unsqueeze(1)  # (B, 1, D_key)
        k_norm = F.normalize(K, dim=-1)  # (B, N, D_key)
        sims = (q * k_norm).sum(-1)  # (B, N)

        if top_k == 1:
            weights = F.softmax(sims, dim=-1)  # (B, N)
        else:
            # Zero out all but top-k scores before softmax.
            top_k_values, _ = sims.topk(min(top_k, sims.size(-1)), dim=-1, sorted=False)  # (B, k)
            threshold = top_k_values.min(dim=-1, keepdim=True).values  # (B, 1)
            mask = sims >= threshold  # (B, N) bool
            masked_sims = sims.masked_fill(~mask, float("-inf"))
            weights = F.softmax(masked_sims, dim=-1)  # (B, N)

        return (weights.unsqueeze(-1) * M).sum(dim=1)  # (B, D_val)

    def reset_episode(self, state: MemoryState, mask: Tensor | None = None) -> MemoryState:
        """Zero out the memory for environments indicated by *mask*.

        Parameters
        ----------
        state:
            Current memory state (not mutated).
        mask:
            ``(B,)`` bool tensor.  If ``None``, all environments are reset.

        Returns
        -------
        New state with masked environments zeroed.
        """
        if mask is None:
            return self.init_state(state["M"].shape[0], state["M"].device)

        new_M = state["M"].clone()
        new_keys = state["keys"].clone()
        new_usage = state["usage"].clone()

        new_M[mask] = 0.0
        new_keys[mask] = 0.0
        new_usage[mask] = 0.0

        return {"M": new_M, "keys": new_keys, "usage": new_usage}

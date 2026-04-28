"""TransformerBrain — Phase-1 causal Transformer policy.

A small, dependency-free implementation of a Pre-LayerNorm Transformer policy
that satisfies :class:`evolux.core.protocols.Brain`.

Architectural notes
-------------------
- **Pre-LN** residual blocks (LN inside the residual branch, not after).
- **RMSNorm** instead of LayerNorm (LLaMA-style, no mean-centering, no bias).
- **RoPE** rotary position embeddings applied to Q/K inside the attention.
- **KV-cache recurrence**: ``forward`` is single-step; the cached keys / values
  live inside the returned :class:`BrainState`, so successive calls attend over
  the growing history (capped at ``max_seq_len``).

The brain is intentionally architecture-only — it does not pull in optional
``perception`` / ``memory`` modules; instead it consumes a generic
``ObsSpec`` whose tensor fields are flattened and projected into the model
hidden width.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from evolux.brain import BRAIN_REGISTRY
from evolux.core.types import (
    Action,
    ActionSpec,
    AuxInfo,
    BrainState,
    Obs,
    ObsSpec,
    StateSpec,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _flat_dim(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (LLaMA-style)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _rope_freqs(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10_000.0
) -> tuple[Tensor, Tensor]:
    """Return ``(cos, sin)`` of shape ``(seq_len, head_dim // 2)``."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # (seq_len, half)
    return freqs.cos(), freqs.sin()


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the last dim of ``x`` by RoPE.

    ``x``    : ``(..., T, head_dim)``
    ``cos`` / ``sin`` : ``(T, head_dim // 2)``
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


# ── modules ────────────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    """Single-step multi-head self-attention with an external KV cache.

    The new query attends over the cached keys/values *plus* itself, which is
    causally correct because nothing in the future has been generated yet.
    """

    def __init__(self, dim: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"hidden dim {dim} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Precompute RoPE cos/sin tables up to max_seq_len; non-persistent so
        # they don't bloat checkpoints and follow the module's device on .to().
        cos, sin = _rope_freqs(max_seq_len, self.head_dim, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self, x: Tensor, k_cache: Tensor, v_cache: Tensor, max_seq_len: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        # x: (B, 1, H);  k_cache / v_cache: (B, T_cur, n_heads, head_dim)
        b, t_new, _ = x.shape
        qkv = self.qkv(x).reshape(b, t_new, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, 1, n_heads, head_dim)

        t_cur = k_cache.shape[1]
        # RoPE positions: cached tokens live at [0, t_cur), the new one at t_cur.
        # Position of the freshly written token is min(t_cur, max_seq_len - 1)
        # because the cache is shift-truncated; this keeps RoPE indices in range.
        pos = min(t_cur, max_seq_len - 1)
        cos = self.rope_cos[pos : pos + t_new]
        sin = self.rope_sin[pos : pos + t_new]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        k_full = torch.cat([k_cache, k], dim=1)  # (B, T_cur + 1, n_heads, head_dim)
        v_full = torch.cat([v_cache, v], dim=1)

        # Truncate cache to max_seq_len (drop oldest).
        if k_full.shape[1] > max_seq_len:
            k_full = k_full[:, -max_seq_len:]
            v_full = v_full[:, -max_seq_len:]

        # (B, n_heads, T_q, head_dim) and (B, n_heads, T_k, head_dim).
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)

        # No attention mask needed: the query is the most-recent token and
        # already attends only to past + self.
        attn = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        out = attn.transpose(1, 2).reshape(b, t_new, self.n_heads * self.head_dim)
        return self.out_proj(out), k_full, v_full


class TransformerBlock(nn.Module):
    """Pre-LN block: ``x = x + Attn(RMSNorm(x)); x = x + MLP(RMSNorm(x))``."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int, max_seq_len: int) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_seq_len)
        self.norm_mlp = RMSNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(
        self, x: Tensor, k_cache: Tensor, v_cache: Tensor, max_seq_len: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        h, k_new, v_new = self.attn(self.norm_attn(x), k_cache, v_cache, max_seq_len)
        x = x + h
        x = x + self.mlp(self.norm_mlp(x))
        return x, k_new, v_new


# ── Brain ──────────────────────────────────────────────────────────────────


@BRAIN_REGISTRY.register("transformer")
class TransformerBrain(nn.Module):
    """Causal Transformer policy implementing :class:`Brain`.

    Parameters
    ----------
    obs_spec, action_spec:
        Standard evolux specs. All tensor fields in ``obs_spec`` are flattened
        per-sample and concatenated into a single token of width ``hidden_dim``.
    hidden_dim:
        Model width ``H``. Must be divisible by ``n_heads``.
    n_layers:
        Number of stacked Pre-LN Transformer blocks ``L``.
    n_heads:
        Number of attention heads.
    mlp_ratio:
        MLP hidden width is ``mlp_ratio * hidden_dim``.
    max_seq_len:
        Cap on the KV-cache length per env.
    """

    obs_spec: ObsSpec
    action_spec: ActionSpec
    state_spec: StateSpec

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        *,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        mlp_ratio: int = 4,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        if not obs_spec.fields:
            raise ValueError("obs_spec must declare at least one observation field")
        if action_spec.n <= 0:
            raise ValueError("action_spec.n must be positive")

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.max_seq_len = max_seq_len

        # Field-wise input projection: flatten each obs field then concat.
        self._field_dims: dict[str, int] = {
            name: _flat_dim(shape) for name, (shape, _) in obs_spec.fields.items()
        }
        in_dim = sum(self._field_dims.values())
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, n_heads, mlp_ratio, max_seq_len) for _ in range(n_layers)]
        )
        self.final_norm = RMSNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_spec.n)
        self.value_head = nn.Linear(hidden_dim, 1)

        # State spec is descriptive only (per-sample shapes, leading B implicit).
        kv_shape = (n_layers, max_seq_len, n_heads, self.head_dim)
        self.state_spec = StateSpec(
            fields={
                "k": (kv_shape, torch.float32),
                "v": (kv_shape, torch.float32),
                "length": ((), torch.long),
            }
        )

    # ── Brain Protocol ────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        empty_kv = torch.zeros(
            self.n_layers, batch_size, 0, self.n_heads, self.head_dim, device=device
        )
        return {
            "k": empty_kv,
            "v": empty_kv.clone(),
            "length": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]:
        x = self._encode_obs(obs)  # (B, 1, H)
        b = x.shape[0]

        k_stack = state["k"]  # (L, B, T, n_heads, head_dim)
        v_stack = state["v"]
        new_k: list[Tensor] = []
        new_v: list[Tensor] = []
        for i, block in enumerate(self.blocks):
            x, k_new, v_new = block(x, k_stack[i], v_stack[i], self.max_seq_len)
            new_k.append(k_new)
            new_v.append(v_new)

        x = self.final_norm(x).squeeze(1)  # (B, H)
        logits = self.action_head(x)  # (B, action_spec.n)
        value = self.value_head(x).squeeze(-1)  # (B,)

        action = self._sample_action(logits)

        new_state: BrainState = {
            "k": torch.stack(new_k, dim=0),
            "v": torch.stack(new_v, dim=0),
            "length": torch.full(
                (b,),
                fill_value=int(new_k[0].shape[1]),
                dtype=torch.long,
                device=x.device,
            ),
        }
        aux: AuxInfo = {"logits": logits, "value": value}
        return action, new_state, aux

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    # ── helpers ───────────────────────────────────────────────────────────

    def _encode_obs(self, obs: Obs) -> Tensor:
        """Flatten each declared field per-sample, concat, project to hidden."""
        parts: list[Tensor] = []
        b: int | None = None
        for name in self.obs_spec.fields:
            if name not in obs:
                raise KeyError(f"missing obs field '{name}'")
            t = obs[name]
            if b is None:
                b = t.shape[0]
            elif t.shape[0] != b:
                raise ValueError(f"obs field '{name}' batch dim {t.shape[0]} != {b}")
            parts.append(t.reshape(b, -1).to(dtype=torch.float32))
        if b is None:  # pragma: no cover — guarded by __init__ check
            raise ValueError("empty obs")
        cat = torch.cat(parts, dim=-1)  # (B, in_dim)
        return self.input_proj(cat).unsqueeze(1)  # (B, 1, H)

    def _sample_action(self, logits: Tensor) -> Action:
        """Greedy / bounded action — deterministic so callers control sampling."""
        if self.action_spec.discrete:
            return logits.argmax(dim=-1)
        if self.action_spec.bounds is not None:
            lo, hi = self.action_spec.bounds
            return torch.tanh(logits) * (0.5 * (hi - lo)) + 0.5 * (hi + lo)
        return logits


__all__ = ["CausalSelfAttention", "RMSNorm", "TransformerBlock", "TransformerBrain"]

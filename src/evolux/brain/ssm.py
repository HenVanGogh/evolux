"""SSMBrain — Phase-2 Mamba-style selective SSM policy.

A pure-PyTorch implementation of the Mamba selective state-space model (SSM)
that satisfies :class:`evolux.core.protocols.Brain`.

Architectural notes
-------------------
- **Selective SSM**: input-dependent A, B, C, Δ matrices (S4 → Mamba style).
- **Parallel scan**: uses an associative parallel scan for efficient training
  over sequences (pure PyTorch, no Triton/custom CUDA kernels).
- **Recurrent inference**: at inference time (T=1), the SSM reduces to a simple
  RNN-like update: h_t = A_bar * h_{t-1} + B_bar * u_t. The hidden state h: (B, D, N)
  is stored in :class:`BrainState` for episode-level recurrence.
- **Single-step forward** is compatible with the Brain Protocol: each call
  processes one observation token and returns the updated SSM state.

References
----------
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
  (2023). https://arxiv.org/abs/2312.00752
- Reference implementation (pure-PyTorch scan):
  https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
"""

from __future__ import annotations

import math
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


def _ssm_scan_step(
    h: Tensor,
    u: Tensor,
    A_bar: Tensor,
    B_bar: Tensor,
) -> Tensor:
    """Single recurrent SSM step (inference).

    Parameters
    ----------
    h : (B, D, N)  — current hidden state
    u : (B, D)     — input at this step
    A_bar : (B, D, N)  — discretised state-transition matrix (diagonal)
    B_bar : (B, D, N)  — discretised input matrix

    Returns
    -------
    h_new : (B, D, N)
    """
    # h_t = A_bar * h_{t-1} + B_bar * u_t
    # B_bar * u:  B_bar is (B, D, N), u is (B, D) → unsqueeze to (B, D, 1)
    return A_bar * h + B_bar * u.unsqueeze(-1)  # (B, D, N)


class SelectiveSSMBlock(nn.Module):
    """A single Mamba-style selective SSM block.

    For each block:
      1. Input projection + gating (two parallel projections).
      2. Depthwise conv (causal, width=conv_kernel_size).
      3. Selective SSM: compute input-dependent (Δ, B, C, A) → recurrent update.
      4. Output projection.

    All shapes below use::

        B = batch size
        D = hidden_dim (model width, also called ``d_model``)
        E = inner_dim = D * expand   (``d_inner`` in Mamba paper, default 2xD)
        N = state_dim                (SSM state size, ``d_state`` in Mamba)
        L = sequence length (1 at inference time)

    Notes
    -----
    The parallel-scan path (L > 1) follows the standard ``pscan`` recurrence::

        For l in 0..L:
            h[l] = A_bar[l] * h[l-1] + B_bar[l] * u[l]
            y[l] = (C[l] * h[l]).sum(-1)

    implemented via a sequential loop (pure PyTorch).  For L=1 (inference),
    this collapses to a single multiplicative update, enabling efficient
    rollout.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 16,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_kernel_size: int = 4,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim  # N
        inner_dim = hidden_dim * expand  # E
        self.inner_dim = inner_dim
        self.expand = expand

        if dt_rank is None:
            dt_rank = max(1, hidden_dim // 16)
        self.dt_rank = dt_rank

        # ── projections ──────────────────────────────────────────────────
        # in_proj: D → 2E  (one for x, one for z/gate)
        self.in_proj = nn.Linear(hidden_dim, 2 * inner_dim, bias=False)

        # Causal depthwise conv over the inner channel
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_kernel_size,
            groups=inner_dim,
            padding=conv_kernel_size - 1,
            bias=True,
        )

        # Selective SSM projections: x → (Δ, B, C)
        # Δ (dt): rank-reduced, then promoted to inner_dim
        self.x_proj = nn.Linear(inner_dim, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner_dim, bias=True)

        # Log-space diagonal A: shape (E, N), initialised from 1..N (HiPPO-like)
        A_vals = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(inner_dim, 1)
        self.A_log = nn.Parameter(torch.log(A_vals))  # (E, N)

        # D skip connection (same as Mamba paper)
        self.D = nn.Parameter(torch.ones(inner_dim))

        self.out_proj = nn.Linear(inner_dim, hidden_dim, bias=False)

        # LayerNorm for the residual
        self.norm = nn.LayerNorm(hidden_dim)

        # initialise dt_proj bias so Δ ≈ softplus^{-1}(uniform(dt_min, dt_max))
        nn.init.uniform_(self.dt_proj.bias, dt_min, dt_max)

    # ── Forward: sequence path (training / multi-step) ────────────────────

    def _ssm_seq(self, u: Tensor, h_init: Tensor) -> tuple[Tensor, Tensor]:
        """Run SSM on a sequence u: (B, E, L).

        Parameters
        ----------
        u : (B, E, L)  — pre-processed (after conv + SiLU) inner activations
        h_init : (B, E, N)  — initial hidden state (usually zeros for training)

        Returns
        -------
        y : (B, L, E)
        h_last : (B, E, N)
        """
        _b, _e, length = u.shape
        n = self.state_dim

        # u → (B, L, E) for projection
        u_t = u.transpose(1, 2)  # (B, L, E)

        # Compute input-dependent (Δ, B, C)
        x_proj_out = self.x_proj(u_t)  # (B, L, dt_rank + 2N)
        dt_raw = x_proj_out[..., : self.dt_rank]  # (B, L, dt_rank)
        B_raw = x_proj_out[..., self.dt_rank : self.dt_rank + n]  # (B, L, N)
        C = x_proj_out[..., self.dt_rank + n :]  # (B, L, N)

        # Δ: (B, L, E)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, L, E)

        # A: (E, N) → negative for stability
        A = -torch.exp(self.A_log)  # (E, N)

        # Discretise: zero-order hold
        # A_bar[t] = exp(Δ[t] * A)  shape (B, L, E, N)
        # B_bar[t] = Δ[t] * B[t]   shape (B, L, E, N)
        # dt: (B, L, E) → (B, L, E, 1);  A: (E, N) → (1, 1, E, N)
        dt_e = dt.unsqueeze(-1)  # (B, L, E, 1)
        A_bar = torch.exp(dt_e * A.unsqueeze(0).unsqueeze(0))  # (B, L, E, N)
        B_bar = dt_e * B_raw.unsqueeze(2)  # (B, L, E, N)

        # Sequential scan: h_{t} = A_bar_{t} * h_{t-1} + B_bar_{t} * u_{t}
        h = h_init  # (B, E, N)
        ys: list[Tensor] = []
        for t in range(length):
            # u_t: (B, E); A_bar_t: (B, E, N); B_bar_t: (B, E, N); C_t: (B, N)
            h = A_bar[:, t] * h + B_bar[:, t] * u_t[:, t].unsqueeze(-1)  # (B, E, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, E)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, E)
        return y, h

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        """Selective SSM block forward pass.

        Parameters
        ----------
        x : (B, L, D)  — input token sequence (L=1 for single-step inference)
        h : (B, E, N)  — SSM hidden state (carried across calls)

        Returns
        -------
        out : (B, L, D)
        h_new : (B, E, N)
        """
        residual = x
        x = self.norm(x)

        _b, length, _ = x.shape

        # Gated input projection → x_in, z
        xz = self.in_proj(x)  # (B, L, 2E)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, L, E)

        # Causal conv: apply along sequence length
        # conv1d expects (B, C, L); padding is left-only (causal)
        x_conv = x_in.transpose(1, 2)  # (B, E, L)
        x_conv = self.conv1d(x_conv)[..., :length]  # (B, E, L) — trim right pad
        x_conv = F.silu(x_conv)  # (B, E, L)

        # SSM
        y_ssm, h_new = self._ssm_seq(x_conv, h)  # (B, L, E), (B, E, N)

        # D skip: y = y_ssm + u * D
        y_ssm = y_ssm + x_in * self.D.unsqueeze(0).unsqueeze(0)

        # Gate output
        out_inner = y_ssm * F.silu(z)  # (B, L, E)

        # Output projection + residual
        out = self.out_proj(out_inner) + residual  # (B, L, D)

        return out, h_new


# ── Brain ──────────────────────────────────────────────────────────────────


@BRAIN_REGISTRY.register("ssm")
class SSMBrain(nn.Module):
    """Selective SSM (Mamba-style) policy implementing :class:`Brain`.

    Parameters
    ----------
    obs_spec, action_spec:
        Standard evolux specs. All tensor fields in ``obs_spec`` are flattened
        per-sample and concatenated into a single token of width ``hidden_dim``.
    hidden_dim:
        Model width ``D``. Feature dimension fed into SSM blocks.
    n_layers:
        Number of stacked SSM blocks.
    state_dim:
        SSM state (recurrent) dimension ``N``.
    expand:
        Inner dim expansion factor (``E = D * expand``).
    dt_rank:
        Rank for the Δ low-rank projection. Defaults to ``max(1, D // 16)``.
    conv_kernel_size:
        Causal depthwise conv kernel width (default 4, same as Mamba paper).
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
        state_dim: int = 16,
        expand: int = 2,
        dt_rank: int | None = None,
        conv_kernel_size: int = 4,
    ) -> None:
        super().__init__()

        if not obs_spec.fields:
            raise ValueError("obs_spec must declare at least one observation field")
        if action_spec.n <= 0:
            raise ValueError("action_spec.n must be positive")

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.state_dim = state_dim
        self.expand = expand
        self.inner_dim = hidden_dim * expand  # E

        # Input projection: flatten obs → hidden_dim
        self._field_dims: dict[str, int] = {
            name: _flat_dim(shape) for name, (shape, _) in obs_spec.fields.items()
        }
        in_dim = sum(self._field_dims.values())
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # Stacked SSM blocks
        self.blocks = nn.ModuleList(
            [
                SelectiveSSMBlock(
                    hidden_dim=hidden_dim,
                    state_dim=state_dim,
                    expand=expand,
                    dt_rank=dt_rank,
                    conv_kernel_size=conv_kernel_size,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, action_spec.n)
        self.value_head = nn.Linear(hidden_dim, 1)

        # State spec: L blocks x (B, E, N) hidden state
        h_shape = (n_layers, self.inner_dim, state_dim)
        self.state_spec = StateSpec(
            fields={
                "h": (h_shape, torch.float32),
            }
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming-uniform init for linear layers (PyTorch default is OK,
        but be explicit so it's deterministic across versions)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)

    # ── Brain Protocol ────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        """Return zero SSM hidden states: ``h`` shape (L, B, E, N)."""
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.inner_dim,
            self.state_dim,
            device=device,
        )
        return {"h": h}

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]:
        """Single-step forward.

        Parameters
        ----------
        obs:
            Dict of tensors with leading batch dim ``B``.
        state:
            BrainState from ``init_state`` or a previous ``forward`` call.
            ``state["h"]`` has shape ``(L, B, E, N)``.

        Returns
        -------
        action, new_state, aux
        """
        x = self._encode_obs(obs)  # (B, 1, D)

        h_stack = state["h"]  # (L, B, E, N)
        new_h_list: list[Tensor] = []

        for i, block in enumerate(self.blocks):
            x, h_new = block(x, h_stack[i])  # x: (B, 1, D), h_new: (B, E, N)
            new_h_list.append(h_new)

        x = self.final_norm(x).squeeze(1)  # (B, D)
        logits = self.action_head(x)  # (B, action_spec.n)
        value = self.value_head(x).squeeze(-1)  # (B,)

        action = self._sample_action(logits)

        new_state: BrainState = {
            "h": torch.stack(new_h_list, dim=0),  # (L, B, E, N)
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
        return self.input_proj(cat).unsqueeze(1)  # (B, 1, D)

    def _sample_action(self, logits: Tensor) -> Action:
        """Greedy / bounded action — deterministic so callers control sampling."""
        if self.action_spec.discrete:
            return logits.argmax(dim=-1)
        if self.action_spec.bounds is not None:
            lo, hi = self.action_spec.bounds
            return torch.tanh(logits) * (0.5 * (hi - lo)) + 0.5 * (hi + lo)
        return logits


__all__ = ["SSMBrain", "SelectiveSSMBlock"]

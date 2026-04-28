"""NeuromodWrapper — Phase-2 neuromodulator gating wrapper.

Wraps another :class:`~evolux.core.protocols.Brain` and adds a small modulator
MLP that, given the current observation, produces a ``(B, n_modulators)``
gating tensor.  The gates are multiplicatively applied to the inner brain's
output: for discrete action spaces the logits (if exposed in ``aux``) are
scaled, preserving the argmax; for continuous action spaces the action tensor
itself is scaled.  The mean of the ``n_modulators`` gates forms the effective
scalar gain per sample, so forcing all gates to ``1.0`` leaves the inner
brain's behaviour unchanged.

Notes
-----
Soltoggio et al., *Born to Learn: The Forgotten Feature of Neuromodulation*
(2018) — motivation for neuromodulated plasticity as a meta-learning signal.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from evolux.brain import BRAIN_REGISTRY
from evolux.core.protocols import Brain
from evolux.core.types import (
    Action,
    ActionSpec,
    AuxInfo,
    BrainState,
    Obs,
    ObsSpec,
    StateSpec,
)


def _flat_dim(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


@BRAIN_REGISTRY.register("neuromod")
class NeuromodWrapper(nn.Module):
    """Neuromodulator gating wrapper implementing :class:`~evolux.core.protocols.Brain`.

    Parameters
    ----------
    inner_brain:
        Any object satisfying the :class:`~evolux.core.protocols.Brain`
        Protocol.  Must also be a :class:`torch.nn.Module` so that its
        parameters are reachable via ``self.parameters()``.
    obs_spec:
        Observation spec (shared with the inner brain).
    action_spec:
        Action spec (shared with the inner brain).
    n_modulators:
        Number of independent gating signals emitted by the modulator MLP.
        The effective per-sample gate is the mean of the ``n_modulators``
        values, so all-ones gives a gain of exactly ``1.0``.
    hidden_dim:
        Width of the single hidden layer in the modulator MLP.
    """

    obs_spec: ObsSpec
    action_spec: ActionSpec
    state_spec: StateSpec

    def __init__(
        self,
        inner_brain: Brain,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        *,
        n_modulators: int = 8,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not isinstance(inner_brain, nn.Module):
            raise TypeError(
                f"inner_brain must be a torch.nn.Module, got {type(inner_brain).__name__}"
            )
        if n_modulators < 1:
            raise ValueError(f"n_modulators must be >= 1, got {n_modulators}")

        self.inner_brain = inner_brain
        self.obs_spec = obs_spec
        self.action_spec = action_spec
        # The wrapper adds no extra state — it delegates entirely to the inner brain.
        self.state_spec: StateSpec = inner_brain.state_spec
        self.n_modulators = n_modulators

        # Flattened observation dimension for the modulator's input.
        obs_dim = sum(_flat_dim(shape) for shape, _ in obs_spec.fields.values())

        # Modulator MLP: obs → hidden → n_modulators with sigmoid output ∈ (0, 1).
        self.modulator_mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_modulators),
            nn.Sigmoid(),
        )

    # ── Brain Protocol ──────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        """Delegate state initialisation to the inner brain."""
        return self.inner_brain.init_state(batch_size, device)

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]:
        """Run the inner brain then apply neuromodulator gating.

        Returns
        -------
        action:
            Same shape as the inner brain's action output.
        new_state:
            Recurrent state returned by the inner brain (unchanged by wrapper).
        aux:
            All entries from the inner brain's ``aux`` dict, plus
            ``"modulators"`` of shape ``(B, n_modulators)``.
        """
        # ── inner brain forward pass ──────────────────────────────────────
        inner_action, new_state, inner_aux = self.inner_brain.forward(obs, state)

        # ── build flat observation tensor ─────────────────────────────────
        parts: list[Tensor] = []
        b: int | None = None
        for name in self.obs_spec.fields:
            t = obs[name]
            if b is None:
                b = t.shape[0]
            parts.append(t.reshape(b, -1).to(dtype=torch.float32))
        flat_obs = torch.cat(parts, dim=-1)  # (B, obs_dim)

        # ── modulator gates ───────────────────────────────────────────────
        gates: Tensor = self.modulator_mlp(flat_obs)  # (B, n_modulators)

        # Effective scalar gain per sample: mean over n_modulators axes.
        # When all gates == 1.0, eff_gate == 1.0 → identity scaling.
        eff_gate = gates.mean(dim=-1, keepdim=True)  # (B, 1)

        # ── apply gates multiplicatively ──────────────────────────────────
        action: Action
        aux_out: AuxInfo = dict(inner_aux)  # shallow copy so we can update safely

        if self.action_spec.discrete:
            # argmax is invariant to positive scaling, so the action index is
            # preserved while the gated logits carry gradient through the wrapper.
            if "logits" in inner_aux:
                gated_logits: Tensor = inner_aux["logits"] * eff_gate  # (B, n_classes)
                action = gated_logits.argmax(dim=-1)
                aux_out["logits"] = gated_logits
            else:
                action = inner_action
        else:
            # Continuous: gate the action tensor directly.
            action = inner_action * eff_gate  # (B, A) or (B,)

        aux_out["modulators"] = gates  # (B, n_modulators)
        return action, new_state, aux_out

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """All trainable parameters (wrapper + inner brain)."""
        return (p for p in self.parameters() if p.requires_grad)


__all__ = ["NeuromodWrapper"]

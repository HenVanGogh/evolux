"""WorldModelBrain — Phase-2 Dreamer-V3-lite RSSM policy.

Implements a lightweight Recurrent State-Space Model (RSSM) inspired by
Dreamer-V3 (Hafner et al., 2023). The brain satisfies both
:class:`evolux.core.protocols.Brain` and :class:`evolux.core.protocols.Imaginer`.

Architecture
------------
State representation
    h_t   : deterministic GRU recurrent hidden state  (B, det_dim)
    z_t   : stochastic categorical latent              (B, n_cats * cat_dim)

RSSM step (prior / posterior)
    1. obs → encoder → embed_t
    2. posterior:  (h_t, embed_t) → Linear → z_t  (used during forward)
    3. prior:      (h_t)          → Linear → z_t  (used during imagine)
    4. GRU: h_{t+1} = GRU(cat(z_t, a_t), h_t)

Policy head
    (h_t, z_t) → action logits / continuous action

Reward head
    (h_t, z_t) → scalar reward prediction

Decoder head
    (h_t, z_t) → reconstructed obs (same shape as obs input)

Notes
-----
Hafner et al., *Mastering Diverse Domains through World Models* (Dreamer-V3, 2023).
Categorical latents: one-hot straight-through estimator for gradient flow.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from evolux.brain import BRAIN_REGISTRY
from evolux.core.types import (
    Action,
    ActionSpec,
    AuxInfo,
    BrainState,
    Obs,
    ObsSpec,
    StateSpec,
    Trajectory,
)

# ── helpers ────────────────────────────────────────────────────────────────


def _flat_dim(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _encode_obs_flat(obs: Obs, obs_spec: ObsSpec) -> Tensor:
    """Flatten and concatenate all declared obs fields → (B, in_dim)."""
    parts: list[Tensor] = []
    b: int | None = None
    for name in obs_spec.fields:
        if name not in obs:
            raise KeyError(f"missing obs field '{name}'")
        t = obs[name]
        if b is None:
            b = t.shape[0]
        parts.append(t.reshape(b, -1).to(dtype=torch.float32))
    if b is None:
        raise ValueError("empty obs")
    return torch.cat(parts, dim=-1)  # (B, in_dim)


def _straight_through_one_hot(logits: Tensor, n_cats: int, cat_dim: int) -> Tensor:
    """Categorical straight-through estimator.

    Parameters
    ----------
    logits:
        Shape ``(B, n_cats * cat_dim)`` — raw logits, reshaped internally.
    n_cats, cat_dim:
        Number of categorical variables and their size.

    Returns
    -------
    Tensor of shape ``(B, n_cats * cat_dim)`` — one-hot with gradient via
    straight-through.
    """
    b = logits.shape[0]
    logits_2d = logits.reshape(b, n_cats, cat_dim)  # (B, n_cats, cat_dim)
    # Hard one-hot (no grad)
    indices = logits_2d.argmax(dim=-1, keepdim=True)  # (B, n_cats, 1)
    hard = torch.zeros_like(logits_2d).scatter_(-1, indices, 1.0)
    # Soft via softmax (has grad)
    soft = F.softmax(logits_2d, dim=-1)
    # Straight-through: forward = hard, backward through soft
    out = (hard - soft).detach() + soft
    return out.reshape(b, n_cats * cat_dim)


# ── WorldModelBrain ────────────────────────────────────────────────────────


@BRAIN_REGISTRY.register("world_model")
class WorldModelBrain(nn.Module):
    """Dreamer-V3-lite RSSM policy implementing :class:`Brain` and :class:`Imaginer`.

    Parameters
    ----------
    obs_spec, action_spec:
        Standard evolux specs. All tensor fields in ``obs_spec`` are flattened
        and concatenated before encoding.
    det_dim:
        Dimension of the deterministic GRU hidden state ``h``.
    n_cats:
        Number of independent categorical variables in the stochastic latent.
    cat_dim:
        Size of each categorical variable (num classes).
    hidden_dim:
        Width of MLP hidden layers in the policy, reward, and decoder heads.
    """

    obs_spec: ObsSpec
    action_spec: ActionSpec
    state_spec: StateSpec

    def __init__(
        self,
        obs_spec: ObsSpec,
        action_spec: ActionSpec,
        *,
        det_dim: int = 128,
        n_cats: int = 8,
        cat_dim: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        if not obs_spec.fields:
            raise ValueError("obs_spec must declare at least one observation field")
        if action_spec.n <= 0:
            raise ValueError("action_spec.n must be positive")

        self.obs_spec = obs_spec
        self.action_spec = action_spec
        self.det_dim = det_dim
        self.n_cats = n_cats
        self.cat_dim = cat_dim
        self.stoch_dim = n_cats * cat_dim  # z dim

        # ── obs encoder ───────────────────────────────────────────────────
        obs_in_dim = sum(_flat_dim(shape) for shape, _ in obs_spec.fields.values())
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        # ── action dim for GRU input ──────────────────────────────────────
        # discrete: one-hot of size n; continuous: raw vector of size n
        act_in_dim = action_spec.n

        # ── GRU (deterministic part of RSSM) ─────────────────────────────
        # input = [stoch_dim + act_in_dim]
        self.gru_cell = nn.GRUCell(self.stoch_dim + act_in_dim, det_dim)

        # ── posterior: (h, obs_embed) → z logits ─────────────────────────
        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.stoch_dim),
        )

        # ── prior: (h) → z logits ─────────────────────────────────────────
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self.stoch_dim),
        )

        # ── policy head: (h, z) → action ─────────────────────────────────
        feat_dim = det_dim + self.stoch_dim
        self.policy_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_spec.n),
        )

        # ── reward head: (h, z) → reward ─────────────────────────────────
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ── decoder head: (h, z) → obs reconstruction ────────────────────
        self.decoder_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, obs_in_dim),
        )

        # ── value head ────────────────────────────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        # State spec (per-sample shapes, leading B implicit)
        self.state_spec = StateSpec(
            fields={
                "h": ((det_dim,), torch.float32),
                "z": ((self.stoch_dim,), torch.float32),
                "prev_action": ((action_spec.n,), torch.float32),
            }
        )

    # ── Brain Protocol ────────────────────────────────────────────────────

    def init_state(self, batch_size: int, device: torch.device) -> BrainState:
        return {
            "h": torch.zeros(batch_size, self.det_dim, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
            "prev_action": torch.zeros(batch_size, self.action_spec.n, device=device),
        }

    def forward(self, obs: Obs, state: BrainState) -> tuple[Action, BrainState, AuxInfo]:
        """Single-step RSSM posterior + policy.

        Parameters
        ----------
        obs:
            Dict of observation tensors with leading batch dim B.
        state:
            Previous recurrent state from :meth:`init_state` or a prior call.

        Returns
        -------
        action, new_state, aux
        """
        h = state["h"]  # (B, det_dim)
        prev_action = state["prev_action"]  # (B, action_n)

        # 1. Encode obs
        obs_flat = _encode_obs_flat(obs, self.obs_spec)  # (B, obs_in)
        obs_embed = self.obs_encoder(obs_flat)  # (B, hidden_dim)

        # 2. Posterior: infer z from (h, obs_embed)
        post_input = torch.cat([h, obs_embed], dim=-1)  # (B, det_dim + hidden_dim)
        z_logits = self.posterior_net(post_input)  # (B, stoch_dim)
        z = _straight_through_one_hot(z_logits, self.n_cats, self.cat_dim)  # (B, stoch_dim)

        # 3. GRU step: update h using (z, prev_action)
        gru_in = torch.cat([z, prev_action], dim=-1)  # (B, stoch_dim + act_n)
        h_new = self.gru_cell(gru_in, h)  # (B, det_dim)

        # 4. Prior prediction (h only, no obs) — used for KL loss in training.
        #    Computing this in forward ensures prior_net receives gradients.
        prior_logits = self.prior_net(h_new)  # (B, stoch_dim)

        # 5. Policy head
        feat = torch.cat([h_new, z], dim=-1)  # (B, det_dim + stoch_dim)
        action_logits = self.policy_head(feat)  # (B, action_n)
        action = self._sample_action(action_logits)

        # 6. Value & reward prediction (for aux info)
        value = self.value_head(feat).squeeze(-1)  # (B,)
        reward_pred = self.reward_head(feat).squeeze(-1)  # (B,)
        recon = self.decoder_head(feat)  # (B, obs_in)

        # Build prev_action for next step (one-hot or raw continuous).
        # Always detach: prev_action is state, not a gradient pathway.
        if self.action_spec.discrete:
            next_prev_action = F.one_hot(action, self.action_spec.n).float()
        else:
            next_prev_action = action.detach().clone()

        new_state: BrainState = {
            "h": h_new,
            "z": z,
            "prev_action": next_prev_action,
        }
        aux: AuxInfo = {
            "logits": action_logits,
            "value": value,
            "reward_pred": reward_pred,
            "recon": recon,
            "z_logits": z_logits,
            "prior_z_logits": prior_logits,  # for KL(posterior || prior) loss
        }
        return action, new_state, aux

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    # ── Imaginer Protocol ─────────────────────────────────────────────────

    def imagine(self, obs: Obs, state: BrainState, horizon: int) -> Trajectory:
        """Roll out imagined trajectories using the prior (no obs needed after t=0).

        The first step uses the posterior to get an initial latent from ``obs``.
        Subsequent steps use the prior (open-loop imagination).

        Parameters
        ----------
        obs:
            Initial observation used to seed the posterior at t=0.
        state:
            Initial recurrent state (h, z, prev_action).
        horizon:
            Number of imagined steps ``T``.

        Returns
        -------
        :class:`~evolux.core.types.Trajectory` with ``T = horizon``.
        """
        b = next(iter(obs.values())).shape[0]
        device = next(iter(obs.values())).device

        obs_list: list[dict[str, Tensor]] = []
        action_list: list[Tensor] = []
        reward_list: list[Tensor] = []
        done_list: list[Tensor] = []

        h = state["h"]  # (B, det_dim)
        prev_action = state["prev_action"]  # (B, action_n)

        # Seed posterior from the real obs at t=0
        obs_flat = _encode_obs_flat(obs, self.obs_spec)
        obs_embed = self.obs_encoder(obs_flat)
        post_input = torch.cat([h, obs_embed], dim=-1)
        z_logits = self.posterior_net(post_input)
        z = _straight_through_one_hot(z_logits, self.n_cats, self.cat_dim)

        for _ in range(horizon):
            # GRU step
            gru_in = torch.cat([z, prev_action], dim=-1)
            h = self.gru_cell(gru_in, h)

            # Prior z
            prior_logits = self.prior_net(h)  # (B, stoch_dim)
            z = _straight_through_one_hot(prior_logits, self.n_cats, self.cat_dim)

            feat = torch.cat([h, z], dim=-1)  # (B, det_dim + stoch_dim)

            # Policy
            action_logits = self.policy_head(feat)
            action = self._sample_action(action_logits)

            # Reward
            reward = self.reward_head(feat).squeeze(-1)  # (B,)

            # Reconstructed obs
            recon_flat = self.decoder_head(feat)  # (B, obs_in)

            # Build imagined obs dict (reconstructed flattened, per spec field)
            imag_obs = self._recon_to_obs(recon_flat, device)

            obs_list.append(imag_obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(torch.zeros(b, dtype=torch.bool, device=device))

            # Prepare next action embed
            if self.action_spec.discrete:
                prev_action = F.one_hot(action, self.action_spec.n).float()
            else:
                prev_action = action

        # Stack along time dim → (B, T, ...)
        stacked_obs: dict[str, Tensor] = {}
        obs_keys = list(obs_list[0].keys())
        for k in obs_keys:
            stacked_obs[k] = torch.stack([o[k] for o in obs_list], dim=1)

        actions_t = torch.stack(action_list, dim=1)  # (B, T) or (B, T, A)
        rewards_t = torch.stack(reward_list, dim=1)  # (B, T)
        dones_t = torch.stack(done_list, dim=1)  # (B, T)
        length = torch.full((b,), horizon, dtype=torch.long, device=device)

        return Trajectory(
            obs=stacked_obs,
            actions=actions_t,
            rewards=rewards_t,
            dones=dones_t,
            length=length,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _sample_action(self, logits: Tensor) -> Action:
        """Greedy / bounded action — deterministic so callers control sampling."""
        if self.action_spec.discrete:
            return logits.argmax(dim=-1)
        if self.action_spec.bounds is not None:
            lo, hi = self.action_spec.bounds
            return torch.tanh(logits) * (0.5 * (hi - lo)) + 0.5 * (hi + lo)
        return logits

    def _recon_to_obs(self, recon_flat: Tensor, device: torch.device) -> dict[str, Tensor]:
        """Split a flat reconstruction tensor back into the obs_spec fields."""
        result: dict[str, Tensor] = {}
        b = recon_flat.shape[0]
        offset = 0
        for name, (shape, _) in self.obs_spec.fields.items():
            size = _flat_dim(shape)
            chunk = recon_flat[:, offset : offset + size].reshape(b, *shape)
            result[name] = chunk
            offset += size
        return result


__all__ = ["WorldModelBrain"]

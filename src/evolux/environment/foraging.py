"""Phase-1 foraging environment — eat food, avoid hazards.

``ForagingEnv`` wraps a :class:`~evolux.core.protocols.World` (accessed only
via the Protocol) and applies task-level reward shaping on top of the world's
base signals.

Reward shaping
--------------
- **+food_reward_scale** x (world positive reward) per food item eaten
- **-idle_penalty** per step  (default 0.01)
- **-death_penalty** on death  (default 10.0; applied only when
  ``done=True`` and ``info["timeout"]`` is ``False``)
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from evolux.core.protocols import World
from evolux.core.types import Action, AuxInfo, Done, Obs, Reward
from evolux.environment import ENVIRONMENT_REGISTRY

log = logging.getLogger(__name__)


@ENVIRONMENT_REGISTRY.register("foraging_v1")
class ForagingEnv:
    """Phase-1 foraging environment.

    Wraps a *World* instance (accessed through the
    :class:`~evolux.core.protocols.World` Protocol) and layers task semantics
    on top:

    - Episode terminates when the wrapped world signals ``done``.
    - Reward shaping: +food_reward_scale x (positive reward) per food eaten,
      -idle_penalty per step, -death_penalty on death (not timeout).

    Parameters
    ----------
    world:
        Any object satisfying the :class:`~evolux.core.protocols.World`
        Protocol.  ``ForagingEnv`` never imports the concrete ``GridWorld``
        class.
    idle_penalty:
        Scalar subtracted from the reward every step (default 0.01).
    death_penalty:
        Scalar subtracted when the episode ends with a death, i.e.
        ``done=True`` and ``info["timeout"]=False`` (default 10.0).
    food_reward_scale:
        Multiplier applied to the positive-reward portion returned by the
        world (default 1.0).
    """

    def __init__(
        self,
        world: World,
        idle_penalty: float = 0.01,
        death_penalty: float = 10.0,
        food_reward_scale: float = 1.0,
    ) -> None:
        self.world: World = world
        self._idle_penalty: float = idle_penalty
        self._death_penalty: float = death_penalty
        self._food_reward_scale: float = food_reward_scale

    # ── Environment Protocol ───────────────────────────────────────────────────

    def step(self) -> None:
        """Advance ecological dynamics (no-op in Phase 1).

        In later phases this will handle food regrowth, season changes, etc.
        """

    def fields(self) -> dict[str, Tensor]:
        """Return current ecological field tensors.

        Delegates to ``world.observe()`` and filters to ``Tensor`` values.
        Phase 2 will expose dedicated food / hazard field tensors here.
        """
        obs: Obs = self.world.observe()
        return {k: v for k, v in obs.items() if isinstance(v, Tensor)}

    # ── Additional RL interface ────────────────────────────────────────────────

    def reset(self, mask: Tensor | None = None) -> None:
        """Reset environments; with *mask*, only resets selected indices.

        Parameters
        ----------
        mask:
            Boolean ``(B,)`` tensor.  If ``None``, all envs are reset.
        """
        self.world.reset(mask)

    def observe(self) -> Obs:
        """Return current observations from the wrapped world."""
        return self.world.observe()

    def interact(self, action: Action) -> tuple[Reward, Done, AuxInfo]:
        """Step the world and return a shaped reward signal.

        Applies reward shaping:

        1. Scales positive (food) rewards by ``food_reward_scale``.
        2. Subtracts ``idle_penalty`` every step.
        3. Subtracts ``death_penalty`` when ``done=True`` *and* not a timeout.

        Parameters
        ----------
        action:
            ``(B, A)`` action tensor forwarded verbatim to ``world.step``.

        Returns
        -------
        reward : Tensor
            ``(B,)`` shaped reward.
        done : Tensor
            ``(B,)`` bool — episode termination flag (from world).
        info : dict
            Auxiliary information forwarded from the world, extended with
            ``"death"`` (bool tensor) and ``"shaped_reward"`` (float tensor).
        """
        base_reward: Reward
        done: Done
        raw_info: Any

        base_reward, done, raw_info = self.world.step(action)

        # Positive signal = food eaten
        food: Tensor = base_reward.clamp(min=0.0) * self._food_reward_scale

        # Idle penalty every step
        reward: Tensor = food - self._idle_penalty

        # Death penalty: done without timeout
        timeout: Tensor = raw_info.get(
            "timeout",
            torch.zeros_like(done),
        )
        death: Tensor = done & ~timeout
        reward = reward - self._death_penalty * death.float()

        # Shallow-copy info so we don't mutate the world's dict
        info: AuxInfo = dict(raw_info)
        info["death"] = death
        info["shaped_reward"] = reward

        return reward, done, info

"""Phase-1 discrete grid world -- ``(B, H, W, C)`` tensor world.

:class:`GridWorld` implements the :class:`~evolux.core.protocols.World`
Protocol for ``B`` parallel environments laid out on a fixed ``H x W``
grid.  Each cell is a float32 vector over four channels:

    0 -- wall          (1 = blocked, 0 = passable)
    1 -- food          (1 = food present, 0 = empty)
    2 -- agent occ.    (1 = an agent is here)
    3 -- hazard        (1 = hazard cell)

The world is *stateful*: reset, step, and observe mutate internal tensors
in-place.  For determinism, all randomness flows through :class:`RNG`.

Usage::

    from evolux.core.rng import RNG
    from evolux.world.grid import GridWorld
    import torch

    rng = RNG(seed=42)
    world = GridWorld(batch_size=8, height=16, width=16, rng=rng)
    world.reset()

    action = torch.zeros(8, 2)
    reward, done, info = world.step(action)
    obs = world.observe()
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

from evolux.core.rng import RNG
from evolux.core.types import Action, AuxInfo, Done, Obs, ObsSpec, Reward
from evolux.physics.discrete import DiscretePhysics
from evolux.world.observation import build_obs

logger = logging.getLogger(__name__)

# ---- Channel indices ----------------------------------------------------------
CHANNEL_WALL: int = 0
CHANNEL_FOOD: int = 1
CHANNEL_OCCUPANCY: int = 2
CHANNEL_HAZARD: int = 3
N_CHANNELS: int = 4

# Proprioception feature dim: [energy, hunger, last_action]
_PROPRIO_DIM: int = 3


class GridWorld:
    """Batched ``(B, H, W, C)`` grid world implementing the World Protocol.

    Parameters
    ----------
    batch_size:
        Number of parallel environments ``B``.
    height:
        Grid height in cells ``H``.
    width:
        Grid width in cells ``W``.
    crop_size:
        Side length ``k`` of the local vision crop (must be odd).
    max_steps:
        Maximum steps per episode before ``done`` fires.
    food_density:
        Bernoulli probability that each non-wall cell spawns food on reset.
    initial_energy:
        Starting energy for each agent (also the max).
    energy_per_food:
        Energy gained when an agent eats a food cell.
    step_energy_cost:
        Base energy drained every step. Actual cost is
        ``step_energy_cost * (1 + action_cost)`` where
        ``action_cost`` is the sum-of-squares of the action vector.
    rng:
        :class:`~evolux.core.rng.RNG` instance for reproducibility.
        If ``None``, a fresh RNG with seed 0 is used.
    device:
        PyTorch device for all tensors.
    """

    def __init__(
        self,
        batch_size: int,
        height: int = 32,
        width: int = 32,
        crop_size: int = 5,
        max_steps: int = 200,
        food_density: float = 0.1,
        initial_energy: float = 1.0,
        energy_per_food: float = 0.2,
        step_energy_cost: float = 0.01,
        rng: RNG | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.batch_size: int = batch_size
        self.device: torch.device = torch.device(device)

        self.height: int = height
        self.width: int = width
        self.crop_size: int = crop_size
        self.max_steps: int = max_steps
        self.food_density: float = food_density
        self.initial_energy: float = initial_energy
        self.energy_per_food: float = energy_per_food
        self.step_energy_cost: float = step_energy_cost

        self._rng: RNG = rng if rng is not None else RNG(seed=0)
        self._physics: DiscretePhysics = DiscretePhysics(height, width)

        # Declared observation spec
        self.obs_spec: ObsSpec = ObsSpec(
            fields={
                "vision": ((N_CHANNELS, crop_size, crop_size), torch.float32),
                "proprio": ((_PROPRIO_DIM,), torch.float32),
            }
        )

        # Internal state tensors
        self._world: Tensor = torch.zeros(batch_size, height, width, N_CHANNELS, device=self.device)
        self._positions: Tensor = torch.zeros(batch_size, 2, dtype=torch.long, device=self.device)
        self._headings: Tensor = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._energy: Tensor = torch.full((batch_size,), initial_energy, device=self.device)
        self._hunger: Tensor = torch.zeros(batch_size, device=self.device)
        self._last_action: Tensor = torch.zeros(batch_size, device=self.device)
        self._step_count: Tensor = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Initialise all environments on construction
        self.reset()

    # ---- Protocol methods -----------------------------------------------------

    def reset(self, mask: Tensor | None = None) -> None:
        """Reset environments indicated by *mask*.

        Parameters
        ----------
        mask:
            Bool tensor of shape ``(B,)``.  If ``None`` all envs are reset.
            Environments where ``mask[b] == False`` are left unchanged.
        """
        B = self.batch_size
        if mask is None:
            mask = torch.ones(B, dtype=torch.bool, device=self.device)
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)

        if not mask.any():
            return

        indices = mask.nonzero(as_tuple=True)[0]  # (n_reset,)
        n = int(indices.shape[0])

        # Spawn food (Bernoulli)
        food_gen = self._rng.split("food_spawn")
        food_mask = torch.bernoulli(
            torch.full((n, self.height, self.width), self.food_density, device=self.device),
            generator=food_gen,
        )  # (n, H, W)

        # Assign random positions
        pos_gen = self._rng.split("pos_spawn")
        row_rand = torch.randint(0, self.height, (n,), generator=pos_gen, device=self.device)
        col_rand = torch.randint(0, self.width, (n,), generator=pos_gen, device=self.device)

        # Assign random headings
        head_gen = self._rng.split("head_spawn")
        head_rand = torch.randint(0, 4, (n,), generator=head_gen, device=self.device)

        # Build new world slices
        n_idx = torch.arange(n, device=self.device)
        new_world = torch.zeros(n, self.height, self.width, N_CHANNELS, device=self.device)
        new_world[:, :, :, CHANNEL_FOOD] = food_mask
        new_world[n_idx, row_rand, col_rand, CHANNEL_OCCUPANCY] = 1.0
        # Remove food at spawn position
        new_world[n_idx, row_rand, col_rand, CHANNEL_FOOD] = 0.0

        # Apply to the batch
        self._world[indices] = new_world
        self._positions[indices] = torch.stack([row_rand, col_rand], dim=1)
        self._headings[indices] = head_rand
        self._energy[indices] = self.initial_energy
        self._hunger[indices] = 0.0
        self._last_action[indices] = 0.0
        self._step_count[indices] = 0

    def observe(self) -> Obs:
        """Return the current observation dict.

        Returns
        -------
        dict with keys:
            ``"vision"``  -- ``(B, C, k, k)`` float32 local crop
            ``"proprio"`` -- ``(B, 3)`` float32 ``[energy, hunger, last_action]``
        """
        max_hunger = float(self.max_steps)
        hunger_norm = (self._hunger / max(max_hunger, 1.0)).clamp(0.0, 1.0)

        return build_obs(
            world=self._world,
            positions=self._positions,
            headings=self._headings,
            energy=self._energy.clamp(0.0, 1.0),
            hunger=hunger_norm,
            last_action=self._last_action,
            crop_size=self.crop_size,
        )

    def step(self, action: Action) -> tuple[Reward, Done, AuxInfo]:
        """Advance all agents by one step.

        Parameters
        ----------
        action:
            ``(B, A)`` float32 action tensor.
            ``action[:, 0]`` -- forward (> 0.5 moves 1 cell).
            ``action[:, 1]`` -- turn (> 0.5 right, < -0.5 left).

        Returns
        -------
        reward : ``(B,)`` float32
        done   : ``(B,)`` bool
        info   : dict with extra diagnostics
        """
        B = self.batch_size
        device = self.device

        # Ensure action is 2-D and has at least 2 columns
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.shape[1] < 2:
            action = torch.cat([action, torch.zeros(B, 2 - action.shape[1], device=device)], dim=1)

        # Physics step
        new_pos, new_head, collision_mask, energy_cost = self._physics.step(
            self._world, self._positions, self._headings, action
        )

        # Update occupancy channel
        b_idx = torch.arange(B, device=device)
        self._world[b_idx, self._positions[:, 0], self._positions[:, 1], CHANNEL_OCCUPANCY] = 0.0
        self._world[b_idx, new_pos[:, 0], new_pos[:, 1], CHANNEL_OCCUPANCY] = 1.0

        # Eat food at new position
        food_at_pos = self._world[b_idx, new_pos[:, 0], new_pos[:, 1], CHANNEL_FOOD]  # (B,)
        ate_food = food_at_pos > 0.5  # (B,) bool
        self._world[b_idx, new_pos[:, 0], new_pos[:, 1], CHANNEL_FOOD] = torch.where(
            ate_food, torch.zeros(B, device=device), food_at_pos
        )

        # Update energy:
        # total cost = step_energy_cost * (1 + action_cost)
        #   where action_cost = sum-of-squares of action magnitudes.
        # This gives a fixed idle penalty scaled by the same factor as
        # the variable action cost, keeping both terms in the same units.
        energy_gain = ate_food.float() * self.energy_per_food
        self._energy = (
            self._energy - self.step_energy_cost * (1.0 + energy_cost) + energy_gain
        ).clamp(0.0, self.initial_energy)

        # Update hunger
        self._hunger = self._hunger + 1.0
        self._hunger = torch.where(ate_food, torch.zeros(B, device=device), self._hunger)

        # Update positions / headings / step counter
        self._positions = new_pos
        self._headings = new_head
        self._step_count = self._step_count + 1

        # Store last forward action
        self._last_action = action[:, 0].clamp(-1.0, 1.0)

        # Reward: +1 per food eaten
        reward: Reward = ate_food.float()  # (B,)

        # Done condition
        energy_done = self._energy <= 0.0  # (B,)
        step_done = self._step_count >= self.max_steps  # (B,)
        done: Done = energy_done | step_done  # (B,) bool

        info: AuxInfo = {
            "collision": collision_mask,
            "ate_food": ate_food,
            "energy": self._energy.clone(),
            "step_count": self._step_count.clone(),
        }

        return reward, done, info

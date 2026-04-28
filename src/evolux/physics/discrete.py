"""Discrete grid-cell physics for Phase-1 worlds.

Implements :class:`DiscretePhysics`, which handles per-step movement,
collision detection, and occupancy updates for a batch of ``B`` agents
on a ``(B, H, W, C)`` world tensor.

Channel layout (must match ``world.grid.CHANNEL_*`` constants):

    0 -- wall (1 = blocked)
    1 -- food (1 = food present)
    2 -- agent occupancy (1 = agent present)
    3 -- hazard (1 = hazard present)

Heading encoding (integers 0-3):

    0 = North  (drow = -1, dcol =  0)
    1 = East   (drow =  0, dcol = +1)
    2 = South  (drow = +1, dcol =  0)
    3 = West   (drow =  0, dcol = -1)
"""

from __future__ import annotations

import torch
from torch import Tensor

from evolux.physics.energy import compute_action_cost

# Heading → (Δrow, Δcol) mapping, indexed 0..3
_ROW_DELTA = torch.tensor([-1, 0, 1, 0], dtype=torch.long)
_COL_DELTA = torch.tensor([0, 1, 0, -1], dtype=torch.long)

# Channel indices — must match GridWorld.CHANNEL_*
_WALL_CH: int = 0


class DiscretePhysics:
    """Batched grid-cell physics for discrete worlds.

    All operations are fully vectorised over the batch dimension ``B``.
    There are no Python-level loops over individual agents.

    Parameters
    ----------
    height:
        Grid height in cells.
    width:
        Grid width in cells.
    """

    def __init__(self, height: int, width: int) -> None:
        self.height: int = height
        self.width: int = width

    # ── Public interface ──────────────────────────────────────────────────────

    def step(
        self,
        world: Tensor,
        positions: Tensor,
        headings: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance all agents by one discrete step.

        Parameters
        ----------
        world:
            ``(B, H, W, C)`` float32 world tensor.  Must already include
            the occupancy channel so collision checks are consistent.
        positions:
            ``(B, 2)`` int64 agent grid positions ``[row, col]``.
        headings:
            ``(B,)`` int64 agent headings (0 = N, 1 = E, 2 = S, 3 = W).
        actions:
            ``(B, A)`` float32 action tensor.
            ``actions[:, 0]`` -- forward impulse (> 0.5 moves agent).
            ``actions[:, 1]`` -- turn signal (> 0.5 right, < -0.5 left).

        Returns
        -------
        new_positions : ``(B, 2)`` int64
        new_headings  : ``(B,)`` int64
        collision_mask: ``(B,)`` bool -- True where agent was blocked
        energy_cost   : ``(B,)`` float32
        """
        device = positions.device

        # ---- 1. Turn ---------------------------------------------------------
        if actions.shape[1] > 1:
            turn_signal = actions[:, 1]
        else:
            turn_signal = torch.zeros(actions.shape[0], device=device)
        turn_right = turn_signal > 0.5  # (B,)
        turn_left = turn_signal < -0.5  # (B,)
        new_headings = (headings + turn_right.long() - turn_left.long()) % 4  # (B,)

        # ── 2. Compute intended new positions ─────────────────────────────────
        move_fwd = actions[:, 0] > 0.5  # (B,) bool

        row_d = _ROW_DELTA.to(device)[new_headings]  # (B,)
        col_d = _COL_DELTA.to(device)[new_headings]  # (B,)

        intended_row = positions[:, 0] + row_d * move_fwd.long()  # (B,)
        intended_col = positions[:, 1] + col_d * move_fwd.long()  # (B,)

        # ── 3. Clip to world boundaries ───────────────────────────────────────
        clamped_row = intended_row.clamp(0, self.height - 1)  # (B,)
        clamped_col = intended_col.clamp(0, self.width - 1)  # (B,)

        # Boundary collision: tried to move outside world
        boundary_collision = (intended_row != clamped_row) | (intended_col != clamped_col)  # (B,)

        # ── 4. Wall collision ─────────────────────────────────────────────────
        B = positions.shape[0]
        b_idx = torch.arange(B, device=device)
        wall_at_target = world[b_idx, clamped_row, clamped_col, _WALL_CH] > 0.5  # (B,)

        # ── 5. Resolve: blocked agents stay put ───────────────────────────────
        collision_mask = (boundary_collision | wall_at_target) & move_fwd  # (B,)

        new_row = torch.where(collision_mask, positions[:, 0], clamped_row)
        new_col = torch.where(collision_mask, positions[:, 1], clamped_col)
        new_positions = torch.stack([new_row, new_col], dim=1)  # (B, 2)

        # ── 6. Energy cost ────────────────────────────────────────────────────
        energy_cost = compute_action_cost(actions).to(dtype=torch.float32)  # (B,)

        return new_positions, new_headings, collision_mask, energy_cost

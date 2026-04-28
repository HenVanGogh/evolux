"""Discrete grid-cell physics — movement, occupancy, collision detection.

``DiscretePhysics`` operates on a ``(B, H, W, C)`` world tensor and
``(B, 2)`` integer agent positions (row, col).  It is the Phase-1 physics
back-end used by ``GridWorld``.

Headings are encoded as floats in ``{0.0, 1.0, 2.0, 3.0}`` where

    0 -> North (row - 1)
    1 -> East  (col + 1)
    2 -> South (row + 1)
    3 -> West  (col - 1)

This matches the four-directional convention used in the morphology module.
"""

from __future__ import annotations

import torch
from torch import Tensor

from evolux.physics.energy import compute_action_cost

# Row / column deltas indexed by heading 0-3
_ROW_DELTA = torch.tensor([-1, 0, 1, 0], dtype=torch.long)  # indexed by heading: 0=N,1=E,2=S,3=W
_COL_DELTA = torch.tensor([0, 1, 0, -1], dtype=torch.long)  # indexed by heading: 0=N,1=E,2=S,3=W


class DiscretePhysics:
    """Vectorised discrete-grid physics for *B* parallel environments.

    Parameters
    ----------
    world_h:
        Grid height (number of rows).
    world_w:
        Grid width (number of columns).
    wall_channel:
        Index of the wall channel in the ``C`` dimension of the world tensor.
        Cells with ``world[b, r, c, wall_channel] > 0.5`` are solid walls.
    wrap:
        If ``True``, positions wrap at the grid boundary (toroidal topology).
        If ``False`` (default), movement is clamped to ``[0, H-1] x [0, W-1]``
        and out-of-bounds moves count as collisions.
    """

    def __init__(
        self,
        world_h: int,
        world_w: int,
        wall_channel: int = 0,
        wrap: bool = False,
    ) -> None:
        if world_h <= 0 or world_w <= 0:
            raise ValueError(f"world_h and world_w must be > 0, got {world_h}x{world_w}.")
        if wall_channel < 0:
            raise ValueError(f"wall_channel must be >= 0, got {wall_channel}.")
        self.world_h: int = world_h
        self.world_w: int = world_w
        self.wall_channel: int = wall_channel
        self.wrap: bool = wrap

    def step(
        self,
        world: Tensor,
        positions: Tensor,
        headings: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Advance one physics tick for all creatures in the batch.

        Parameters
        ----------
        world:
            ``(B, H, W, C)`` float tensor encoding grid channels.
        positions:
            ``(B, 2)`` float tensor of current ``(row, col)`` positions.
        headings:
            ``(B,)`` float tensor of current headings in ``{0, 1, 2, 3}``.
        actions:
            ``(B, A)`` float tensor where

            * ``actions[:, 0] > 0.5`` → move forward one cell.
            * ``actions[:, 1] > 0``   → turn right 90°.
            * ``actions[:, 1] < 0``   → turn left 90°.

            Extra action columns (A > 2) are ignored for movement but
            included in the energy-cost sum.

        Returns
        -------
        new_positions : Tensor
            ``(B, 2)`` updated positions (float32, row/col).
        new_headings : Tensor
            ``(B,)`` updated headings (float32, values in ``{0, 1, 2, 3}``).
        collision_mask : Tensor
            ``(B,)`` bool — ``True`` where movement was blocked by a wall.
        energy_cost : Tensor
            ``(B,)`` float32 — sum-of-squares action cost per creature.
        """
        device = positions.device

        # ── 1. Apply turns ────────────────────────────────────────────────────
        # Turn right when actions[:, 1] > 0; turn left when < 0.
        turn_action = actions[:, 1]
        turn_right = (turn_action > 0).long()
        turn_left = (turn_action < 0).long()
        new_headings_long = (headings.long() + turn_right - turn_left) % 4

        # ── 2. Compute candidate target positions ─────────────────────────────
        should_move = actions[:, 0] > 0.5  # (B,) bool

        row_delta = _ROW_DELTA.to(device)
        col_delta = _COL_DELTA.to(device)

        dr = row_delta[new_headings_long]  # (B,)
        dc = col_delta[new_headings_long]  # (B,)

        curr_row = positions[:, 0].long()
        curr_col = positions[:, 1].long()

        target_row = curr_row + dr * should_move.long()
        target_col = curr_col + dc * should_move.long()

        # ── 3. Boundary handling ──────────────────────────────────────────────
        if self.wrap:
            target_row = target_row % self.world_h
            target_col = target_col % self.world_w
        else:
            # Out-of-bounds → clamp and mark as collision
            out_of_bounds = (
                (target_row < 0)
                | (target_row >= self.world_h)
                | (target_col < 0)
                | (target_col >= self.world_w)
            )
            target_row = target_row.clamp(0, self.world_h - 1)
            target_col = target_col.clamp(0, self.world_w - 1)

        # ── 4. Wall collision detection ───────────────────────────────────────
        batch_idx = torch.arange(positions.shape[0], device=device)
        wall_at_target = world[batch_idx, target_row, target_col, self.wall_channel] > 0.5

        if not self.wrap:
            wall_collision = (wall_at_target | out_of_bounds) & should_move
        else:
            wall_collision = wall_at_target & should_move

        # Block movement for colliding agents
        final_row = torch.where(wall_collision, curr_row, target_row)
        final_col = torch.where(wall_collision, curr_col, target_col)

        new_positions = torch.stack([final_row.float(), final_col.float()], dim=1)

        # ── 5. Energy cost ────────────────────────────────────────────────────
        energy_cost = compute_action_cost(actions)

        return new_positions, new_headings_long.float(), wall_collision, energy_cost

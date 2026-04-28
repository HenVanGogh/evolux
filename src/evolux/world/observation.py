"""Observation builder — local-crop vision + proprioception.

These helpers extract per-agent observations from the ``(B, H, W, C)``
world tensor.  They are kept in a separate file so that :mod:`world.grid`
stays concise and these primitives are independently testable.

Shapes used throughout
----------------------
B  — batch size (parallel envs)
H  — grid height
W  — grid width
C  — number of world channels
k  — local crop side length (crop_size)
P  — proprioception feature dim (3)
"""

from __future__ import annotations

import torch
from torch import Tensor


def extract_local_crop(
    world: Tensor,
    positions: Tensor,
    headings: Tensor,
    crop_size: int,
) -> Tensor:
    """Extract a ``k x k`` agent-centric crop from the world tensor.

    The crop is centred on the agent's grid cell and padded with "wall"
    values (channel-0 = 1, all others = 0) for cells outside the grid.
    The result is rotated so that the agent's heading faces "up" (dim -2).

    Parameters
    ----------
    world:
        ``(B, H, W, C)`` float32 world tensor.
    positions:
        ``(B, 2)`` int64 agent positions ``[row, col]``.
    headings:
        ``(B,)`` int64 agent headings (0 = N, 1 = E, 2 = S, 3 = W).
    crop_size:
        Odd integer ``k``.  The extracted crop is ``(k, k)`` cells.

    Returns
    -------
    Tensor
        ``(B, C, k, k)`` float32 cropped and heading-aligned observation.
    """
    B, H, W, C = world.shape
    k = crop_size
    pad = k // 2
    device = world.device

    # ── Pad the world: (B, H, W, C) → (B, C, H+2p, W+2p) ────────────────────
    # Fill padding with wall = 1.0 on channel 0, 0.0 elsewhere.
    world_c = world.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

    padded = torch.zeros(B, C, H + 2 * pad, W + 2 * pad, device=device)
    padded[:, 0, :, :] = 1.0  # fill with walls
    padded[:, :, pad : pad + H, pad : pad + W] = world_c

    # ---- Extract k x k crops ------------------------------------------------
    # positions[:, 0] is row (with pad offset applied below)
    rows = positions[:, 0]  # (B,)
    cols = positions[:, 1]  # (B,)

    crops = torch.stack(
        [padded[b, :, rows[b] : rows[b] + k, cols[b] : cols[b] + k] for b in range(B)],
        dim=0,
    )  # (B, C, k, k)

    # ── Rotate to agent frame ─────────────────────────────────────────────────
    # heading 0 (N): no rotation — "up" is already north
    # heading 1 (E): rotate 90° CW → 3 * 90° CCW
    # heading 2 (S): rotate 180° → 2 * 90° CCW
    # heading 3 (W): rotate 90° CCW → 1 * 90° CCW
    # torch.rot90(k=n) rotates n * 90° CCW.
    _heading_to_rot90 = [0, 3, 2, 1]
    rotated = crops.clone()
    for h_val, n_rot in enumerate(_heading_to_rot90):
        if n_rot == 0:
            continue
        mask = headings == h_val
        if mask.any():
            rotated[mask] = torch.rot90(crops[mask], n_rot, dims=(-2, -1))

    return rotated  # (B, C, k, k)


def build_obs(
    world: Tensor,
    positions: Tensor,
    headings: Tensor,
    energy: Tensor,
    hunger: Tensor,
    last_action: Tensor,
    crop_size: int,
) -> dict[str, Tensor]:
    """Build the observation dict from world state.

    Parameters
    ----------
    world:
        ``(B, H, W, C)`` float32 world tensor.
    positions:
        ``(B, 2)`` int64 agent positions.
    headings:
        ``(B,)`` int64 agent headings.
    energy:
        ``(B,)`` float32 current energy level (normalised to [0, 1]).
    hunger:
        ``(B,)`` float32 steps since last food (normalised).
    last_action:
        ``(B,)`` float32 last forward action (normalised to [0, 1]).
    crop_size:
        Odd integer ``k`` for the local vision crop.

    Returns
    -------
    dict with keys:
        ``"vision"``  — ``(B, C, k, k)`` float32
        ``"proprio"`` — ``(B, 3)`` float32 ``[energy, hunger, last_action]``
    """
    vision = extract_local_crop(world, positions, headings, crop_size)  # (B, C, k, k)
    proprio = torch.stack([energy, hunger, last_action], dim=-1)  # (B, 3)
    return {"vision": vision, "proprio": proprio}

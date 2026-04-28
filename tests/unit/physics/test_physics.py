"""Unit tests for the Phase-1 physics module.

Covers all acceptance tests from ``src/evolux/physics/SPEC.md``:

- Movement is bounded by world dims (with or without wrap).
- Collisions resolved deterministically per RNG seed.
- Energy is conservative (cost ≥ 0) within rounding.
"""

from __future__ import annotations

import torch

from evolux.physics.discrete import DiscretePhysics
from evolux.physics.energy import compute_action_cost

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_world(B: int, H: int, W: int, C: int = 4, wall_channel: int = 0) -> torch.Tensor:
    """Return an empty (B, H, W, C) world tensor (no walls)."""
    return torch.zeros(B, H, W, C)


def _add_wall(world: torch.Tensor, row: int, col: int, wall_channel: int = 0) -> torch.Tensor:
    """Place a wall at (row, col) in every batch item."""
    world = world.clone()
    world[:, row, col, wall_channel] = 1.0
    return world


def _physics(H: int = 8, W: int = 8, wrap: bool = False) -> DiscretePhysics:
    return DiscretePhysics(world_h=H, world_w=W, wall_channel=0, wrap=wrap)


# ── Energy cost (SPEC: energy ≥ 0) ───────────────────────────────────────────


def test_energy_cost_non_negative() -> None:
    """compute_action_cost always returns non-negative values."""
    actions = torch.randn(16, 4)
    cost = compute_action_cost(actions)
    assert cost.shape == (16,)
    assert torch.all(cost >= 0.0)


def test_energy_cost_zero_for_zero_action() -> None:
    """Zero action has zero cost."""
    actions = torch.zeros(4, 2)
    cost = compute_action_cost(actions)
    assert torch.allclose(cost, torch.zeros(4))


def test_energy_cost_sum_of_squares() -> None:
    """Cost equals sum of squared action components."""
    actions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([1.0 + 4.0, 9.0 + 16.0])
    assert torch.allclose(compute_action_cost(actions), expected)


def test_energy_cost_shape() -> None:
    """Output shape is (B,) for (B, A) input."""
    for A in (1, 2, 5):
        actions = torch.ones(8, A)
        cost = compute_action_cost(actions)
        assert cost.shape == (8,)


# ── Bounded movement (SPEC: positions stay within world dims) ─────────────────


def test_positions_stay_within_bounds_no_wrap() -> None:
    """Agents never leave the grid when wrap=False."""
    B, H, W = 32, 8, 8
    physics = _physics(H, W, wrap=False)
    world = _make_world(B, H, W)

    # Place agents at corners/edges pointing outward
    positions = torch.tensor(
        [[0.0, 0.0]] * B,  # top-left corner
    )
    headings = torch.zeros(B)  # heading 0 = North (row decreases)

    # Try moving forward (action 0 > 0.5)
    actions = torch.ones(B, 2)
    actions[:, 1] = 0.0  # no turn

    new_positions, _, _, _ = physics.step(world, positions, headings, actions)

    assert torch.all(new_positions[:, 0] >= 0)
    assert torch.all(new_positions[:, 0] < H)
    assert torch.all(new_positions[:, 1] >= 0)
    assert torch.all(new_positions[:, 1] < W)


def test_positions_wrap_correctly() -> None:
    """With wrap=True, agents wrap around the grid boundary."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W, wrap=True)
    world = _make_world(B, H, W)

    # Place agents at row 0, facing North (heading 0) → should wrap to row H-1
    positions = torch.zeros(B, 2)  # (0, 0)
    headings = torch.zeros(B)  # North

    actions = torch.ones(B, 2)
    actions[:, 1] = 0.0

    new_positions, _, _, _ = physics.step(world, positions, headings, actions)

    assert torch.all(new_positions[:, 0] == H - 1)
    assert torch.all(new_positions[:, 0] >= 0)
    assert torch.all(new_positions[:, 0] < H)


def test_no_move_when_forward_action_below_threshold() -> None:
    """Agent does not move if forward action ≤ 0.5."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)
    world = _make_world(B, H, W)

    positions = torch.full((B, 2), 4.0)  # centre
    headings = torch.ones(B)  # East

    actions = torch.zeros(B, 2)
    actions[:, 0] = 0.3  # below 0.5 threshold

    new_positions, _, collision_mask, _ = physics.step(world, positions, headings, actions)

    assert torch.all(new_positions == positions)
    assert not torch.any(collision_mask)


# ── Collision detection (SPEC: collisions block movement) ─────────────────────


def test_wall_collision_blocks_movement() -> None:
    """Agent does not move into a wall cell."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)

    # Place wall at (4, 5)
    world = _make_world(B, H, W)
    world = _add_wall(world, row=4, col=5)

    # Place agents at (4, 4) facing East (heading 1)
    positions = torch.tensor([[4.0, 4.0]] * B)
    headings = torch.ones(B)  # East

    # Try to move forward
    actions = torch.ones(B, 2)
    actions[:, 1] = 0.0

    new_positions, _, collision_mask, _ = physics.step(world, positions, headings, actions)

    # Position unchanged (blocked by wall)
    assert torch.all(new_positions == positions)
    assert torch.all(collision_mask)


def test_no_collision_on_empty_cell() -> None:
    """Agent moves freely when target cell has no wall."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)
    world = _make_world(B, H, W)

    positions = torch.tensor([[3.0, 3.0]] * B)
    headings = torch.ones(B)  # East → col+1

    actions = torch.ones(B, 2)
    actions[:, 1] = 0.0

    new_positions, _, collision_mask, _ = physics.step(world, positions, headings, actions)

    expected = torch.tensor([[3.0, 4.0]] * B)
    assert torch.all(new_positions == expected)
    assert not torch.any(collision_mask)


def test_collision_mask_false_when_not_moving() -> None:
    """collision_mask is False when agent is not attempting to move."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)
    world = _add_wall(_make_world(B, H, W), row=4, col=5)

    positions = torch.tensor([[4.0, 4.0]] * B)
    headings = torch.ones(B)

    # No forward movement
    actions = torch.zeros(B, 2)

    _, _, collision_mask, _ = physics.step(world, positions, headings, actions)
    assert not torch.any(collision_mask)


# ── Turning mechanics ─────────────────────────────────────────────────────────


def test_turn_right_updates_heading() -> None:
    """Positive actions[:, 1] rotates heading clockwise (+1 mod 4)."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)
    world = _make_world(B, H, W)

    positions = torch.full((B, 2), 4.0)
    headings = torch.zeros(B)  # North (0)

    actions = torch.zeros(B, 2)
    actions[:, 1] = 1.0  # turn right

    _, new_headings, _, _ = physics.step(world, positions, headings, actions)
    assert torch.all(new_headings == 1.0)  # East


def test_turn_left_updates_heading() -> None:
    """Negative actions[:, 1] rotates heading counter-clockwise (-1 mod 4)."""
    B, H, W = 4, 8, 8
    physics = _physics(H, W)
    world = _make_world(B, H, W)

    positions = torch.full((B, 2), 4.0)
    headings = torch.zeros(B)  # North (0)

    actions = torch.zeros(B, 2)
    actions[:, 1] = -1.0  # turn left

    _, new_headings, _, _ = physics.step(world, positions, headings, actions)
    assert torch.all(new_headings == 3.0)  # West


def test_full_rotation() -> None:
    """Four right turns should return to original heading."""
    B, H, W = 2, 8, 8
    physics = _physics(H, W)
    world = _make_world(B, H, W)
    positions = torch.full((B, 2), 4.0)

    headings = torch.zeros(B)
    actions = torch.zeros(B, 2)
    actions[:, 1] = 1.0

    for _ in range(4):
        _, headings, _, _ = physics.step(world, positions, headings, actions)

    assert torch.all(headings == 0.0)


# ── Determinism (SPEC: resolved deterministically per RNG seed) ───────────────


def test_step_is_deterministic() -> None:
    """Same inputs always produce the same outputs."""
    B, H, W = 8, 8, 8
    physics = _physics(H, W)

    world = _make_world(B, H, W)
    world = _add_wall(world, row=3, col=3)

    positions = torch.rand(B, 2) * 6 + 1  # positions in [1, 7)
    headings = torch.randint(0, 4, (B,)).float()
    actions = torch.rand(B, 2)

    out1 = physics.step(world, positions, headings, actions)
    out2 = physics.step(world, positions, headings, actions)

    for t1, t2 in zip(out1, out2, strict=True):
        assert torch.all(t1 == t2)


# ── Performance budget (<=1 ms CPU, B=64, 32x32) ─────────────────────────────


def test_step_performance(benchmark: object) -> None:
    """Discrete step at B=64, 32x32 should complete well within 1 ms."""
    B, H, W = 64, 32, 32
    physics = _physics(H, W)
    world = _make_world(B, H, W)
    positions = torch.randint(1, 30, (B, 2)).float()
    headings = torch.randint(0, 4, (B,)).float()
    actions = torch.rand(B, 2)

    benchmark(physics.step, world, positions, headings, actions)

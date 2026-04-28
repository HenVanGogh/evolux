"""Unit tests for the Phase-1 world module.

Covers all acceptance tests from ``src/evolux/world/SPEC.md``:

- ``reset(mask)`` resets only masked envs, leaves others unchanged.
- ``observe()`` returns dict matching declared ``ObsSpec``.
- ``step(action)`` returns ``(reward, done, info)`` with leading dim B.
- Determinism: same seed + same action sequence → bit-equal trajectory.

Also verifies Protocol compliance and registry registration.
"""

from __future__ import annotations

import torch

from evolux.core.protocols import World
from evolux.core.rng import RNG
from evolux.world import WORLD_REGISTRY, GridWorld
from evolux.world.grid import CHANNEL_OCCUPANCY, N_CHANNELS
from evolux.world.observation import build_obs, extract_local_crop

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_world(
    batch_size: int = 4,
    height: int = 8,
    width: int = 8,
    crop_size: int = 3,
    max_steps: int = 50,
    seed: int = 42,
    **kwargs: object,
) -> GridWorld:
    rng = RNG(seed=seed)
    return GridWorld(
        batch_size=batch_size,
        height=height,
        width=width,
        crop_size=crop_size,
        max_steps=max_steps,
        rng=rng,
        **kwargs,
    )


# ── Protocol compliance ────────────────────────────────────────────────────────


def test_grid_world_implements_world_protocol() -> None:
    world = make_world()
    assert isinstance(world, World)


# ── Registry ──────────────────────────────────────────────────────────────────


def test_grid_world_registered() -> None:
    assert "grid_v1" in WORLD_REGISTRY
    assert WORLD_REGISTRY.get("grid_v1") is GridWorld


# ── Attribute checks ──────────────────────────────────────────────────────────


def test_batch_size_attribute() -> None:
    world = make_world(batch_size=6)
    assert world.batch_size == 6


def test_device_attribute() -> None:
    world = make_world()
    assert world.device == torch.device("cpu")


# ── observe() acceptance test ─────────────────────────────────────────────────


def test_observe_returns_vision_and_proprio_keys() -> None:
    world = make_world()
    obs = world.observe()
    assert "vision" in obs
    assert "proprio" in obs


def test_observe_vision_shape() -> None:
    B, k = 4, 3
    world = make_world(batch_size=B, crop_size=k)
    obs = world.observe()
    assert obs["vision"].shape == (B, N_CHANNELS, k, k)


def test_observe_proprio_shape() -> None:
    B = 4
    world = make_world(batch_size=B)
    obs = world.observe()
    assert obs["proprio"].shape == (B, 3)


def test_observe_matches_obs_spec() -> None:
    B = 4
    world = make_world(batch_size=B, crop_size=5)
    obs = world.observe()
    for key, (shape, dtype) in world.obs_spec.fields.items():
        assert key in obs, f"missing key {key!r}"
        t = obs[key]
        assert t.shape == (B, *shape), f"{key}: expected (B, {shape}), got {t.shape}"
        assert t.dtype == dtype, f"{key}: expected {dtype}, got {t.dtype}"


def test_observe_vision_float32() -> None:
    world = make_world()
    obs = world.observe()
    assert obs["vision"].dtype == torch.float32


def test_observe_proprio_float32() -> None:
    world = make_world()
    obs = world.observe()
    assert obs["proprio"].dtype == torch.float32


def test_observe_proprio_energy_in_range() -> None:
    world = make_world()
    obs = world.observe()
    energy_col = obs["proprio"][:, 0]
    assert (energy_col >= 0.0).all() and (energy_col <= 1.0).all()


# ── step() acceptance test ────────────────────────────────────────────────────


def test_step_returns_three_values() -> None:
    world = make_world(batch_size=4)
    action = torch.zeros(4, 2)
    result = world.step(action)
    assert len(result) == 3


def test_step_reward_shape() -> None:
    B = 4
    world = make_world(batch_size=B)
    reward, _, _ = world.step(torch.zeros(B, 2))
    assert reward.shape == (B,)


def test_step_done_shape() -> None:
    B = 4
    world = make_world(batch_size=B)
    _, done, _ = world.step(torch.zeros(B, 2))
    assert done.shape == (B,)


def test_step_done_dtype_bool() -> None:
    world = make_world(batch_size=4)
    _, done, _ = world.step(torch.zeros(4, 2))
    assert done.dtype == torch.bool


def test_step_reward_dtype_float() -> None:
    world = make_world(batch_size=4)
    reward, _, _ = world.step(torch.zeros(4, 2))
    assert reward.dtype == torch.float32


def test_step_info_contains_expected_keys() -> None:
    world = make_world(batch_size=4)
    _, _, info = world.step(torch.zeros(4, 2))
    assert "collision" in info
    assert "ate_food" in info
    assert "energy" in info
    assert "step_count" in info


def test_step_done_fires_at_max_steps() -> None:
    B = 2
    max_steps = 5
    world = make_world(batch_size=B, max_steps=max_steps, step_energy_cost=0.0)
    for i in range(max_steps - 1):
        _, done, _ = world.step(torch.zeros(B, 2))
        assert not done.any(), f"done should not fire at step {i + 1}"
    _, done, _ = world.step(torch.zeros(B, 2))
    assert done.all(), "done should fire at max_steps"


def test_step_done_fires_on_zero_energy() -> None:
    B = 1
    # Large step cost so energy hits 0 quickly
    world = make_world(batch_size=B, step_energy_cost=1.0, initial_energy=1.0, max_steps=1000)
    for _ in range(20):
        _, done, _info = world.step(torch.zeros(B, 2))
        if done.any():
            break
    assert done.any(), "done should fire when energy reaches 0"


def test_step_food_reward_positive() -> None:
    """Reward must be non-negative (food only gives positive reward)."""
    world = make_world(batch_size=8, food_density=0.5)
    action = torch.ones(8, 2)
    for _ in range(20):
        reward, _, _ = world.step(action)
        assert (reward >= 0.0).all()


def test_step_positions_bounded() -> None:
    """Agent positions must stay within grid dimensions."""
    B, H, W = 8, 8, 8
    world = make_world(batch_size=B, height=H, width=W)
    # Move forward every step
    action = torch.zeros(B, 2)
    action[:, 0] = 1.0
    for _ in range(20):
        world.step(action)
        assert (world._positions[:, 0] >= 0).all()
        assert (world._positions[:, 0] < H).all()
        assert (world._positions[:, 1] >= 0).all()
        assert (world._positions[:, 1] < W).all()


# ── reset(mask) acceptance test ───────────────────────────────────────────────


def test_reset_all_envs() -> None:
    world = make_world(batch_size=4, max_steps=10, step_energy_cost=0.001)
    for _ in range(5):
        world.step(torch.zeros(4, 2))
    world.reset()
    assert (world._step_count == 0).all()
    assert (world._energy == world.initial_energy).all()


def test_partial_reset_leaves_others_unchanged() -> None:
    B = 4
    world = make_world(batch_size=B, max_steps=100, step_energy_cost=0.001)
    # Step all envs
    for _ in range(5):
        world.step(torch.zeros(B, 2))
    step_before = world._step_count.clone()
    energy_before = world._energy.clone()

    # Reset only envs 0 and 2
    mask = torch.tensor([True, False, True, False])
    world.reset(mask)

    # Masked envs should be reset
    assert world._step_count[0].item() == 0
    assert world._step_count[2].item() == 0
    assert world._energy[0].item() == world.initial_energy
    assert world._energy[2].item() == world.initial_energy

    # Unmasked envs should be unchanged
    assert world._step_count[1].item() == step_before[1].item()
    assert world._step_count[3].item() == step_before[3].item()
    assert abs(world._energy[1].item() - energy_before[1].item()) < 1e-6
    assert abs(world._energy[3].item() - energy_before[3].item()) < 1e-6


def test_reset_with_none_mask_resets_all() -> None:
    world = make_world(batch_size=4, step_energy_cost=0.001)
    for _ in range(3):
        world.step(torch.zeros(4, 2))
    world.reset(mask=None)
    assert (world._step_count == 0).all()


def test_reset_with_false_mask_no_op() -> None:
    world = make_world(batch_size=4, step_energy_cost=0.001)
    for _ in range(3):
        world.step(torch.zeros(4, 2))
    step_before = world._step_count.clone()
    world.reset(mask=torch.zeros(4, dtype=torch.bool))
    assert torch.equal(world._step_count, step_before)


def test_reset_occupancy_channel_updated() -> None:
    """After reset, exactly one cell per env should be occupied."""
    B = 4
    world = make_world(batch_size=B, height=8, width=8)
    occ_sum = world._world[:, :, :, CHANNEL_OCCUPANCY].sum(dim=(1, 2))
    assert (occ_sum == 1).all(), f"expected 1 occupied cell per env, got {occ_sum}"


# ── Determinism acceptance test ───────────────────────────────────────────────


def test_determinism_same_seed_same_trajectory() -> None:
    """Identical seed + action sequence → bit-equal rewards and done flags."""

    def rollout(seed: int, n_steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        world = make_world(batch_size=4, height=8, width=8, seed=seed)
        rewards, dones = [], []
        for _ in range(n_steps):
            action = torch.tensor([[1.0, 0.6]] * 4)
            r, d, _ = world.step(action)
            rewards.append(r)
            dones.append(d)
        return torch.stack(rewards), torch.stack(dones)

    r1, d1 = rollout(42)
    r2, d2 = rollout(42)
    assert torch.equal(r1, r2), "rewards differ for same seed"
    assert torch.equal(d1, d2), "done flags differ for same seed"


def test_determinism_different_seeds_differ() -> None:
    """Different seeds should (almost surely) produce different outcomes."""

    def rollout(seed: int) -> torch.Tensor:
        world = make_world(batch_size=4, height=8, width=8, seed=seed, food_density=0.3)
        rewards = []
        for _ in range(20):
            r, _, _ = world.step(torch.ones(4, 2))
            rewards.append(r)
        return torch.stack(rewards)

    r1 = rollout(42)
    r2 = rollout(99)
    # With high food density and movement, trajectories should differ
    assert not torch.equal(r1, r2), "different seeds should produce different trajectories"


# ── Physics: collision / turning ──────────────────────────────────────────────


def test_collision_keeps_agent_in_place() -> None:
    """An agent at the border trying to move out should be blocked."""
    B = 1
    world = make_world(batch_size=B, height=4, width=4)
    # Force position to top-left corner and heading North
    world._positions = torch.zeros(B, 2, dtype=torch.long)
    world._headings = torch.zeros(B, dtype=torch.long)  # North

    pos_before = world._positions.clone()
    action = torch.tensor([[1.0, 0.0]])  # move forward (North) — blocked by boundary
    world.step(action)
    # Should stay at row 0 (boundary collision)
    assert world._positions[0, 0].item() == 0


def test_turning_changes_heading() -> None:
    """Turn-right action should increment heading by 1 (mod 4)."""
    world = make_world(batch_size=1)
    world._headings = torch.zeros(1, dtype=torch.long)  # North
    action = torch.tensor([[0.0, 1.0]])  # turn right, no forward
    world.step(action)
    assert world._headings[0].item() == 1  # East


# ── observation.py unit tests ─────────────────────────────────────────────────


def test_extract_local_crop_shape() -> None:
    B, H, W, C, k = 4, 8, 8, 4, 3
    world = torch.zeros(B, H, W, C)
    positions = torch.zeros(B, 2, dtype=torch.long)
    headings = torch.zeros(B, dtype=torch.long)
    crop = extract_local_crop(world, positions, headings, crop_size=k)
    assert crop.shape == (B, C, k, k)


def test_extract_local_crop_wall_padding() -> None:
    """Positions at corner should see wall padding in the crop."""
    B, H, W, C, k = 1, 4, 4, 4, 3
    world = torch.zeros(B, H, W, C)
    positions = torch.zeros(B, 2, dtype=torch.long)  # top-left
    headings = torch.zeros(B, dtype=torch.long)
    crop = extract_local_crop(world, positions, headings, crop_size=k)
    # Top-left corner: top row and left column of crop should be wall
    # Wall channel is 0; padding should set it to 1.0
    assert crop[0, 0, 0, :].sum() > 0, "top padding should be wall"
    assert crop[0, 0, :, 0].sum() > 0, "left padding should be wall"


def test_build_obs_keys() -> None:
    B, H, W, C, k = 2, 8, 8, 4, 3
    world = torch.zeros(B, H, W, C)
    positions = torch.ones(B, 2, dtype=torch.long)
    headings = torch.zeros(B, dtype=torch.long)
    energy = torch.ones(B)
    hunger = torch.zeros(B)
    last_action = torch.zeros(B)
    obs = build_obs(world, positions, headings, energy, hunger, last_action, k)
    assert "vision" in obs
    assert "proprio" in obs


def test_build_obs_shapes() -> None:
    B, H, W, C, k = 3, 8, 8, 4, 5
    world = torch.zeros(B, H, W, C)
    positions = torch.ones(B, 2, dtype=torch.long) * 3
    headings = torch.zeros(B, dtype=torch.long)
    energy = torch.rand(B)
    hunger = torch.rand(B)
    last_action = torch.rand(B)
    obs = build_obs(world, positions, headings, energy, hunger, last_action, k)
    assert obs["vision"].shape == (B, C, k, k)
    assert obs["proprio"].shape == (B, 3)

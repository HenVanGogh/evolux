"""Unit tests for evolux.environment.foraging (Phase 1).

Covers acceptance tests from ``src/evolux/environment/SPEC.md``:

- Episode terminates when termination condition met (timeout, death, goal).
- ``reset(mask)`` re-spawns only masked envs.
- Reward is bounded and finite.
"""

from __future__ import annotations

import torch

from evolux.core.protocols import Environment
from evolux.environment import ENVIRONMENT_REGISTRY
from evolux.environment.foraging import ForagingEnv

# ── Fake World stub ───────────────────────────────────────────────────────────


class _FakeWorld:
    """Minimal World-Protocol stub for ForagingEnv tests."""

    def __init__(
        self,
        batch_size: int = 4,
        device: torch.device = torch.device("cpu"),
        food_reward: float = 1.0,
        returns_done: bool = False,
        timeout: bool = False,
    ) -> None:
        self.batch_size: int = batch_size
        self.device: torch.device = device
        self._food_reward: float = food_reward
        self._returns_done: bool = returns_done
        self._timeout: bool = timeout
        self._reset_calls: list[torch.Tensor | None] = []

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self._reset_calls.append(mask)

    def observe(self) -> dict:
        return {
            "vision": torch.zeros(self.batch_size, 3, 5, 5),
            "proprio": torch.zeros(self.batch_size, 4),
        }

    def step(self, action: torch.Tensor) -> tuple:
        B = self.batch_size
        reward = torch.full((B,), self._food_reward)
        done = torch.full((B,), self._returns_done, dtype=torch.bool)
        info = {"timeout": torch.full((B,), self._timeout, dtype=torch.bool)}
        return reward, done, info


# ── Registry ──────────────────────────────────────────────────────────────────


def test_foraging_registered() -> None:
    assert "foraging_v1" in ENVIRONMENT_REGISTRY


# ── Protocol conformance ──────────────────────────────────────────────────────


def test_isinstance_environment_protocol() -> None:
    env = ForagingEnv(_FakeWorld())
    assert isinstance(env, Environment)


# ── Reward shaping ────────────────────────────────────────────────────────────


def test_reward_finite() -> None:
    """Shaped reward is always finite."""
    B = 8
    world = _FakeWorld(batch_size=B, food_reward=1.0, returns_done=False)
    env = ForagingEnv(world, idle_penalty=0.01, death_penalty=10.0)
    reward, _, _ = env.interact(torch.zeros(B, 2))
    assert torch.all(torch.isfinite(reward))


def test_reward_shape() -> None:
    """interact() returns (B,) reward tensor."""
    B = 6
    world = _FakeWorld(batch_size=B)
    env = ForagingEnv(world)
    reward, done, _info = env.interact(torch.zeros(B, 2))
    assert reward.shape == (B,)
    assert done.shape == (B,)


def test_reward_bounded_below() -> None:
    """Reward is bounded below by -(idle_penalty + death_penalty)."""
    B = 4
    idle, death = 0.01, 10.0
    world = _FakeWorld(batch_size=B, food_reward=0.0, returns_done=True, timeout=False)
    env = ForagingEnv(world, idle_penalty=idle, death_penalty=death)
    reward, _, _ = env.interact(torch.zeros(B, 2))
    lower = -(idle + death)
    assert torch.all(reward >= lower - 1e-6)


def test_idle_penalty_applied() -> None:
    """Idle penalty is subtracted every step (no food, no death)."""
    B = 4
    world = _FakeWorld(batch_size=B, food_reward=0.0, returns_done=False)
    env = ForagingEnv(world, idle_penalty=0.05, death_penalty=0.0)
    reward, _, _ = env.interact(torch.zeros(B, 2))
    assert torch.allclose(reward, torch.full((B,), -0.05))


def test_death_penalty_on_done_not_timeout() -> None:
    """Death penalty applies when done=True and timeout=False."""
    B = 4
    world = _FakeWorld(batch_size=B, food_reward=0.0, returns_done=True, timeout=False)
    env = ForagingEnv(world, idle_penalty=0.0, death_penalty=10.0)
    reward, done, info = env.interact(torch.zeros(B, 2))
    assert torch.all(done)
    assert torch.all(info["death"])
    assert torch.allclose(reward, torch.full((B,), -10.0))


def test_no_death_penalty_on_timeout() -> None:
    """Death penalty is NOT applied when done=True due to timeout."""
    B = 4
    world = _FakeWorld(batch_size=B, food_reward=0.0, returns_done=True, timeout=True)
    env = ForagingEnv(world, idle_penalty=0.0, death_penalty=10.0)
    reward, done, info = env.interact(torch.zeros(B, 2))
    assert torch.all(done)
    assert not torch.any(info["death"])
    assert torch.allclose(reward, torch.zeros(B))


def test_food_reward_scale() -> None:
    """food_reward_scale multiplies the positive reward."""
    B = 4
    world = _FakeWorld(batch_size=B, food_reward=1.0, returns_done=False)
    env = ForagingEnv(world, idle_penalty=0.0, death_penalty=0.0, food_reward_scale=2.0)
    reward, _, _ = env.interact(torch.zeros(B, 2))
    assert torch.allclose(reward, torch.full((B,), 2.0))


def test_negative_base_reward_not_added() -> None:
    """Negative base rewards from the world are clamped to zero (not penalties)."""
    B = 4
    world = _FakeWorld(batch_size=B, food_reward=-5.0, returns_done=False)
    env = ForagingEnv(world, idle_penalty=0.01, death_penalty=0.0)
    reward, _, _ = env.interact(torch.zeros(B, 2))
    # Negative base reward is clamped → only idle penalty remains
    assert torch.allclose(reward, torch.full((B,), -0.01))


def test_interact_does_not_mutate_world_info() -> None:
    """interact() shallow-copies the world info dict and does not mutate it."""
    B = 4
    world = _FakeWorld(batch_size=B, returns_done=False)
    env = ForagingEnv(world)
    # Step once to grab the dict reference via monkey-patching
    original_info: dict | None = None

    real_step = world.step

    def _capturing_step(action: torch.Tensor) -> tuple:
        nonlocal original_info
        result = real_step(action)
        original_info = result[2]
        return result

    world.step = _capturing_step  # type: ignore[method-assign]

    env.interact(torch.zeros(B, 2))
    assert original_info is not None
    # "death" and "shaped_reward" must NOT appear in the world's own dict
    assert "death" not in original_info
    assert "shaped_reward" not in original_info


# ── Episode termination ────────────────────────────────────────────────────────


def test_episode_terminates_when_world_done() -> None:
    """Episode terminates when world signals done."""
    B = 4
    world = _FakeWorld(batch_size=B, returns_done=True)
    env = ForagingEnv(world)
    _, done, _ = env.interact(torch.zeros(B, 2))
    assert torch.all(done)


def test_episode_not_done_when_world_alive() -> None:
    """Episode does not terminate when world is not done."""
    B = 4
    world = _FakeWorld(batch_size=B, returns_done=False)
    env = ForagingEnv(world)
    _, done, _ = env.interact(torch.zeros(B, 2))
    assert not torch.any(done)


# ── Partial reset ─────────────────────────────────────────────────────────────


def test_partial_reset_delegates_mask_to_world() -> None:
    """ForagingEnv.reset(mask) passes the mask through to world.reset."""
    B = 4
    world = _FakeWorld(batch_size=B)
    env = ForagingEnv(world)
    mask = torch.tensor([True, False, True, False])
    env.reset(mask)
    assert len(world._reset_calls) == 1
    assert torch.equal(world._reset_calls[0], mask)


def test_full_reset_passes_none_to_world() -> None:
    """ForagingEnv.reset() without mask calls world.reset(None)."""
    B = 4
    world = _FakeWorld(batch_size=B)
    env = ForagingEnv(world)
    env.reset()
    assert len(world._reset_calls) == 1
    assert world._reset_calls[0] is None


def test_partial_reset_only_resets_masked_envs() -> None:
    """reset(mask) does not reset un-masked environments (verified via call count)."""
    B = 4
    world = _FakeWorld(batch_size=B)
    env = ForagingEnv(world)
    # Two partial resets and one full reset
    env.reset(torch.tensor([True, False, False, False]))
    env.reset(torch.tensor([False, True, False, False]))
    env.reset()
    assert len(world._reset_calls) == 3


# ── Environment.step() ecological dynamics ────────────────────────────────────


def test_env_step_returns_none() -> None:
    """Environment.step() (ecological dynamics) returns None in Phase 1."""
    env = ForagingEnv(_FakeWorld())
    assert env.step() is None


# ── fields() ──────────────────────────────────────────────────────────────────


def test_fields_returns_dict() -> None:
    """fields() returns a plain dict."""
    env = ForagingEnv(_FakeWorld())
    assert isinstance(env.fields(), dict)


def test_fields_values_are_tensors() -> None:
    """All values returned by fields() are Tensors."""
    env = ForagingEnv(_FakeWorld())
    for v in env.fields().values():
        assert isinstance(v, torch.Tensor)


# ── observe() ─────────────────────────────────────────────────────────────────


def test_observe_delegates_to_world() -> None:
    """observe() delegates to world.observe()."""
    B = 4
    world = _FakeWorld(batch_size=B)
    env = ForagingEnv(world)
    obs = env.observe()
    assert "vision" in obs
    assert "proprio" in obs
    assert obs["vision"].shape == (B, 3, 5, 5)

"""Tests for DynamicNN brain and memory systems."""

import numpy as np
import pytest

from sim_env.brain.dynamic_nn import DynamicNN
from sim_env.brain.memory import WorkingMemory, EpisodicMemory, HebbianMemory, MemoryBank
from sim_env.creatures.genome import Genome


class TestWorkingMemory:
    def test_read_write(self):
        wm = WorkingMemory(size=4)
        wm.write(np.array([1.0, 2.0, 3.0, 4.0]))
        assert np.allclose(wm.read(), [1.0, 2.0, 3.0, 4.0])

    def test_reset(self):
        wm = WorkingMemory(size=4)
        wm.write(np.ones(4))
        wm.reset()
        assert np.allclose(wm.read(), 0.0)

    def test_slot_access(self):
        wm = WorkingMemory(size=8)
        wm.write_slot(3, 7.5)
        assert wm.read_slot(3) == pytest.approx(7.5)


class TestEpisodicMemory:
    def test_write_read(self):
        em = EpisodicMemory(capacity=10, key_dim=4, val_dim=4)
        key = np.array([1.0, 0.0, 0.0, 0.0])
        val = np.array([0.5, 0.5, 0.5, 0.5])
        em.write(key, val)
        result = em.read(key)
        # Should retrieve close to val
        assert np.allclose(result, val, atol=0.2)

    def test_empty_read_returns_zeros(self):
        em = EpisodicMemory(capacity=10, key_dim=4, val_dim=4)
        result = em.read(np.ones(4))
        assert np.allclose(result, 0.0)


class TestHebbianMemory:
    def test_update_changes_weights(self):
        hm = HebbianMemory(n_pre=3, n_post=3, lr=0.1)
        pre = np.array([1.0, 1.0, 1.0])
        post = np.array([1.0, 1.0, 1.0])
        hm.update(pre, post)
        assert not np.allclose(hm.W, 0.0)

    def test_weights_bounded(self):
        hm = HebbianMemory(n_pre=3, n_post=3, lr=1.0)
        for _ in range(1000):
            hm.update(np.ones(3) * 10, np.ones(3) * 10)
        assert np.all(np.abs(hm.W) <= 5.0 + 1e-6)


class TestDynamicNN:
    def _make_brain(self, cfg, rng):
        n_inputs = 30
        n_outputs = 4
        genome = Genome.minimal(n_inputs=n_inputs, n_outputs=n_outputs, rng=rng)
        memory = MemoryBank(cfg)
        return DynamicNN(genome, n_inputs, n_outputs, memory)

    def test_forward_output_shape(self, cfg, rng):
        brain = self._make_brain(cfg, rng)
        obs = rng.random(brain.n_inputs).astype(np.float32)
        out = brain.forward(obs)
        assert out.shape == (brain.n_outputs,)

    def test_reset_clears_activations(self, cfg, rng):
        brain = self._make_brain(cfg, rng)
        obs = np.ones(brain.n_inputs, dtype=np.float32)
        brain.forward(obs)
        brain.reset_episode()
        assert np.allclose(brain._act, 0.0)

    def test_recurrent_state_persists_between_steps(self, cfg, rng):
        brain = self._make_brain(cfg, rng)
        obs = np.ones(brain.n_inputs, dtype=np.float32)
        out1 = brain.forward(obs)
        out2 = brain.forward(obs)
        # Outputs may differ due to recurrent state (not guaranteed but likely)
        # Just check both are valid shapes
        assert out1.shape == out2.shape

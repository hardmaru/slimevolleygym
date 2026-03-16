"""Tests for functional step/reset pipeline (Phase 5-6)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing, set_backend
from slime_volleyball.core.step_pipeline import (
    build_reset_fn,
    build_step_fn,
    reset,
    step,
)


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


class TestNumpyPipeline:
    def test_reset_shapes(self):
        obs, state, _ = reset(np.random.default_rng(42))
        assert obs.shape == (2, 12)
        assert state.ball_pos.shape == (2,)
        assert state.agent_pos.shape == (2, 2)
        assert int(state.time) == 0
        assert int(state.delay_life) == 30

    def test_step_shapes(self):
        obs, state, _ = reset(np.random.default_rng(42))
        actions = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
        obs2, state2, rewards, terms, truncs, _ = step(state, actions)
        assert obs2.shape == (2, 12)
        assert rewards.shape == (2,)
        assert terms.shape == (2,)
        assert truncs.shape == (2,)
        assert int(state2.time) == 1

    def test_delay_screen_counts_down(self):
        _, state, _ = reset(np.random.default_rng(0))
        assert int(state.delay_life) == 30
        actions = np.zeros((2, 3), dtype=np.float32)
        for _ in range(30):
            _, state, _, _, _, _ = step(state, actions)
        assert int(state.delay_life) == 0

    def test_ball_frozen_during_delay(self):
        _, state, _ = reset(np.random.default_rng(0))
        init_ball_pos = state.ball_pos.copy()
        actions = np.zeros((2, 3), dtype=np.float32)
        _, state2, _, _, _, _ = step(state, actions)
        np.testing.assert_array_equal(state2.ball_pos, init_ball_pos)

    def test_episode_terminates_on_goal(self):
        _, state, _ = reset(np.random.default_rng(42))
        actions = np.zeros((2, 3), dtype=np.float32)
        terminated = False
        for _ in range(3000):
            _, state, rewards, terms, truncs, _ = step(state, actions)
            if bool(terms[0]) or bool(truncs[0]):
                terminated = True
                break
        assert terminated

    def test_truncation_at_max_steps(self):
        _, state, _ = reset(np.random.default_rng(42), max_steps=10)
        actions = np.zeros((2, 3), dtype=np.float32)
        for _ in range(10):
            _, state, _, terms, truncs, _ = step(state, actions)
        assert bool(truncs[0])

    def test_boost_mode_reset(self):
        obs, state, _ = reset(np.random.default_rng(42), boost=True)
        assert obs.shape == (2, 16)
        assert state.boost is True

    def test_boost_mode_step(self):
        obs, state, _ = reset(np.random.default_rng(42), boost=True)
        actions = np.array([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
        obs2, state2, _, _, _, _ = step(state, actions)
        assert obs2.shape == (2, 16)
        assert int(state2.agent_powerup_timer[0]) == 89  # activated & decremented

    def test_reward_signs(self):
        """Verify reward signs: positive for right when right scores."""
        _, state, _ = reset(np.random.default_rng(42))
        actions = np.zeros((2, 3), dtype=np.float32)
        total_reward_right = 0.0
        for _ in range(3000):
            _, state, rewards, terms, truncs, _ = step(state, actions)
            total_reward_right += float(rewards[1])
            if bool(terms[0]):
                break
        # Reward and life tracking should be consistent
        if total_reward_right > 0:
            assert int(state.agent_life[0]) <= 0
        elif total_reward_right < 0:
            assert int(state.agent_life[1]) <= 0


class TestJaxPipeline:
    def test_reset_shapes(self):
        set_backend("jax")
        import jax

        obs, state, _ = reset(jax.random.PRNGKey(42))
        assert obs.shape == (2, 12)
        assert state.ball_pos.shape == (2,)

    def test_step_shapes(self):
        set_backend("jax")
        import jax
        import jax.numpy as jnp

        obs, state, _ = reset(jax.random.PRNGKey(42))
        actions = jnp.array([[0, 0, 0], [0, 0, 1]], dtype=jnp.float32)
        obs2, state2, rewards, terms, truncs, _ = step(state, actions)
        assert obs2.shape == (2, 12)
        assert rewards.shape == (2,)

    def test_jit_step(self):
        set_backend("jax")
        import jax
        import jax.numpy as jnp

        step_fn = build_step_fn()
        reset_fn = build_reset_fn()

        obs, state, _ = reset_fn(jax.random.PRNGKey(0))
        actions = jnp.zeros((2, 3), dtype=jnp.float32)
        obs2, state2, rewards, terms, truncs, _ = step_fn(state, actions)
        assert obs2.shape == (2, 12)

    def test_vmap(self):
        set_backend("jax")
        import jax
        import jax.numpy as jnp

        reset_fn = build_reset_fn()
        step_fn = build_step_fn()

        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        obs_batch, state_batch, _ = jax.vmap(reset_fn)(keys)
        assert obs_batch.shape == (8, 2, 12)

        actions = jnp.zeros((8, 2, 3), dtype=jnp.float32)
        obs_b, state_b, rew_b, _, _, _ = jax.vmap(step_fn)(state_batch, actions)
        assert obs_b.shape == (8, 2, 12)
        assert rew_b.shape == (8, 2)

    def test_lax_scan(self):
        set_backend("jax")
        import jax
        import jax.numpy as jnp

        reset_fn = build_reset_fn()
        step_fn = build_step_fn()

        obs, state, _ = reset_fn(jax.random.PRNGKey(0))

        def scan_step(state, _):
            actions = jnp.zeros((2, 3), dtype=jnp.float32)
            obs, new_state, rewards, _, _, _ = step_fn(state, actions)
            return new_state, rewards

        final_state, all_rewards = jax.lax.scan(scan_step, state, None, length=50)
        assert all_rewards.shape == (50, 2)
        assert int(final_state.time) == 50

    def test_boost_vmap(self):
        set_backend("jax")
        import jax
        import jax.numpy as jnp

        reset_fn = build_reset_fn(boost=True)
        step_fn = build_step_fn()

        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs_batch, state_batch, _ = jax.vmap(reset_fn)(keys)
        assert obs_batch.shape == (4, 2, 16)

        actions = jnp.zeros((4, 2, 4), dtype=jnp.float32)
        obs_b, _, _, _, _, _ = jax.vmap(step_fn)(state_batch, actions)
        assert obs_b.shape == (4, 2, 16)

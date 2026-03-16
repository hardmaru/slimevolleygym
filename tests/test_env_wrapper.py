"""Tests for the environment wrapper (Phase 7-8)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


class TestNumpyEnv:
    def test_reset(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv()
        obs, info = env.reset(seed=42)
        assert "agent_left" in obs
        assert "agent_right" in obs
        assert obs["agent_left"]["obs"].shape == (12,)
        assert obs["agent_right"]["obs"].shape == (12,)

    def test_step_multiagent(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv()
        env.reset(seed=42)
        actions = {"agent_left": 0, "agent_right": 3}
        obs, rewards, terms, truncs, info = env.step(actions)
        assert "agent_left" in rewards
        assert "agent_right" in rewards
        assert "__all__" in terms

    def test_step_single_agent(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv()
        env.reset(seed=42)
        obs, rewards, terms, truncs, info = env.step(3)  # right agent jumps
        assert "agent_left" in rewards

    def test_episode_completion(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv()
        env.reset(seed=42)
        done = False
        for _ in range(3001):
            obs, rewards, terms, truncs, info = env.step(
                {"agent_left": 0, "agent_right": 0}
            )
            if terms.get("__all__") or truncs.get("__all__"):
                done = True
                break
        assert done

    def test_survival_reward(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv(config={"survival_reward": True})
        env.reset(seed=42)
        _, rewards, _, _, _ = env.step({"agent_left": 0, "agent_right": 0})
        # Survival reward adds 0.01 per step
        assert rewards["agent_right"] >= 0.01 - 1e-6


class TestJaxEnv:
    def test_reset(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv(config={"backend": "jax"})
        obs, info = env.reset(seed=42)
        assert obs["agent_left"]["obs"].shape == (12,)

    def test_step(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv(config={"backend": "jax"})
        env.reset(seed=42)
        obs, rewards, terms, truncs, info = env.step(
            {"agent_left": 0, "agent_right": 3}
        )
        assert isinstance(rewards["agent_right"], float)

    def test_jax_step_property(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv(config={"backend": "jax"})
        env.reset(seed=42)
        step_fn = env.jax_step
        assert callable(step_fn)

    def test_jax_reset_property(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv(config={"backend": "jax"})
        env.reset(seed=42)
        reset_fn = env.jax_reset
        assert callable(reset_fn)

    def test_jax_step_not_available_on_numpy(self):
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv

        env = SlimeVolleyEnv()
        env.reset(seed=42)
        with pytest.raises(RuntimeError, match="only available with backend='jax'"):
            _ = env.jax_step

    def test_vmap_training_loop(self):
        """Simulate a JaxMARL-style training loop."""
        from slime_volleyball.slimevolley_env import SlimeVolleyEnv
        import jax
        import jax.numpy as jnp

        env = SlimeVolleyEnv(config={"backend": "jax"})
        env.reset(seed=0)

        step_fn = env.jax_step
        reset_fn = env.jax_reset
        num_envs = 4

        # Vectorized reset
        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
        obs, states, _ = jax.vmap(reset_fn)(keys)
        assert obs.shape == (num_envs, 2, 12)

        # Vectorized step
        actions = jnp.zeros((num_envs, 2, 3), dtype=jnp.float32)
        obs, states, rewards, terms, truncs, _ = jax.vmap(step_fn)(states, actions)
        assert obs.shape == (num_envs, 2, 12)
        assert rewards.shape == (num_envs, 2)

        # lax.scan rollout (extract single state from batch)
        single_state = jax.tree_util.tree_map(lambda x: x[0], states)

        def rollout_step(state, _):
            actions = jnp.zeros((2, 3), dtype=jnp.float32)
            obs, ns, rew, _, _, _ = step_fn(state, actions)
            return ns, rew

        final, all_rew = jax.lax.scan(rollout_step, single_state, None, length=100)
        assert all_rew.shape == (100, 2)


class TestBoostEnv:
    def test_numpy_boost(self):
        _reset_backend_for_testing()
        from slime_volleyball.slimevolley_boost_env import SlimeVolleyBoostEnv

        env = SlimeVolleyBoostEnv()
        obs, _ = env.reset(seed=42)
        assert obs["agent_left"]["obs"].shape == (16,)

        obs, _, _, _, _ = env.step({"agent_left": 6, "agent_right": 0})
        assert env._env_state.agent_powerup_timer[0] > 0

    def test_jax_boost(self):
        _reset_backend_for_testing()
        from slime_volleyball.slimevolley_boost_env import SlimeVolleyBoostEnv

        env = SlimeVolleyBoostEnv(config={"backend": "jax"})
        obs, _ = env.reset(seed=42)
        assert obs["agent_left"]["obs"].shape == (16,)

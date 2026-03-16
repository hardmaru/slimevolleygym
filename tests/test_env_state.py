"""Tests for EnvState dataclass (Phase 2)."""

import dataclasses

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing, set_backend
from slime_volleyball.backend.env_state import EnvState, create_env_state


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


def _make_state(**overrides):
    defaults = dict(
        ball_pos=np.zeros(2, dtype=np.float32),
        ball_vel=np.zeros(2, dtype=np.float32),
        ball_prev_pos=np.zeros(2, dtype=np.float32),
        agent_pos=np.zeros((2, 2), dtype=np.float32),
        agent_vel=np.zeros((2, 2), dtype=np.float32),
        agent_desired_vel=np.zeros((2, 2), dtype=np.float32),
        agent_life=np.array([1, 1], dtype=np.int32),
        agent_powerup_avail=np.array([1, 1], dtype=np.int32),
        agent_powerup_timer=np.array([0, 0], dtype=np.int32),
        delay_life=np.int32(30),
        time=np.int32(0),
        done=np.bool_(False),
        rng_key=None,
    )
    defaults.update(overrides)
    return create_env_state(**defaults)


def test_create_numpy():
    state = _make_state()
    assert state.max_steps == 3000
    assert state.max_lives == 1
    assert state.boost is False
    np.testing.assert_array_equal(state.ball_pos, [0, 0])


def test_frozen():
    state = _make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.time = 5


def test_replace():
    state = _make_state()
    new_state = dataclasses.replace(state, time=np.int32(10))
    assert int(new_state.time) == 10
    assert int(state.time) == 0  # original unchanged


def test_create_jax():
    set_backend("jax")
    import jax
    import jax.numpy as jnp

    state = create_env_state(
        ball_pos=jnp.zeros(2, dtype=jnp.float32),
        ball_vel=jnp.zeros(2, dtype=jnp.float32),
        ball_prev_pos=jnp.zeros(2, dtype=jnp.float32),
        agent_pos=jnp.zeros((2, 2), dtype=jnp.float32),
        agent_vel=jnp.zeros((2, 2), dtype=jnp.float32),
        agent_desired_vel=jnp.zeros((2, 2), dtype=jnp.float32),
        agent_life=jnp.array([1, 1], dtype=jnp.int32),
        agent_powerup_avail=jnp.array([1, 1], dtype=jnp.int32),
        agent_powerup_timer=jnp.array([0, 0], dtype=jnp.int32),
        delay_life=jnp.int32(30),
        time=jnp.int32(0),
        done=jnp.bool_(False),
        rng_key=jax.random.PRNGKey(0),
    )
    assert isinstance(state.ball_pos, jnp.ndarray)

    # Verify pytree compatibility
    leaves = jax.tree_util.tree_leaves(state)
    assert len(leaves) > 0

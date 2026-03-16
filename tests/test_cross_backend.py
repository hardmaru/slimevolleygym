"""Cross-backend parity tests.

Verify that numpy and JAX backends produce structurally consistent results
when given the same scripted actions. Note: exact values may differ due to
RNG differences (PCG64 vs ThreeFry), but shapes, signs, and terminal
conditions should match.
"""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


def test_reset_shapes_match():
    """Both backends produce same observation shapes."""
    _reset_backend_for_testing()
    from slime_volleyball.core.step_pipeline import reset

    obs_np, state_np, _ = reset(np.random.default_rng(42))

    _reset_backend_for_testing()
    from slime_volleyball.backend import set_backend

    set_backend("jax")
    import jax

    obs_jax, state_jax, _ = reset(jax.random.PRNGKey(42))

    assert obs_np.shape == obs_jax.shape
    assert state_np.ball_pos.shape == state_jax.ball_pos.shape
    assert state_np.agent_pos.shape == state_jax.agent_pos.shape


def test_step_shapes_match():
    """Both backends produce same output shapes after stepping."""
    _reset_backend_for_testing()
    from slime_volleyball.core.step_pipeline import reset, step

    obs_np, state_np, _ = reset(np.random.default_rng(42))
    actions = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.float32)
    obs2_np, state2_np, rew_np, term_np, trunc_np, _ = step(state_np, actions)

    _reset_backend_for_testing()
    from slime_volleyball.backend import set_backend

    set_backend("jax")
    import jax
    import jax.numpy as jnp

    obs_jax, state_jax, _ = reset(jax.random.PRNGKey(42))
    actions_jax = jnp.array([[0, 0, 1], [1, 0, 0]], dtype=jnp.float32)
    obs2_jax, state2_jax, rew_jax, term_jax, trunc_jax, _ = step(state_jax, actions_jax)

    assert obs2_np.shape == obs2_jax.shape
    assert rew_np.shape == rew_jax.shape
    assert term_np.shape == term_jax.shape
    assert trunc_np.shape == trunc_jax.shape


def test_initial_state_structure_match():
    """Initial state has same structure on both backends."""
    _reset_backend_for_testing()
    from slime_volleyball.core.step_pipeline import reset

    _, state_np, _ = reset(np.random.default_rng(0))

    _reset_backend_for_testing()
    from slime_volleyball.backend import set_backend

    set_backend("jax")
    import jax

    _, state_jax, _ = reset(jax.random.PRNGKey(0))

    # Same initial agent positions (these are deterministic)
    np.testing.assert_array_almost_equal(
        np.array(state_np.agent_pos), np.array(state_jax.agent_pos), decimal=5
    )

    # Same initial delay
    assert int(state_np.delay_life) == int(state_jax.delay_life)

    # Same initial time
    assert int(state_np.time) == int(state_jax.time)

    # Same initial lives
    np.testing.assert_array_equal(
        np.array(state_np.agent_life), np.array(state_jax.agent_life)
    )


def test_deterministic_physics():
    """Scripted actions produce same physics trajectory (ignoring RNG for ball init).

    Both backends start with same ball position and zero-action steps,
    so physics should be identical during the delay period (ball frozen).
    """
    _reset_backend_for_testing()
    from slime_volleyball.core.step_pipeline import reset, step

    _, state_np, _ = reset(np.random.default_rng(0))
    actions = np.zeros((2, 3), dtype=np.float32)

    # Step through delay period (ball frozen, deterministic)
    for _ in range(5):
        _, state_np, _, _, _, _ = step(state_np, actions)

    _reset_backend_for_testing()
    from slime_volleyball.backend import set_backend

    set_backend("jax")
    import jax
    import jax.numpy as jnp

    _, state_jax, _ = reset(jax.random.PRNGKey(0))
    actions_jax = jnp.zeros((2, 3), dtype=jnp.float32)

    for _ in range(5):
        _, state_jax, _, _, _, _ = step(state_jax, actions_jax)

    # Agent positions should be identical (gravity, no ball interaction)
    np.testing.assert_array_almost_equal(
        np.array(state_np.agent_pos), np.array(state_jax.agent_pos), decimal=4
    )

    # Ball positions should be identical (frozen during delay)
    # Note: ball initial velocities differ due to RNG, but positions stay at init
    # since ball is frozen during delay
    np.testing.assert_array_almost_equal(
        np.array(state_np.ball_pos), np.array(state_jax.ball_pos), decimal=4
    )


def test_boost_shapes_match():
    """Boost mode produces same shapes on both backends."""
    _reset_backend_for_testing()
    from slime_volleyball.core.step_pipeline import reset, step

    obs_np, state_np, _ = reset(np.random.default_rng(0), boost=True)
    assert obs_np.shape == (2, 16)

    _reset_backend_for_testing()
    from slime_volleyball.backend import set_backend

    set_backend("jax")
    import jax

    obs_jax, state_jax, _ = reset(jax.random.PRNGKey(0), boost=True)
    assert obs_jax.shape == (2, 16)
    assert obs_np.shape == obs_jax.shape

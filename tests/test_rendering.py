"""Tests for rendering (Phase 12)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


def test_legacy_game_creation():
    from slime_volleyball.slimevolley_env import SlimeVolleyEnv

    env = SlimeVolleyEnv()
    env.reset(seed=42)
    env._build_legacy_game()
    assert env._game is not None


def test_state_sync():
    from slime_volleyball.slimevolley_env import SlimeVolleyEnv

    env = SlimeVolleyEnv()
    env.reset(seed=42)

    for _ in range(10):
        env.step({"agent_left": 1, "agent_right": 3})

    env._build_legacy_game()
    env._update_legacy_game_from_state()

    s = env._env_state
    g = env._game
    assert abs(float(s.ball_pos[0]) - g.ball.x) < 1e-5
    assert abs(float(s.ball_pos[1]) - g.ball.y) < 1e-5
    assert abs(float(s.agent_pos[0, 0]) - g.agent_left.x) < 1e-5
    assert abs(float(s.agent_pos[1, 0]) - g.agent_right.x) < 1e-5


def test_render_raises_on_jax():
    from slime_volleyball.slimevolley_env import SlimeVolleyEnv

    env = SlimeVolleyEnv(config={"backend": "jax"})
    env.reset(seed=42)
    with pytest.raises(NotImplementedError, match="Rendering is only available"):
        env.render()

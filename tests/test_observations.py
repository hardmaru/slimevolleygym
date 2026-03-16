"""Tests for observation computation (Phase 5)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing
from slime_volleyball.core.observations import SCALE_FACTOR, compute_observations


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


def test_obs_shape():
    obs = compute_observations(
        agent_pos=np.array([[-12, 1.5], [12, 1.5]], dtype=np.float32),
        agent_vel=np.zeros((2, 2), dtype=np.float32),
        ball_pos=np.array([0, 12], dtype=np.float32),
        ball_vel=np.zeros(2, dtype=np.float32),
        agent_powerup_avail=np.array([1, 1], dtype=np.int32),
        agent_powerup_timer=np.array([0, 0], dtype=np.int32),
    )
    assert obs.shape == (2, 12)


def test_obs_boost_shape():
    obs = compute_observations(
        agent_pos=np.array([[-12, 1.5], [12, 1.5]], dtype=np.float32),
        agent_vel=np.zeros((2, 2), dtype=np.float32),
        ball_pos=np.array([0, 12], dtype=np.float32),
        ball_vel=np.zeros(2, dtype=np.float32),
        agent_powerup_avail=np.array([1, 1], dtype=np.int32),
        agent_powerup_timer=np.array([0, 0], dtype=np.int32),
        boost=True,
    )
    assert obs.shape == (2, 16)


def test_perspective_symmetry():
    """Left and right agents should see symmetric observations (x flipped)."""
    agent_pos = np.array([[-12, 1.5], [12, 1.5]], dtype=np.float32)
    agent_vel = np.zeros((2, 2), dtype=np.float32)
    ball_pos = np.array([0, 12], dtype=np.float32)
    ball_vel = np.zeros(2, dtype=np.float32)
    pa = np.array([1, 1], dtype=np.int32)
    pt = np.array([0, 0], dtype=np.int32)

    obs = compute_observations(agent_pos, agent_vel, ball_pos, ball_vel, pa, pt)

    # Both agents have x=12/10 (their own pos in normalized frame)
    assert obs[0, 0] == pytest.approx(12.0 / SCALE_FACTOR)  # left: -(-12) = 12
    assert obs[1, 0] == pytest.approx(12.0 / SCALE_FACTOR)  # right: 12

    # Ball at x=0: both see 0 for ball x
    assert obs[0, 4] == pytest.approx(0.0)
    assert obs[1, 4] == pytest.approx(0.0)

    # Opponent: left agent sees right at ox = 12 * (-(-1)) = 12
    # Right agent sees left at ox = -12 * (-(1)) = 12
    # Both see opponent at positive x (opponent is always "in front")
    assert obs[0, 8] == pytest.approx(12.0 / SCALE_FACTOR)  # opponent x
    assert obs[1, 8] == pytest.approx(12.0 / SCALE_FACTOR)  # opponent x


def test_obs_scaled():
    """Observations should be divided by SCALE_FACTOR."""
    agent_pos = np.array([[-12, 1.5], [12, 1.5]], dtype=np.float32)
    obs = compute_observations(
        agent_pos, np.zeros((2, 2), dtype=np.float32),
        np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32),
        np.array([1, 1], dtype=np.int32), np.array([0, 0], dtype=np.int32),
    )
    # y should be 1.5 / 10.0 = 0.15
    assert obs[0, 1] == pytest.approx(0.15)
    assert obs[1, 1] == pytest.approx(0.15)

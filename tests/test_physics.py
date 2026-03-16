"""Tests for functional physics (Phase 3-4)."""

import numpy as np
import pytest

from slime_volleyball.backend._dispatch import _reset_backend_for_testing
from slime_volleyball.core.physics import (
    AGENT_RADIUS,
    BALL_RADIUS,
    actions_to_desired_vel,
    ball_apply_acceleration_and_move,
    ball_check_collision,
    ball_check_edges,
    process_powerup_action,
    update_agent,
    update_all_agents,
)
from slime_volleyball.core.constants import PLAYER_SPEED_X, PLAYER_SPEED_Y, REF_U, REF_W


@pytest.fixture(autouse=True)
def reset_backend():
    _reset_backend_for_testing()
    yield
    _reset_backend_for_testing()


class TestActionsToDesiredVel:
    def test_noop(self):
        actions = np.array([[0, 0, 0]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == 0.0
        assert dv[0, 1] == 0.0

    def test_forward(self):
        actions = np.array([[1, 0, 0]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == -PLAYER_SPEED_X
        assert dv[0, 1] == 0.0

    def test_backward(self):
        actions = np.array([[0, 1, 0]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == PLAYER_SPEED_X
        assert dv[0, 1] == 0.0

    def test_jump(self):
        actions = np.array([[0, 0, 1]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == 0.0
        assert dv[0, 1] == PLAYER_SPEED_Y

    def test_forward_and_backward_cancel(self):
        actions = np.array([[1, 1, 0]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == 0.0

    def test_two_agents(self):
        actions = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        dv = actions_to_desired_vel(actions)
        assert dv[0, 0] == -PLAYER_SPEED_X  # left forward
        assert dv[0, 1] == PLAYER_SPEED_Y   # left jump
        assert dv[1, 0] == PLAYER_SPEED_X   # right backward
        assert dv[1, 1] == 0.0


class TestUpdateAgent:
    def test_agent_on_ground_stays(self):
        pos = np.array([12.0, REF_U], dtype=np.float32)
        vel = np.zeros(2, dtype=np.float32)
        desired = np.zeros(2, dtype=np.float32)
        timer = np.int32(0)
        new_pos, new_vel, new_timer = update_agent(pos, vel, desired, timer, np.float32(1.0))
        assert new_pos[1] >= REF_U

    def test_agent_jumps(self):
        pos = np.array([12.0, REF_U], dtype=np.float32)
        vel = np.zeros(2, dtype=np.float32)
        desired = np.array([0.0, PLAYER_SPEED_Y], dtype=np.float32)
        timer = np.int32(0)
        new_pos, new_vel, new_timer = update_agent(pos, vel, desired, timer, np.float32(1.0))
        assert new_pos[1] > REF_U

    def test_agent_clamped_to_own_half(self):
        # Right agent (dir=1) can't go past fence
        pos = np.array([1.0, REF_U], dtype=np.float32)  # too close to center
        vel = np.zeros(2, dtype=np.float32)
        desired = np.array([-PLAYER_SPEED_X, 0.0], dtype=np.float32)
        timer = np.int32(0)
        new_pos, new_vel, _ = update_agent(pos, vel, desired, timer, np.float32(1.0))
        min_x_right = 1.0 * (0.5 + AGENT_RADIUS)
        assert new_pos[0] >= min_x_right - 0.01

    def test_powerup_timer_decrements(self):
        pos = np.array([12.0, REF_U], dtype=np.float32)
        vel = np.zeros(2, dtype=np.float32)
        desired = np.zeros(2, dtype=np.float32)
        timer = np.int32(5)
        _, _, new_timer = update_agent(pos, vel, desired, timer, np.float32(1.0))
        assert int(new_timer) == 4


class TestBallPhysics:
    def test_ball_frozen_during_delay(self):
        pos = np.array([0.0, 12.0], dtype=np.float32)
        vel = np.array([5.0, 10.0], dtype=np.float32)
        new_pos, new_vel = ball_apply_acceleration_and_move(pos, vel, True)
        np.testing.assert_array_equal(new_pos, pos)
        np.testing.assert_array_equal(new_vel, vel)

    def test_ball_moves_after_delay(self):
        pos = np.array([0.0, 12.0], dtype=np.float32)
        vel = np.array([5.0, 10.0], dtype=np.float32)
        new_pos, new_vel = ball_apply_acceleration_and_move(pos, vel, False)
        assert new_pos[0] != pos[0]  # moved horizontally
        assert new_vel[1] < vel[1]   # gravity pulled down

    def test_ball_collision_detection(self):
        ball_pos = np.array([3.0, 5.0], dtype=np.float32)
        ball_vel = np.array([-5.0, 0.0], dtype=np.float32)
        agent_pos = np.array([2.0, 5.0], dtype=np.float32)
        agent_vel = np.zeros(2, dtype=np.float32)
        new_pos, new_vel, collided = ball_check_collision(
            ball_pos, ball_vel, agent_pos, AGENT_RADIUS, agent_vel, np.float32(1.0)
        )
        assert collided  # ball and agent overlap

    def test_ball_no_collision_when_far(self):
        ball_pos = np.array([20.0, 20.0], dtype=np.float32)
        ball_vel = np.array([5.0, 5.0], dtype=np.float32)
        agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        agent_vel = np.zeros(2, dtype=np.float32)
        new_pos, new_vel, collided = ball_check_collision(
            ball_pos, ball_vel, agent_pos, AGENT_RADIUS, agent_vel, np.float32(1.0)
        )
        assert not collided
        np.testing.assert_array_equal(new_vel, ball_vel)


class TestBallEdges:
    def test_left_ground_goal(self):
        """Ball hitting left ground: right scores."""
        pos = np.array([-10.0, 1.5], dtype=np.float32)  # below ground+radius
        vel = np.array([0.0, -5.0], dtype=np.float32)
        prev = np.array([-10.0, 5.0], dtype=np.float32)
        _, _, goal = ball_check_edges(pos, vel, prev)
        assert float(goal) > 0  # right scored

    def test_right_ground_goal(self):
        """Ball hitting right ground: left scores."""
        pos = np.array([10.0, 1.5], dtype=np.float32)
        vel = np.array([0.0, -5.0], dtype=np.float32)
        prev = np.array([10.0, 5.0], dtype=np.float32)
        _, _, goal = ball_check_edges(pos, vel, prev)
        assert float(goal) < 0  # left scored

    def test_no_goal_in_air(self):
        pos = np.array([0.0, 12.0], dtype=np.float32)
        vel = np.array([5.0, 5.0], dtype=np.float32)
        prev = np.array([0.0, 12.0], dtype=np.float32)
        _, _, goal = ball_check_edges(pos, vel, prev)
        assert float(goal) == 0.0

    def test_ceiling_bounce(self):
        pos = np.array([0.0, 47.9], dtype=np.float32)  # near ceiling (REF_H=48)
        vel = np.array([0.0, 10.0], dtype=np.float32)
        prev = np.array([0.0, 47.0], dtype=np.float32)
        new_pos, new_vel, _ = ball_check_edges(pos, vel, prev)
        assert new_vel[1] < 0  # bounced down


class TestPowerup:
    def test_powerup_activation(self):
        actions = np.array([[0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float32)
        avail = np.array([1, 1], dtype=np.int32)
        timer = np.array([0, 0], dtype=np.int32)
        new_avail, new_timer = process_powerup_action(actions, avail, timer)
        assert int(new_avail[0]) == 0  # consumed
        assert int(new_timer[0]) == 90  # activated
        assert int(new_avail[1]) == 1  # unchanged
        assert int(new_timer[1]) == 0  # unchanged

    def test_no_powerup_when_unavailable(self):
        actions = np.array([[0, 0, 0, 1]], dtype=np.float32)
        avail = np.array([0], dtype=np.int32)
        timer = np.array([0], dtype=np.int32)
        new_avail, new_timer = process_powerup_action(actions, avail, timer)
        assert int(new_timer[0]) == 0  # not activated

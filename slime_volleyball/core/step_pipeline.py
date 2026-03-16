"""Functional step and reset pipeline for Slime Volleyball.

Pure functions that operate on EnvState. On the JAX backend these are
JIT-compiled; on numpy they run eagerly. No mutable state — all
transitions produce new EnvState via dataclasses.replace.
"""

import dataclasses

from slime_volleyball.backend import xp
from slime_volleyball.backend._dispatch import get_backend
from slime_volleyball.backend.env_state import create_env_state
from slime_volleyball.core.constants import (
    INIT_DELAY_FRAMES,
    REF_W,
    MAXLIVES,
)
from slime_volleyball.core.physics import (
    AGENT_RADIUS,
    BALL_RADIUS,
    BALL_INIT_Y,
    AGENT_LEFT_INIT_X,
    AGENT_RIGHT_INIT_X,
    AGENT_INIT_Y,
    FENCE_STUB_X,
    FENCE_STUB_Y,
    FENCE_STUB_R,
    actions_to_desired_vel,
    process_powerup_action,
    update_all_agents,
    ball_apply_acceleration_and_move,
    ball_check_collision,
    ball_check_edges,
    limit_ball_speed,
)
from slime_volleyball.core.observations import compute_observations


# ---------------------------------------------------------------------------
# RNG helpers
# ---------------------------------------------------------------------------

def _backend_rng_ball_vel(rng_key):
    """Generate random ball initial velocity.

    Returns (new_key, vx, vy).
    vx ~ Uniform(-20, 20), vy ~ Uniform(10, 25).
    """
    if get_backend() == "jax":
        import jax
        key, k1, k2 = jax.random.split(rng_key, 3)
        vx = jax.random.uniform(k1, (), minval=-20.0, maxval=20.0)
        vy = jax.random.uniform(k2, (), minval=10.0, maxval=25.0)
        return key, vx, vy
    else:
        import numpy as np
        if isinstance(rng_key, np.random.Generator):
            rng = rng_key
        else:
            rng = np.random.default_rng(rng_key if rng_key is not None else None)
        vx = np.float32(rng.uniform(-20.0, 20.0))
        vy = np.float32(rng.uniform(10.0, 25.0))
        return rng, vx, vy


def _maybe_stop_gradient(*arrays):
    """Apply jax.lax.stop_gradient on JAX; identity on numpy."""
    if get_backend() == "jax":
        import jax.lax as lax
        return tuple(lax.stop_gradient(a) for a in arrays)
    return arrays


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def reset(rng_key, max_steps=3000, max_lives=1, boost=False):
    """Create initial EnvState and observations.

    Args:
        rng_key: JAX PRNGKey or int seed (numpy) or np.random.Generator
        max_steps: max timesteps before truncation
        max_lives: lives per agent
        boost: enable powerup actions

    Returns: (obs, state, infos)
    """
    key, ball_vx, ball_vy = _backend_rng_ball_vel(rng_key)

    ball_pos = xp.array([0.0, BALL_INIT_Y], dtype=xp.float32)
    ball_vel = xp.array([ball_vx, ball_vy], dtype=xp.float32)
    ball_prev_pos = xp.array([0.0, BALL_INIT_Y], dtype=xp.float32)

    agent_pos = xp.array([
        [AGENT_LEFT_INIT_X, AGENT_INIT_Y],
        [AGENT_RIGHT_INIT_X, AGENT_INIT_Y],
    ], dtype=xp.float32)
    agent_vel = xp.zeros((2, 2), dtype=xp.float32)
    agent_desired_vel = xp.zeros((2, 2), dtype=xp.float32)

    agent_life = xp.array([max_lives, max_lives], dtype=xp.int32)
    agent_powerup_avail = xp.array([1, 1], dtype=xp.int32)
    agent_powerup_timer = xp.array([0, 0], dtype=xp.int32)

    delay_life = xp.int32(INIT_DELAY_FRAMES)
    time = xp.int32(0)
    done = xp.bool_(False)

    state = create_env_state(
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        ball_prev_pos=ball_prev_pos,
        agent_pos=agent_pos,
        agent_vel=agent_vel,
        agent_desired_vel=agent_desired_vel,
        agent_life=agent_life,
        agent_powerup_avail=agent_powerup_avail,
        agent_powerup_timer=agent_powerup_timer,
        delay_life=delay_life,
        time=time,
        done=done,
        rng_key=key,
        max_steps=max_steps,
        max_lives=max_lives,
        boost=boost,
    )

    obs = compute_observations(
        agent_pos, agent_vel, ball_pos, ball_vel,
        agent_powerup_avail, agent_powerup_timer,
        boost=boost,
    )

    (obs,) = _maybe_stop_gradient(obs)

    return obs, state, {}


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

def step(state, actions):
    """Pure functional step.

    Args:
        state: EnvState
        actions: (2, 3) or (2, 4) float32 array
                 Per agent: [forward, backward, jump, (powerup)]

    Returns: (obs, new_state, rewards, terminateds, truncateds, infos)
        obs: (2, obs_dim) float32
        rewards: (2,) float32 -- [left_reward, right_reward]
        terminateds: (2,) bool
        truncateds: (2,) bool
    """
    boost = state.boost

    # 1. Compute desired velocities from actions
    desired_vel = actions_to_desired_vel(actions)

    # 1b. Process powerup actions (if boost mode)
    powerup_avail = state.agent_powerup_avail
    powerup_timer = state.agent_powerup_timer
    if boost:
        # Pad actions to 4 columns if needed (handled by caller, but be safe)
        powerup_avail, powerup_timer = process_powerup_action(
            actions, powerup_avail, powerup_timer
        )

    # 2. Update agents
    new_agent_pos, new_agent_vel, new_powerup_timer = update_all_agents(
        state.agent_pos, state.agent_vel, desired_vel, powerup_timer
    )

    # 3. Delay screen
    delay_active = state.delay_life > 0
    new_delay = xp.where(delay_active, state.delay_life - 1, state.delay_life)

    # 4. Ball physics (frozen during delay)
    prev_ball_pos = state.ball_pos
    new_ball_pos, new_ball_vel = ball_apply_acceleration_and_move(
        state.ball_pos, state.ball_vel, delay_active
    )

    # 5. Ball-agent collisions
    # Left agent collision
    left_power_factor = xp.float32(1.0) + xp.float32(0.5) * (new_powerup_timer[0] > 0).astype(xp.float32)
    left_agent_vel = xp.array([xp.float32(0.0), xp.float32(0.0)])  # agents are treated as static in bounces
    new_ball_pos, new_ball_vel, _ = ball_check_collision(
        new_ball_pos, new_ball_vel,
        new_agent_pos[0], AGENT_RADIUS, left_agent_vel, left_power_factor
    )

    # Right agent collision
    right_power_factor = xp.float32(1.0) + xp.float32(0.5) * (new_powerup_timer[1] > 0).astype(xp.float32)
    new_ball_pos, new_ball_vel, _ = ball_check_collision(
        new_ball_pos, new_ball_vel,
        new_agent_pos[1], AGENT_RADIUS, left_agent_vel, right_power_factor
    )

    # 6. Ball-fence stub collision
    fence_pos = xp.array([FENCE_STUB_X, FENCE_STUB_Y], dtype=xp.float32)
    fence_vel = xp.array([0.0, 0.0], dtype=xp.float32)
    new_ball_pos, new_ball_vel, _ = ball_check_collision(
        new_ball_pos, new_ball_vel,
        fence_pos, FENCE_STUB_R, fence_vel, xp.float32(1.0)
    )

    # 7. Ball edge checks (walls, ceiling, fence, ground/goals)
    new_ball_pos, new_ball_vel, goal_result = ball_check_edges(
        new_ball_pos, new_ball_vel, prev_ball_pos
    )

    # 8. Handle goal scored: update lives, possibly start new match
    scored = goal_result != 0
    # goal_result > 0 means right scored (left loses life)
    # goal_result < 0 means left scored (right loses life)
    left_lost = (goal_result > 0).astype(xp.int32)
    right_lost = (goal_result < 0).astype(xp.int32)

    new_lives = xp.array([
        state.agent_life[0] - left_lost,
        state.agent_life[1] - right_lost,
    ], dtype=xp.int32)

    # 9. New match on goal (reset ball, delay)
    key, new_ball_vx, new_ball_vy = _backend_rng_ball_vel(state.rng_key)

    match_ball_pos = xp.array([0.0, BALL_INIT_Y], dtype=xp.float32)
    match_ball_vel = xp.array([new_ball_vx, new_ball_vy], dtype=xp.float32)

    # Only reset ball if scored AND max_lives > 1 (multi-match mode)
    do_new_match = scored & (state.max_lives > 1)
    final_ball_pos = xp.where(do_new_match, match_ball_pos, new_ball_pos)
    final_ball_vel = xp.where(do_new_match, match_ball_vel, new_ball_vel)
    final_delay = xp.where(
        do_new_match, xp.int32(INIT_DELAY_FRAMES),
        xp.where(delay_active, new_delay, state.delay_life)
    )
    # Update prev_pos: if new match, reset; otherwise use pre-physics pos
    final_ball_prev = xp.where(do_new_match, match_ball_pos, prev_ball_pos)

    # 10. Build new state
    new_time = state.time + 1
    any_dead = (new_lives[0] <= 0) | (new_lives[1] <= 0)
    truncated = new_time >= state.max_steps
    new_done = any_dead | truncated

    new_state = dataclasses.replace(
        state,
        ball_pos=final_ball_pos,
        ball_vel=final_ball_vel,
        ball_prev_pos=final_ball_prev,
        agent_pos=new_agent_pos,
        agent_vel=new_agent_vel,
        agent_desired_vel=desired_vel,
        agent_life=new_lives,
        agent_powerup_avail=powerup_avail,
        agent_powerup_timer=new_powerup_timer,
        delay_life=final_delay,
        time=new_time,
        done=new_done,
        rng_key=key,
    )

    # 11. Observations
    obs = compute_observations(
        new_agent_pos, new_agent_vel, final_ball_pos, final_ball_vel,
        powerup_avail, new_powerup_timer,
        boost=boost,
    )

    # 12. Rewards
    # goal_result > 0 means right scored, < 0 means left scored
    reward_right = goal_result
    rewards = xp.array([-reward_right, reward_right], dtype=xp.float32)

    terminateds = xp.array([any_dead, any_dead], dtype=xp.bool_)
    truncateds = xp.array([truncated, truncated], dtype=xp.bool_)

    obs, rewards, terminateds, truncateds = _maybe_stop_gradient(
        obs, rewards, terminateds, truncateds
    )

    return obs, new_state, rewards, terminateds, truncateds, {}


# ---------------------------------------------------------------------------
# Factory builders
# ---------------------------------------------------------------------------

def build_step_fn(jit_compile=None):
    """Build a step function closure, optionally JIT-compiled."""
    def step_fn(state, actions):
        return step(state, actions)
    return _maybe_jit(step_fn, jit_compile)


def build_reset_fn(max_steps=3000, max_lives=1, boost=False, jit_compile=None):
    """Build a reset function closure, optionally JIT-compiled."""
    def reset_fn(rng_key):
        return reset(rng_key, max_steps=max_steps, max_lives=max_lives, boost=boost)
    return _maybe_jit(reset_fn, jit_compile)


def _maybe_jit(fn, jit_compile=None):
    """Auto-JIT on JAX backend unless explicitly disabled."""
    if get_backend() == "jax" and jit_compile is not False:
        import jax
        return jax.jit(fn)
    return fn

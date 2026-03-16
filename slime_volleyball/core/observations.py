"""Pure functional observation computation for Slime Volleyball.

Replaces the OOP RelativeState pattern with array operations.
"""

from slime_volleyball.backend import xp

SCALE_FACTOR = 10.0


def compute_observations(agent_pos, agent_vel, ball_pos, ball_vel,
                         agent_powerup_avail, agent_powerup_timer, boost=False):
    """Compute observations for both agents.

    Each agent sees from its own perspective (coordinates normalized by direction).
    Agent 0 (left, dir=-1), Agent 1 (right, dir=1).

    Base observation (12-dim):
        [x, y, vx, vy, bx, by, bvx, bvy, ox, oy, ovx, ovy]

    With boost=True (16-dim), appends:
        [powerups_available, powered_up_timer, opp_powerups_available, opp_powered_up_timer]

    Returns: (2, obs_dim) float32 array
    """
    dirs = xp.array([-1.0, 1.0], dtype=xp.float32)

    obs_list = []
    for i in range(2):
        d = dirs[i]
        opp = 1 - i
        obs_i = xp.array([
            agent_pos[i, 0] * d,
            agent_pos[i, 1],
            agent_vel[i, 0] * d,
            agent_vel[i, 1],
            ball_pos[0] * d,
            ball_pos[1],
            ball_vel[0] * d,
            ball_vel[1],
            agent_pos[opp, 0] * (-d),
            agent_pos[opp, 1],
            agent_vel[opp, 0] * (-d),
            agent_vel[opp, 1],
        ], dtype=xp.float32) / SCALE_FACTOR

        if boost:
            powerup_obs = xp.array([
                agent_powerup_avail[i],
                agent_powerup_timer[i],
                agent_powerup_avail[opp],
                agent_powerup_timer[opp],
            ], dtype=xp.float32) / SCALE_FACTOR
            obs_i = xp.concatenate([obs_i, powerup_obs])

        obs_list.append(obs_i)

    return xp.stack(obs_list, axis=0)

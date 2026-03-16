"""Pure functional physics for Slime Volleyball.

All functions operate on arrays and use xp for backend-agnostic computation.
No mutable state — returns new arrays instead of modifying in place.
All conditionals on dynamic values use xp.where for JAX traceability.
"""

from slime_volleyball.backend import xp
from slime_volleyball.core.constants import (
    GRAVITY,
    TIMESTEP,
    NUDGE,
    FRICTION,
    PLAYER_SPEED_X,
    PLAYER_SPEED_Y,
    MAX_BALL_SPEED,
    REF_W,
    REF_H,
    REF_U,
    REF_WALL_WIDTH,
    REF_WALL_HEIGHT,
)

# Derived constants
AGENT_RADIUS = 1.5
BALL_RADIUS = 0.5
FENCE_STUB_X = 0.0
FENCE_STUB_Y = REF_WALL_HEIGHT
FENCE_STUB_R = REF_WALL_WIDTH / 2
BALL_INIT_Y = REF_W / 4  # = 12.0
AGENT_LEFT_INIT_X = -REF_W / 4  # = -12.0
AGENT_RIGHT_INIT_X = REF_W / 4  # = 12.0
AGENT_INIT_Y = 1.5
AGENT_DIRS = [-1.0, 1.0]


def actions_to_desired_vel(actions):
    """Convert (2, 3) or (2, 4) action array to (2, 2) desired velocities.

    actions[i] = [forward, backward, jump, (optional powerup)]
    forward/backward are booleans (or floats > 0 = True).
    Returns desired_vx, desired_vy per agent (before dir multiplication).
    """
    forward = actions[:, 0] > 0
    backward = actions[:, 1] > 0
    jump = actions[:, 2] > 0

    # forward and not backward -> desired_vx = -PLAYER_SPEED_X
    # backward and not forward -> desired_vx = PLAYER_SPEED_X
    # both or neither -> 0
    fwd_only = forward & (~backward)
    bwd_only = backward & (~forward)

    desired_vx = xp.where(
        fwd_only,
        xp.float32(-PLAYER_SPEED_X),
        xp.where(bwd_only, xp.float32(PLAYER_SPEED_X), xp.float32(0.0)),
    )
    desired_vy = xp.where(jump, xp.float32(PLAYER_SPEED_Y), xp.float32(0.0))

    return xp.stack([desired_vx, desired_vy], axis=-1)


def process_powerup_action(actions, powerup_avail, powerup_timer):
    """Process powerup activation from actions.

    actions: (2, 4) array with powerup in column 3
    Returns new (powerup_avail, powerup_timer) arrays.
    """
    has_powerup_action = actions[:, 3] > 0
    can_activate = (powerup_avail > 0) & has_powerup_action

    new_timer = xp.where(can_activate, xp.int32(90), powerup_timer)
    new_avail = xp.where(can_activate, powerup_avail - 1, powerup_avail)

    return new_avail, new_timer


def update_agent(agent_pos, agent_vel, desired_vel, powerup_timer, agent_dir):
    """Update a single agent's physics. Returns (new_pos, new_vel, new_timer).

    Replicates Agent.update() from agent.py exactly:
    1. Apply gravity to vy
    2. If on ground, set vy = desired_vy
    3. Set vx = desired_vx * dir
    4. Decrement powerup timer
    5. Move (x += vx*dt, y += vy*dt)
    6. Clamp y to ground
    7. Clamp x to own half
    """
    dt = TIMESTEP

    # 1. Gravity
    new_vy = agent_vel[1] + GRAVITY * dt

    # 2. On ground: apply desired_vy
    on_ground = agent_pos[1] <= REF_U + NUDGE * dt
    new_vy = xp.where(on_ground, desired_vel[1], new_vy)

    # 3. Set vx from desired (already multiplied by dir outside or here)
    new_vx = desired_vel[0] * agent_dir

    # 4. Powerup timer
    new_timer = xp.where(powerup_timer > 0, powerup_timer - 1, powerup_timer)

    # 5. Move
    new_x = agent_pos[0] + new_vx * dt
    new_y = agent_pos[1] + new_vy * dt

    # 6. Clamp y to ground
    below_ground = new_y <= REF_U
    new_y = xp.where(below_ground, xp.float32(REF_U), new_y)
    new_vy = xp.where(below_ground, xp.float32(0.0), new_vy)

    # 7. Clamp x to own half
    # Agent must stay on its side: x * dir >= wall_width/2 + radius
    # and x * dir <= ref_w/2 - radius
    min_x = agent_dir * (REF_WALL_WIDTH / 2 + AGENT_RADIUS)
    max_x = agent_dir * (REF_W / 2 - AGENT_RADIUS)

    # For left agent (dir=-1): min_x=-22.5, max_x=-2.0 (inverted)
    # For right agent (dir=1): min_x=2.0, max_x=22.5
    # Use actual min/max to handle both sides
    lo = xp.minimum(min_x, max_x)
    hi = xp.maximum(min_x, max_x)

    too_close_to_fence = new_x * agent_dir <= (REF_WALL_WIDTH / 2 + AGENT_RADIUS)
    at_fence = too_close_to_fence
    new_x = xp.where(at_fence, agent_dir * (REF_WALL_WIDTH / 2 + AGENT_RADIUS), new_x)
    new_vx = xp.where(at_fence, xp.float32(0.0), new_vx)

    too_far = new_x * agent_dir >= (REF_W / 2 - AGENT_RADIUS)
    new_x = xp.where(too_far, agent_dir * (REF_W / 2 - AGENT_RADIUS), new_x)
    new_vx = xp.where(too_far, xp.float32(0.0), new_vx)

    new_pos = xp.array([new_x, new_y])
    new_vel = xp.array([new_vx, new_vy])

    return new_pos, new_vel, new_timer


def update_all_agents(agent_pos, agent_vel, desired_vel, powerup_timer):
    """Update both agents. Returns (new_pos, new_vel, new_timer).

    agent_pos: (2, 2), agent_vel: (2, 2), desired_vel: (2, 2), powerup_timer: (2,)
    """
    pos0, vel0, t0 = update_agent(
        agent_pos[0], agent_vel[0], desired_vel[0], powerup_timer[0], xp.float32(AGENT_DIRS[0])
    )
    pos1, vel1, t1 = update_agent(
        agent_pos[1], agent_vel[1], desired_vel[1], powerup_timer[1], xp.float32(AGENT_DIRS[1])
    )

    new_pos = xp.stack([pos0, pos1], axis=0)
    new_vel = xp.stack([vel0, vel1], axis=0)
    new_timer = xp.array([t0, t1])

    return new_pos, new_vel, new_timer


def ball_apply_acceleration_and_move(ball_pos, ball_vel, delay_active):
    """Apply gravity to ball and move it. Ball is frozen during delay.

    Replicates:
        ball.apply_acceleration(0, GRAVITY)
        ball.limit_speed(0, MAX_BALL_SPEED)
        ball.move()
    """
    dt = TIMESTEP

    # Apply gravity (only when not delayed)
    new_vy = xp.where(delay_active, ball_vel[1], ball_vel[1] + GRAVITY * dt)
    new_vx = ball_vel[0]  # no x acceleration

    # Limit speed (only when not delayed)
    mag2 = new_vx * new_vx + new_vy * new_vy
    mag = xp.sqrt(xp.maximum(mag2, xp.float32(1e-12)))
    # speed_factor is 1 at this point (set during bounce), we need to track it
    # Actually MAX_BALL_SPEED is the cap. speed_factor is applied during bounce.
    # For limit_speed: maxSpeed *= self.speed_factor, but speed_factor is set
    # in bounce and used here. We'll pass speed_factor separately.
    too_fast = mag2 > (MAX_BALL_SPEED * MAX_BALL_SPEED)
    scale = MAX_BALL_SPEED / mag
    new_vx = xp.where(delay_active, ball_vel[0], xp.where(too_fast, new_vx * scale, new_vx))
    new_vy = xp.where(delay_active, ball_vel[1], xp.where(too_fast, new_vy * scale, new_vy))

    # Move (only when not delayed)
    new_x = xp.where(delay_active, ball_pos[0], ball_pos[0] + new_vx * dt)
    new_y = xp.where(delay_active, ball_pos[1], ball_pos[1] + new_vy * dt)

    return xp.array([new_x, new_y]), xp.array([new_vx, new_vy])


def ball_check_collision(ball_pos, ball_vel, obj_pos, obj_r, obj_vel, factor):
    """Check and resolve circle-circle collision between ball and object.

    Replicates Particle.bounce() from objects.py:
    1. Check if colliding (distance < sum of radii)
    2. Push ball out of overlap along collision normal
    3. Compute elastic bounce velocity with factor multiplier

    Returns (new_ball_pos, new_ball_vel, collided).
    """
    dx = ball_pos[0] - obj_pos[0]
    dy = ball_pos[1] - obj_pos[1]
    dist2 = dx * dx + dy * dy
    total_r = BALL_RADIUS + obj_r
    collided = dist2 < (total_r * total_r)

    # Normalize collision direction
    dist = xp.sqrt(xp.maximum(dist2, xp.float32(1e-12)))
    nx = dx / dist
    ny = dy / dist

    # Push ball out of overlap (replicate the while loop with single push)
    # Original code nudges by NUDGE per iteration; we compute exact separation
    overlap = total_r - dist
    push_x = nx * (overlap + NUDGE)
    push_y = ny * (overlap + NUDGE)
    new_bx = xp.where(collided, ball_pos[0] + push_x, ball_pos[0])
    new_by = xp.where(collided, ball_pos[1] + push_y, ball_pos[1])

    # Elastic bounce (relative velocity along normal)
    ux = ball_vel[0] - obj_vel[0]
    uy = ball_vel[1] - obj_vel[1]
    un = ux * nx + uy * ny
    unx = nx * (un * 2.0)
    uny = ny * (un * 2.0)

    new_vx = xp.where(collided, (ux - unx + obj_vel[0]) * factor, ball_vel[0])
    new_vy = xp.where(collided, (uy - uny + obj_vel[1]) * factor, ball_vel[1])

    return xp.array([new_bx, new_by]), xp.array([new_vx, new_vy]), collided


def ball_check_edges(ball_pos, ball_vel, ball_prev_pos):
    """Check ball boundary collisions and goal scoring.

    Replicates Particle.check_edges() from objects.py exactly:
    - Left/right wall bounce
    - Ceiling bounce
    - Fence wall collision (using prev_pos for crossing detection)
    - Ground collision = goal scored

    Returns (new_pos, new_vel, goal_result).
    goal_result: -1 if left loses, +1 if right loses, 0 if no goal.
    """
    dt = TIMESTEP
    r = BALL_RADIUS
    bx = ball_pos[0]
    by = ball_pos[1]
    vx = ball_vel[0]
    vy = ball_vel[1]
    prev_x = ball_prev_pos[0]

    # Left wall bounce
    hit_left = bx <= (r - REF_W / 2)
    vx = xp.where(hit_left, vx * (-FRICTION), vx)
    bx = xp.where(hit_left, r - REF_W / 2 + NUDGE * dt, bx)

    # Right wall bounce
    hit_right = bx >= (REF_W / 2 - r)
    vx = xp.where(hit_right, vx * (-FRICTION), vx)
    bx = xp.where(hit_right, REF_W / 2 - r - NUDGE * dt, bx)

    # Ground collision (goal detection)
    hit_ground = by <= (r + REF_U)
    vy_ground = xp.where(hit_ground, vy * (-FRICTION), vy)
    by_ground = xp.where(hit_ground, r + REF_U + NUDGE * dt, by)

    # Goal: ball hit ground on left side -> right scores (+1)
    # Ball hit ground on right side -> left scores (-1)
    # Convention from original: check_edges returns -1 if x<=0 (left side), +1 if x>0
    # Then game.step negates it: result = -ball.check_edges()
    # So check_edges: left ground = -1, right ground = +1
    # We return goal_result directly as the negated value (from right agent perspective)
    # i.e. goal_result > 0 means right agent scored, < 0 means left scored
    goal_result = xp.where(
        hit_ground,
        xp.where(bx <= 0, xp.float32(1.0), xp.float32(-1.0)),
        xp.float32(0.0),
    )

    vy = vy_ground
    by = by_ground

    # Ceiling bounce
    hit_ceiling = by >= (REF_H - r)
    vy = xp.where(hit_ceiling, vy * (-FRICTION), vy)
    by = xp.where(hit_ceiling, REF_H - r - NUDGE * dt, by)

    # Fence collision (ball crossing from right to left side of fence)
    fence_half = REF_WALL_WIDTH / 2
    # Crossing right side of fence going left
    cross_right = (
        (bx <= (fence_half + r))
        & (prev_x > (fence_half + r))
        & (by <= REF_WALL_HEIGHT)
    )
    vx = xp.where(cross_right, vx * (-FRICTION), vx)
    bx = xp.where(cross_right, fence_half + r + NUDGE * dt, bx)

    # Crossing left side of fence going right
    cross_left = (
        (bx >= (-fence_half - r))
        & (prev_x < (-fence_half - r))
        & (by <= REF_WALL_HEIGHT)
    )
    vx = xp.where(cross_left, vx * (-FRICTION), vx)
    bx = xp.where(cross_left, -fence_half - r - NUDGE * dt, bx)

    new_pos = xp.array([bx, by])
    new_vel = xp.array([vx, vy])

    return new_pos, new_vel, goal_result


def limit_ball_speed(ball_vel, speed_factor):
    """Limit ball speed, accounting for speed_factor from bounce.

    Replicates Particle.limit_speed(0, MAX_BALL_SPEED).
    """
    max_speed = MAX_BALL_SPEED * speed_factor
    mag2 = ball_vel[0] ** 2 + ball_vel[1] ** 2
    mag = xp.sqrt(xp.maximum(mag2, xp.float32(1e-12)))

    too_fast = mag2 > (max_speed * max_speed)
    scale = max_speed / mag
    new_vx = xp.where(too_fast, ball_vel[0] * scale, ball_vel[0])
    new_vy = xp.where(too_fast, ball_vel[1] * scale, ball_vel[1])

    return xp.array([new_vx, new_vy])

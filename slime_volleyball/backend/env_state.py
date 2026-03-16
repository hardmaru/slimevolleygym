"""Immutable environment state container for Slime Volleyball.

Defines EnvState, a frozen dataclass that bundles all environment state
into a single object. On numpy it is a plain data container; on JAX it
is registered as a pytree for jit/vmap/scan compatibility.

State transitions use dataclasses.replace:
    new_state = dataclasses.replace(state, ball_pos=new_pos)
"""

from dataclasses import dataclass, field

_pytree_registered: bool = False


@dataclass(frozen=True)
class EnvState:
    """Immutable slime volleyball environment state.

    Dynamic fields (traced through JIT):
        ball_pos:             (2,) float32 -- [x, y]
        ball_vel:             (2,) float32 -- [vx, vy]
        ball_prev_pos:        (2,) float32 -- previous [x, y] for fence collision
        agent_pos:            (2, 2) float32 -- [[x,y], [x,y]] for left, right
        agent_vel:            (2, 2) float32 -- [[vx,vy], [vx,vy]]
        agent_desired_vel:    (2, 2) float32 -- desired velocities from actions
        agent_life:           (2,) int32 -- lives remaining
        agent_powerup_avail:  (2,) int32 -- powerups available
        agent_powerup_timer:  (2,) int32 -- powerup countdown timer
        delay_life:           () int32 -- DelayScreen frames remaining
        time:                 () int32 -- timestep counter
        done:                 () bool -- episode done flag
        rng_key:              JAX PRNGKey or numpy Generator

    Static fields (compile-time constants, not traced):
        max_steps:  int -- max timesteps before truncation
        max_lives:  int -- lives per agent
        boost:      bool -- whether boost/powerup actions are enabled
    """

    # Dynamic fields
    ball_pos: object           # (2,) float32
    ball_vel: object           # (2,) float32
    ball_prev_pos: object      # (2,) float32
    agent_pos: object          # (2, 2) float32
    agent_vel: object          # (2, 2) float32
    agent_desired_vel: object  # (2, 2) float32
    agent_life: object         # (2,) int32
    agent_powerup_avail: object   # (2,) int32
    agent_powerup_timer: object   # (2,) int32
    delay_life: object         # () int32
    time: object               # () int32
    done: object               # () bool
    rng_key: object            # JAX PRNGKey or None

    # Static fields
    max_steps: int = field(metadata=dict(static=True), default=3000)
    max_lives: int = field(metadata=dict(static=True), default=1)
    boost: bool = field(metadata=dict(static=True), default=False)


def register_envstate_pytree() -> None:
    """Register EnvState as a JAX pytree node (idempotent)."""
    global _pytree_registered
    if _pytree_registered:
        return

    import jax.tree_util

    jax.tree_util.register_dataclass(EnvState)
    _pytree_registered = True


def create_env_state(**kwargs) -> EnvState:
    """Create an EnvState, ensuring JAX pytree registration if needed."""
    from slime_volleyball.backend import get_backend

    if get_backend() == "jax":
        register_envstate_pytree()

    return EnvState(**kwargs)

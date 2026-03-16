# Slime Volleyball

A Gymnasium-compatible multi-agent environment for the classic Slime Volleyball game, with dual numpy/JAX backends. Originally developed by [David Ha](https://otoro.net/slimevolley) and ported from [hardmaru/slimevolleygym](https://github.com/hardmaru/slimevolleygym).

Two agents (`agent_left`, `agent_right`) play volleyball. The game ends when one agent loses all lives (default: 1) or after `max_steps` (default: 3000) timesteps.

## Installation

```bash
pip install -e .
```

**Dependencies:** `numpy`, `gymnasium`

**Optional:** `jax` / `jaxlib` (JAX backend), `pyglet` (rendering), `opencv-python` (pixel observations)

## Quick Start

### Multi-Agent (PettingZoo-style)

```python
from slime_volleyball.slimevolley_env import SlimeVolleyEnv

env = SlimeVolleyEnv()
obs, info = env.reset(seed=42)

for _ in range(1000):
    actions = {"agent_left": env.action_space["agent_left"].sample(),
               "agent_right": env.action_space["agent_right"].sample()}
    obs, rewards, terminateds, truncateds, info = env.step(actions)
    if terminateds["__all__"] or truncateds["__all__"]:
        obs, info = env.reset()
```

### Single-Agent (Right Agent vs Baseline)

When you pass a single integer action, the right agent is controlled by that action and the left agent uses the built-in baseline RNN policy:

```python
env = SlimeVolleyEnv()
obs, info = env.reset(seed=42)

obs, rewards, terminateds, truncateds, info = env.step(3)  # right agent jumps
```

### Boost Mode

```python
from slime_volleyball.slimevolley_boost_env import SlimeVolleyBoostEnv

env = SlimeVolleyBoostEnv()
obs, info = env.reset(seed=42)
# 16-dim observations, Discrete(13) actions (adds boost/powerup variants)
```

## Dual Backend

The environment supports two backends selected at construction time. When `backend="numpy"` (default), no JAX code runs and JAX does not need to be installed. When `backend="jax"`, the environment uses `jax.numpy` for all array operations and exposes JIT-compiled functions for high-performance training.

```python
# NumPy backend (default) — no JAX required
env = SlimeVolleyEnv()

# JAX backend
env = SlimeVolleyEnv(config={"backend": "jax"})
```

Internally, array operations go through a backend proxy (`from slime_volleyball.backend import xp`) that resolves to `numpy` or `jax.numpy` depending on the active backend. The entire game physics and observation pipeline is implemented as pure functions operating on an immutable `EnvState` dataclass.

## JAX Functional API

With `backend="jax"`, the environment exposes `jax_step` and `jax_reset` — JIT-compiled pure functions compatible with `jax.vmap` and `jax.lax.scan` for vectorized training (JaxMARL-style).

```python
import jax
import jax.numpy as jnp
from slime_volleyball.slimevolley_env import SlimeVolleyEnv

env = SlimeVolleyEnv(config={"backend": "jax"})
env.reset(seed=0)

step_fn = env.jax_step    # (EnvState, actions) -> (obs, state, rewards, terms, truncs, infos)
reset_fn = env.jax_reset   # (rng_key) -> (obs, state, infos)
```

### Vectorized Environments (vmap)

```python
num_envs = 64
keys = jax.random.split(jax.random.PRNGKey(0), num_envs)

# Parallel reset
obs, states, _ = jax.vmap(reset_fn)(keys)
# obs.shape == (64, 2, 12), one obs per agent per env

# Parallel step
actions = jnp.zeros((num_envs, 2, 3), dtype=jnp.float32)
obs, states, rewards, terms, truncs, _ = jax.vmap(step_fn)(states, actions)
```

### Rollouts with lax.scan

```python
def rollout_step(state, _):
    actions = jnp.zeros((2, 3), dtype=jnp.float32)
    obs, new_state, rewards, _, _, _ = step_fn(state, actions)
    return new_state, rewards

single_state = jax.tree_util.tree_map(lambda x: x[0], states)
final_state, all_rewards = jax.lax.scan(rollout_step, single_state, None, length=1000)
# all_rewards.shape == (1000, 2)
```

## Observations

Observations are perspective-normalized: each agent sees the world as if it's on a consistent side (x-values and velocities are reflected by the agent's direction).

**Base (12-dim):** `[x, y, vx, vy, ball_x, ball_y, ball_vx, ball_vy, opp_x, opp_y, opp_vx, opp_vy]` — all divided by a scale factor of 10.

**Boost (16-dim):** Appends `[powerup_avail, powerup_timer, opp_powerup_avail, opp_powerup_timer]`.

## Actions

**Base — `Discrete(6)`:**

| Index | Action    | `[fwd, bwd, jump]` |
|-------|-----------|---------------------|
| 0     | Noop      | `[0, 0, 0]`        |
| 1     | Left      | `[1, 0, 0]`        |
| 2     | UpLeft    | `[1, 0, 1]`        |
| 3     | Up        | `[0, 0, 1]`        |
| 4     | UpRight   | `[0, 1, 1]`        |
| 5     | Right     | `[0, 1, 0]`        |

**Boost — `Discrete(13)`:** Indices 0–5 as above, indices 6–12 repeat with boost activated (`[fwd, bwd, jump, 1]`).

The functional step pipeline takes continuous `[fwd, bwd, jump]` arrays directly (shape `(2, 3)` or `(2, 4)` with boost). The discrete-to-continuous mapping is handled by the environment wrapper.

## Configuration

Pass a config dict to the environment constructor:

```python
env = SlimeVolleyEnv(config={
    "backend": "numpy",      # "numpy" or "jax"
    "max_steps": 3000,       # max timesteps per episode
    "survival_reward": False, # add +0.01 reward per step survived
    "from_pixels": False,     # pixel observations (numpy only)
    "human_inputs": False,    # invert left agent controls for human play
    "seed": None,             # initial RNG seed
})
```

## EnvState

All mutable game state is captured in a single frozen dataclass (`EnvState`). State transitions use `dataclasses.replace()` to produce new states. On the JAX backend, `EnvState` is registered as a pytree for `jit`/`vmap`/`scan` compatibility.

| Field                | Shape    | Description                          |
|----------------------|----------|--------------------------------------|
| `ball_pos`           | `(2,)`   | Ball position `[x, y]`              |
| `ball_vel`           | `(2,)`   | Ball velocity `[vx, vy]`            |
| `ball_prev_pos`      | `(2,)`   | Previous ball position               |
| `agent_pos`          | `(2, 2)` | Agent positions `[[x,y], [x,y]]`    |
| `agent_vel`          | `(2, 2)` | Agent velocities                     |
| `agent_desired_vel`  | `(2, 2)` | Desired velocities from actions      |
| `agent_life`         | `(2,)`   | Lives remaining per agent            |
| `agent_powerup_avail`| `(2,)`   | Powerups available                   |
| `agent_powerup_timer`| `(2,)`   | Powerup countdown timer              |
| `delay_life`         | `()`     | DelayScreen frames remaining         |
| `time`               | `()`     | Timestep counter                     |
| `done`               | `()`     | Episode done flag                    |
| `rng_key`            | —        | JAX PRNGKey or numpy Generator       |

## Project Structure

```
slime_volleyball/
├── slimevolley_env.py         # SlimeVolleyEnv (Gymnasium wrapper)
├── slimevolley_boost_env.py   # SlimeVolleyBoostEnv (boost/powerup variant)
├── baseline_policy.py         # 120-parameter RNN baseline opponent
├── rendering.py               # Pyglet-based 2D rendering
├── backend/
│   ├── __init__.py            # xp proxy (numpy or jax.numpy)
│   ├── _dispatch.py           # set_backend(), get_backend(), BackendProxy
│   ├── array_ops.py           # set_at() for JAX .at[].set() vs numpy indexing
│   └── env_state.py           # EnvState frozen dataclass + pytree registration
├── core/
│   ├── constants.py           # Physics constants, rendering dimensions
│   ├── physics.py             # Pure functional physics (agents, ball, collisions)
│   ├── observations.py        # Perspective-normalized observation computation
│   ├── step_pipeline.py       # Functional reset/step + factory builders
│   ├── game.py                # Legacy OOP game engine (rendering only)
│   ├── agent.py               # Legacy Agent class (rendering only)
│   ├── objects.py             # Particle, Wall, RelativeState (rendering only)
│   └── utils.py               # Coordinate conversion, DelayScreen
tests/
├── test_backend.py            # Backend dispatch tests
├── test_env_state.py          # EnvState + JAX pytree tests
├── test_physics.py            # Physics function tests
├── test_observations.py       # Observation computation tests
├── test_step_pipeline.py      # Functional reset/step tests (numpy + JAX)
├── test_env_wrapper.py        # Environment wrapper tests (numpy, JAX, boost)
├── test_rendering.py          # Rendering state sync tests
└── test_cross_backend.py      # Cross-backend parity tests
examples/
├── train_slimevolley_jax.py   # IPPO training script (JAX)
├── episode.gif                # Trained policy visualization
├── policy.onnx                # ONNX-exported trained policy
└── slimevolley_training.png   # Training reward curve
```

## Training

An IPPO (Independent PPO) training script is provided in `examples/train_slimevolley_jax.py`. It uses the JAX functional API with `jax.vmap` for parallel environments and `jax.lax.scan` for the training loop.

```bash
python examples/train_slimevolley_jax.py
```

This produces:
- `slimevolley_training.png` — reward curve over training
- `episode.gif` — GIF of the trained policy playing
- `policy.onnx` — ONNX export of the trained actor network

## Interactive Visualization

```bash
python visualize_env.py
```

Uses the legacy OOP game engine with pyglet rendering. Keyboard controls:

| Key          | Agent       | Action |
|--------------|-------------|--------|
| Left Arrow   | Right agent | Left   |
| Right Arrow  | Right agent | Right  |
| Up Arrow     | Right agent | Jump   |
| A            | Left agent  | Left   |
| D            | Left agent  | Right  |
| W            | Left agent  | Jump   |

## Running Tests

```bash
pytest tests/
```

## Credits

- **Original game:** [David Ha](https://otoro.net/slimevolley) ([blog post](https://blog.otoro.net/2015/03/28/neural-slime-volleyball/))
- **Original gym environment:** [hardmaru/slimevolleygym](https://github.com/hardmaru/slimevolleygym)

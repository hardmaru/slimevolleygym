"""
Port of Neural Slime Volleyball to Python Gym Environment

David Ha (2020)

Original version:

https://otoro.net/slimevolley
https://blog.otoro.net/2015/03/28/neural-slime-volleyball/
https://github.com/hardmaru/neuralslimevolley

Supports dual backends (numpy / jax) via slime_volleyball.backend.
"""

import typing

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.utils import seeding
import numpy as np

from slime_volleyball.backend import set_backend
from slime_volleyball.backend import xp
from slime_volleyball.core import constants
from slime_volleyball.core.step_pipeline import build_step_fn, build_reset_fn

np.set_printoptions(threshold=20, precision=3, suppress=True, linewidth=200)


class Actions:
    Noop = 0
    Left = 1
    UpLeft = 2
    Up = 3
    UpRight = 4
    Right = 5


class SlimeVolleyEnv(gym.Env):
    """
    Gym wrapper for Slime Volley game with dual numpy/jax backend.

    Backend selection:
        env = SlimeVolleyEnv(backend="numpy")  # default, no JAX required
        env = SlimeVolleyEnv(backend="jax")    # JAX backend, exposes jax_step/jax_reset

    The environment wraps a pure functional core. On the JAX backend,
    env.jax_step and env.jax_reset expose JIT-compiled functions that
    work with jax.vmap and jax.lax.scan for vectorized training.

    By default, the agent you are training controls the right agent.
    The agent on the left is controlled by the baseline RNN policy
    when no action is provided for it.
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "state"],
        "video.frames_per_second": 50,
    }

    action_table = [
        [0, 0, 0],  # NOOP
        [1, 0, 0],  # LEFT (forward)
        [1, 0, 1],  # UPLEFT (forward jump)
        [0, 0, 1],  # UP (jump)
        [0, 1, 1],  # UPRIGHT (backward jump)
        [0, 1, 0],  # RIGHT (backward)
    ]

    default_config = {
        "from_pixels": False,
        "survival_reward": False,
        "max_steps": 3000,
        "human_inputs": False,
        "backend": "numpy",
        "seed": None,
    }

    def __init__(
        self,
        config: dict[str, typing.Any] | None = None,
        render_mode: str | None = None,
    ):
        if config is None:
            config = self.default_config

        # Backend setup
        backend = config.get("backend", self.default_config["backend"])
        set_backend(backend)
        self._backend = backend

        self._agent_ids = set(["agent_left", "agent_right"])
        self.t = 0
        self.max_steps = config.get("max_steps", 3000)
        self.max_lives = constants.MAXLIVES
        self.from_pixels = config.get("from_pixels", False)
        self.survival_reward = config.get("survival_reward", False)
        self.human_inputs = config.get("human_inputs", False)
        self.boost = False  # base env has no boost

        if self.from_pixels:
            constants.setPixelObsMode()
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(constants.PIXEL_HEIGHT, constants.PIXEL_WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            obs_dim = 16 if self.boost else 12
            high = np.array([np.finfo(np.float32).max] * obs_dim)
            observation_space = spaces.Dict(
                {"obs": spaces.Box(-high, high, shape=(obs_dim,))}
            )

        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(len(self.action_table))
                for agent_id in self._agent_ids
            }
        )
        self.observation_space = spaces.Dict(
            {agent_id: observation_space for agent_id in self._agent_ids}
        )

        self._env_state = None
        self._step_fn = None
        self._reset_fn = None

        # Baseline policy (numpy only, lazy loaded)
        self._policy = None

        # Rendering state
        self.canvas = None
        self.previous_rgbarray = None
        self.viewer = None
        self.render_mode = render_mode

        # Legacy game object for rendering (numpy only)
        self._game = None

        self.np_random, _ = seeding.np_random(config.get("seed", None))

        super(SlimeVolleyEnv, self).__init__()

    @property
    def policy(self):
        """Lazy-load baseline policy (numpy only)."""
        if self._policy is None:
            from slime_volleyball.baseline_policy import BaselinePolicy
            self._policy = BaselinePolicy()
        return self._policy

    def _make_rng(self, seed=None):
        """Create an RNG key appropriate for the current backend."""
        if self._backend == "jax":
            import jax
            if seed is None:
                seed = 0
            return jax.random.PRNGKey(seed)
        else:
            if seed is not None:
                self.np_random, _ = seeding.np_random(seed)
            return self.np_random

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, typing.Any]]:
        self.t = 0

        # Build functional pipeline
        self._step_fn = build_step_fn()
        self._reset_fn = build_reset_fn(
            max_steps=self.max_steps,
            max_lives=self.max_lives,
            boost=self.boost,
        )

        rng = self._make_rng(seed)
        obs_arr, self._env_state, infos = self._reset_fn(rng)

        # Also create legacy game for rendering (numpy only)
        if self._backend == "numpy" and not self.from_pixels:
            self._sync_legacy_game()

        return self._obs_array_to_dict(obs_arr), {}

    def step(self, actions):
        self.t += 1

        # Handle single-agent mode (int action = right agent only)
        if isinstance(actions, int):
            actions = {"agent_right": actions}

        left_action = self._resolve_action(actions.get("agent_left"), "agent_left")
        right_action = self._resolve_action(actions.get("agent_right"), "agent_right")

        if self.human_inputs and left_action is not None:
            left_action = self.invert_action(left_action)

        # Build action array
        actions_arr = xp.array([left_action, right_action], dtype=xp.float32)

        obs_arr, self._env_state, rewards_arr, terminateds_arr, truncateds_arr, infos = \
            self._step_fn(self._env_state, actions_arr)

        # Add survival reward
        survival = 0.01 if self.survival_reward else 0.0
        rewards_arr = rewards_arr + xp.float32(survival)

        obs = self._obs_array_to_dict(obs_arr)
        rewards = {
            "agent_left": float(rewards_arr[0]),
            "agent_right": float(rewards_arr[1]),
        }
        terminateds = {
            "agent_left": bool(terminateds_arr[0]),
            "agent_right": bool(terminateds_arr[1]),
            "__all__": bool(terminateds_arr[0]),
        }
        truncateds = {
            "agent_left": bool(truncateds_arr[0]),
            "agent_right": bool(truncateds_arr[1]),
            "__all__": bool(truncateds_arr[0]),
        }

        return obs, rewards, terminateds, truncateds, {}

    def _resolve_action(self, action, agent_id):
        """Convert a discrete action to [fwd, bwd, jump] array, or use baseline."""
        if action is None:
            if agent_id == "agent_left":
                # Use baseline policy
                obs = self._get_agent_obs_for_baseline()
                action = self.policy.predict(obs)
                return action
            return [0, 0, 0]

        return self.discrete_to_box(action)

    def _get_agent_obs_for_baseline(self):
        """Get left agent's observation as numpy array for baseline policy."""
        if self._env_state is None:
            return np.zeros(12, dtype=np.float32)
        obs = compute_single_obs(self._env_state, agent_idx=0, boost=False)
        return np.asarray(obs)

    def discrete_to_box(self, n):
        """Convert discrete action n to the actual action array."""
        if n is None:
            return [0, 0, 0]
        if isinstance(n, (list, tuple, np.ndarray)):
            if len(n) >= 3:
                return list(n)
        assert (int(n) == n) and (n >= 0) and (n < len(self.action_table))
        return self.action_table[n]

    @staticmethod
    def invert_action(action: list) -> list:
        left, right, up = action[0], action[1], action[2]
        return [right, left, up] + list(action[3:])

    def _obs_array_to_dict(self, obs_arr):
        """Convert (2, obs_dim) array to dict of per-agent observations."""
        if self.from_pixels:
            # Pixel mode not supported in functional pipeline
            raise NotImplementedError("Pixel observations not yet supported in dual-backend mode")

        return {
            "agent_left": {"obs": np.asarray(obs_arr[0])},
            "agent_right": {"obs": np.asarray(obs_arr[1])},
        }

    def _sync_legacy_game(self):
        """Create legacy game object for rendering from current state."""
        # Only needed for rendering, deferred to render time
        pass

    # ------------------------------------------------------------------
    # JAX functional API (JaxMARL-compatible)
    # ------------------------------------------------------------------

    @property
    def jax_step(self):
        """Raw JIT-compiled step function for direct JIT/vmap usage.

        Signature: (EnvState, actions) -> (obs, EnvState, rewards, terminateds, truncateds, infos)
        """
        if self._backend != "jax":
            raise RuntimeError("jax_step is only available with backend='jax'")
        if self._step_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_step")
        return self._step_fn

    @property
    def jax_reset(self):
        """Raw JIT-compiled reset function for direct JIT/vmap usage.

        Signature: (rng_key) -> (obs, EnvState, infos)
        """
        if self._backend != "jax":
            raise RuntimeError("jax_reset is only available with backend='jax'")
        if self._reset_fn is None:
            raise RuntimeError("Must call reset() before accessing jax_reset")
        return self._reset_fn

    # ------------------------------------------------------------------
    # Rendering (numpy only, uses legacy game objects)
    # ------------------------------------------------------------------

    def render(self):
        if self._backend == "jax":
            raise NotImplementedError("Rendering is only available with backend='numpy'")

        if self._env_state is None:
            return None

        # Lazy import and build legacy game for rendering
        if self._game is None:
            self._build_legacy_game()

        self._update_legacy_game_from_state()

        mode = self.render_mode
        if constants.PIXEL_MODE:
            from slime_volleyball.core import utils
            self.canvas = self._game.display(self.canvas)
            self.canvas = utils.downsize_image(self.canvas)

            if mode == "state":
                return np.copy(self.canvas)

            larger_canvas = utils.upsize_image(self.canvas)
            self._check_viewer()
            self.viewer.imshow(larger_canvas)
            if mode == "rgb_array":
                return larger_canvas
        else:
            from slime_volleyball import rendering as r
            if self.viewer is None:
                self.viewer = r.Viewer(constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT)
            self._game.display(self.viewer)
            return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _build_legacy_game(self):
        """Build a legacy SlimeVolleyGame for rendering."""
        from slime_volleyball.core.game import SlimeVolleyGame
        self._game = SlimeVolleyGame(self.np_random)

    def _update_legacy_game_from_state(self):
        """Sync legacy game objects from EnvState for rendering."""
        if self._game is None or self._env_state is None:
            return
        s = self._env_state
        self._game.ball.x = float(s.ball_pos[0])
        self._game.ball.y = float(s.ball_pos[1])
        self._game.ball.vx = float(s.ball_vel[0])
        self._game.ball.vy = float(s.ball_vel[1])
        self._game.agent_left.x = float(s.agent_pos[0, 0])
        self._game.agent_left.y = float(s.agent_pos[0, 1])
        self._game.agent_right.x = float(s.agent_pos[1, 0])
        self._game.agent_right.y = float(s.agent_pos[1, 1])

    def _check_viewer(self):
        if self.viewer is None:
            from slime_volleyball import rendering
            self.viewer = rendering.SimpleImageViewer()

    def close(self):
        if self.viewer:
            self.viewer.close()


def compute_single_obs(state, agent_idx, boost=False):
    """Compute observation for a single agent from EnvState."""
    from slime_volleyball.core.observations import compute_observations
    obs = compute_observations(
        state.agent_pos, state.agent_vel,
        state.ball_pos, state.ball_vel,
        state.agent_powerup_avail, state.agent_powerup_timer,
        boost=boost,
    )
    return obs[agent_idx]


class SlimeVolleyPixelEnv(SlimeVolleyEnv):
    from_pixels = True


register(
    id="SlimeVolley-v0", entry_point="slimevolleygym.slimevolley:SlimeVolleyEnv"
)
register(
    id="SlimeVolleyPixel-v0",
    entry_point="slimevolleygym.slimevolley:SlimeVolleyPixelEnv",
)

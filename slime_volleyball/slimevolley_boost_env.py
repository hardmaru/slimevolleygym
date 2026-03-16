"""
Slime Volleyball Boost Environment — extends base with powerup mechanic.

16-dim observations, Discrete(13) actions (adds boost variants).
Uses the same dual-backend functional core as the base environment.
"""

import typing

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from slime_volleyball.slimevolley_env import SlimeVolleyEnv
from slime_volleyball.core import constants


class SlimeVolleyBoostEnv(SlimeVolleyEnv):
    """SlimeVolleyEnv with boost/powerup mechanic."""

    action_table = [
        [0, 0, 0, 0],  # NOOP
        [1, 0, 0, 0],  # LEFT (forward)
        [1, 0, 1, 0],  # UPLEFT (forward jump)
        [0, 0, 1, 0],  # UP (jump)
        [0, 1, 1, 0],  # UPRIGHT (backward jump)
        [0, 1, 0, 0],  # RIGHT (backward)
        # Boost versions
        [0, 0, 0, 1],  # NOOP (boost)
        [1, 0, 0, 1],  # LEFT boost
        [1, 0, 1, 1],  # UPLEFT boost
        [0, 0, 1, 1],  # UP boost
        [0, 1, 1, 1],  # UPRIGHT boost
        [0, 1, 0, 1],  # RIGHT boost
        [0, 0, 0, 1],  # NOOP boost (duplicate)
    ]

    def __init__(
        self,
        config: dict[str, typing.Any] | None = None,
        render_mode: str | None = None,
    ):
        super().__init__(config=config, render_mode=render_mode)
        self.boost = True

        # Override observation and action spaces for boost
        if not self.from_pixels:
            high = np.array([np.finfo(np.float32).max] * 16)
            observation_space = spaces.Dict(
                {"obs": spaces.Box(-high, high, shape=(16,))}
            )
            self.observation_space = spaces.Dict(
                {agent_id: observation_space for agent_id in self._agent_ids}
            )

        self.action_space = spaces.Dict(
            {
                agent_id: spaces.Discrete(len(self.action_table))
                for agent_id in self._agent_ids
            }
        )

    @staticmethod
    def invert_action(action: list) -> list:
        left, right, up, boost = action[0], action[1], action[2], action[3]
        return [right, left, up, boost]

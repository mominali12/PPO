from __future__ import annotations

import gymnasium as gym
import numpy as np


class RawEpisodeStatistics(gym.Wrapper):
    """Expose raw full-game return before reward clipping / EpisodicLifeEnv."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.raw_episode_return = 0.0
        self.raw_episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_episode_return = 0.0
        self.raw_episode_length = 0
        info["raw_episode_return"] = np.array(0.0, dtype=np.float32)
        info["raw_episode_length"] = np.array(0, dtype=np.int64)
        info["raw_episode_done"] = np.array(False, dtype=np.bool_)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.raw_episode_return += float(reward)
        self.raw_episode_length += 1
        done = bool(terminated or truncated)

        info["raw_episode_return"] = np.array(
            self.raw_episode_return if done else 0.0,
            dtype=np.float32,
        )
        info["raw_episode_length"] = np.array(
            self.raw_episode_length if done else 0,
            dtype=np.int64,
        )
        info["raw_episode_done"] = np.array(done, dtype=np.bool_)

        return obs, reward, terminated, truncated, info
"""Environment: config wrapper + factory for TorchRL environments.

The Environment holds configuration and produces TransformedEnv instances
on demand.  It never holds a live env itself — the Trainer controls env
lifecycle by calling ``make_env()`` when needed.
"""
from __future__ import annotations

from typing import Sequence

from torchrl.envs import EnvBase

from src.environments.factory import make_env


class Environment:
    """Wraps environment parameters and produces TorchRL env instances.

    Args:
        name: Environment name (e.g. ``"CartPole-v1"``, ``"ALE/Breakout-v5"``).
        backend: Backend to use (``"gymnasium"``, ``"dm_control"``, ``"envpool"``).
        obs_shape: Observation shape after preprocessing (e.g. ``[4]`` or ``[4, 84, 84]``).
        num_actions: Number of actions (discrete count or continuous dim).
        frame_stack: Number of frames to stack via CatFrames.
        grayscale: Convert RGB observations to grayscale.
        resize: ``[H, W]`` to resize pixel observations; ``None`` to skip.
        clip_rewards: Clip rewards to ``{-1, 0, +1}`` (standard Atari preprocessing).
        normalize_obs: Apply running mean/std normalisation to observations.
        task: dm_control task string (e.g. ``"walk"`` for ``humanoid-walk``).
        max_episode_steps: Maximum steps per episode; ``None`` uses env default.
        **kwargs: Extra keyword arguments forwarded to the base env constructor.
    """

    def __init__(
        self,
        name: str,
        backend: str,
        obs_shape: Sequence[int],
        num_actions: int,
        frame_stack: int = 1,
        grayscale: bool = False,
        resize: Sequence[int] | None = None,
        clip_rewards: bool = False,
        normalize_obs: bool = False,
        task: str | None = None,
        max_episode_steps: int | None = None,
        **kwargs,
    ) -> None:
        self.obs_shape: tuple[int, ...] = tuple(obs_shape)
        self.num_actions = int(num_actions)
        self._factory_kwargs: dict = {
            "name": name,
            "backend": backend,
            "obs_shape": obs_shape,
            "num_actions": num_actions,
            "frame_stack": frame_stack,
            "grayscale": grayscale,
            "resize": resize,
            "clip_rewards": clip_rewards,
            "normalize_obs": normalize_obs,
            "task": task,
            "max_episode_steps": max_episode_steps,
            **kwargs,
        }

    def make_env(
        self,
        num_envs: int = 1,
        device: str = "cpu",
    ) -> EnvBase:
        """Create a (possibly vectorised) TorchRL env from stored parameters.

        Args:
            num_envs: Number of parallel envs (>1 → ParallelEnv).
            device: Target device string.

        Returns:
            TransformedEnv (or ParallelEnv wrapping TransformedEnvs).
        """
        return make_env(**self._factory_kwargs, num_envs=num_envs, device=device)

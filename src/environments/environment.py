"""Environment: config wrapper + factory for TorchRL environments.

The Environment holds configuration and produces TransformedEnv instances
on demand.  It never holds a live env itself — the Trainer controls env
lifecycle by calling ``make_env()`` when needed.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
from torchrl.envs import EnvBase

from src.environments.factory import make_env


class Environment:
    """Wraps environment config and produces TorchRL env instances.

    Args:
        cfg: the ``environment`` sub-config from Hydra
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.obs_shape: tuple[int, ...] = tuple(cfg.obs_shape)
        self.num_actions: int = int(cfg.num_actions)

    def make_env(
        self,
        num_envs: int = 1,
        device: str = "cpu",
    ) -> EnvBase:
        """Create a (possibly vectorised) TorchRL env from config.

        Args:
            num_envs: number of parallel envs (>1 → ParallelEnv)
            device: target device string

        Returns:
            TransformedEnv (or ParallelEnv wrapping TransformedEnvs)
        """
        env_kwargs = OmegaConf.to_container(self.cfg, resolve=True)
        env_kwargs.pop("_target_", None)
        return make_env(**env_kwargs, num_envs=num_envs, device=device)

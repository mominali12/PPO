"""Trainers and callback infrastructure.

Two trainer types handle different RL paradigms:

* :class:`EpisodicTrainer` — rolls out full episodes via ``env.rollout()``
  then passes them to ``algorithm.step()``.  Used by REINFORCE.
* :class:`StepTrainer` — uses ``SyncDataCollector`` for fixed-size batches
  then passes them to ``algorithm.step()``.  Used by DQN, PPO.

Both inherit from :class:`BaseTrainer` which owns device resolution,
environment creation, evaluation, and callback dispatch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms.base import BaseAlgorithm
from src.algorithms.utils import last_episode_return
from src.environments.environment import Environment
from src.utils.device import resolve_device


# ---------------------------------------------------------------------------
# Callback protocol 
# ---------------------------------------------------------------------------


class TrainerEvent(Enum):
    ON_TRAIN_START = auto()
    ON_STEP_END = auto()
    ON_TRAIN_END = auto()
    ON_EVAL_START = auto()
    ON_EVAL_END = auto()


@runtime_checkable
class Callback(Protocol):
    def on_train_start(self, state: dict[str, Any]) -> None: ...
    def on_step_end(self, metrics: dict[str, float], step: int) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


def fire_callbacks(
    event: TrainerEvent,
    callbacks: list,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a training event to all callbacks that implement the matching method."""
    method_name = event.name.lower()
    for cb in callbacks:
        method = getattr(cb, method_name, None)
        if callable(method):
            method(*args, **kwargs)


# ---------------------------------------------------------------------------
# Base Trainer
# ---------------------------------------------------------------------------


class BaseTrainer(ABC):
    """Base class for all trainers.

    Owns: device resolution, environment creation, eval loop, callbacks,
    and checkpoint orchestration.

    Args:
        cfg: full Hydra config
        algorithm: algorithm instance (already ``__init__``'d, not yet set up)
        environment: environment config wrapper
        callbacks: list of callback objects
    """

    def __init__(
        self,
        cfg: DictConfig,
        algorithm: BaseAlgorithm,
        environment: Environment,
        callbacks: list | None = None,
    ) -> None:
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.algorithm = algorithm
        self.environment = environment
        self.callbacks = callbacks or []

        # Trainer owns device
        self.device = resolve_device(
            self.trainer_cfg.accelerator,
            list(self.trainer_cfg.devices),
        )
        self.algorithm.device = self.device

        self._step: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Create environment and set up the algorithm."""
        num_envs = int(self.trainer_cfg.get("num_envs", 1))

        self.train_env = self.environment.make_env(
            num_envs=num_envs,
            device=str(self.device),
        )

        # Algorithm reads env specs to build networks
        self.algorithm.setup(self.train_env)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            dict of final training metrics
        """
        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            self.callbacks,
            state={"cfg": self.cfg},
        )

        metrics = self._training_loop()

        fire_callbacks(
            TrainerEvent.ON_TRAIN_END,
            self.callbacks,
            state={"cfg": self.cfg},
        )
        return metrics

    @abstractmethod
    def _training_loop(self) -> dict[str, float]:
        """Subclass-specific training loop."""

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int) -> dict[str, float]:
        """Run evaluation episodes using the greedy policy.

        Creates a fresh single-env for eval (separate from the train env).
        """
        eval_env = self.environment.make_env(
            num_envs=1,
            device=str(self.device),
        )
        policy = self.algorithm.get_policy()

        returns: list[float] = []
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = eval_env.reset()
                episode_return = 0.0
                done = False
                while not done:
                    td = policy(td)
                    td = eval_env.step(td)
                    episode_return += td["next", "reward"].sum().item()
                    done = (
                        td["next", "done"].any().item()
                        or td["next", "terminated"].any().item()
                    )
                    td = td["next"]
                returns.append(episode_return)

        eval_env.close()
        t = torch.tensor(returns, dtype=torch.float32)
        return {
            "eval/return_mean": t.mean().item(),
            "eval/return_std": t.std().item(),
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save algorithm state + trainer step."""
        self.algorithm.save_checkpoint(path, step=self._step)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore algorithm state + trainer step."""
        self._step = self.algorithm.load_checkpoint(path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _should_log(self, log_every: int, batch_frames: int) -> bool:
        """Check if we crossed a ``log_every`` boundary this iteration."""
        prev = self._step - batch_frames
        return prev // log_every < self._step // log_every


# ---------------------------------------------------------------------------
# Episodic Trainer (REINFORCE)
# ---------------------------------------------------------------------------


class EpisodicTrainer(BaseTrainer):
    """Trainer for episodic algorithms that use ``env.rollout()``.

    Each iteration: roll out one full episode → ``algorithm.step(episode)``.
    No ``SyncDataCollector`` is used.
    """

    def _training_loop(self) -> dict[str, float]:
        total_frames = int(self.trainer_cfg.total_frames)
        log_every = int(self.trainer_cfg.log_every_n_steps)

        explore_policy = self.algorithm.get_explore_policy()
        metrics: dict[str, float] = {}

        while self._step < total_frames:
            remaining = total_frames - self._step

            rollout = self.train_env.rollout(
                max_steps=remaining,
                policy=explore_policy,
                auto_reset=True,
            )

            rollout = self.algorithm.on_batch_collected(rollout)

            batch_frames = rollout.batch_size[0]

            if not self.algorithm.should_skip_update(self._step + batch_frames):
                metrics = self.algorithm.step(rollout)

            self._step += batch_frames
            self.algorithm.on_step_complete(self._step)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(rollout)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics


# ---------------------------------------------------------------------------
# Step Trainer (DQN, PPO)
# ---------------------------------------------------------------------------


class StepTrainer(BaseTrainer):
    """Trainer for step-based algorithms using ``SyncDataCollector``.

    Each iteration: collector yields a batch → ``algorithm.step(batch)``.
    """

    def setup(self) -> None:
        """Create env, set up algorithm, then create the collector."""
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import SyncDataCollector

        collector_cfg = self.algorithm.get_collector_config()

        self.collector = SyncDataCollector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=collector_cfg.frames_per_batch,
            total_frames=collector_cfg.total_frames,
            split_trajs=collector_cfg.split_trajs,
            device=self.device,
            storing_device=self.device,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)

        metrics: dict[str, float] = {}
        for batch in self.collector:
            batch = self.algorithm.on_batch_collected(batch)

            batch_frames = batch.numel()
            self._step += batch_frames

            if not self.algorithm.should_skip_update(self._step):
                metrics = self.algorithm.step(batch)

            self.algorithm.on_step_complete(self._step)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(batch)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics

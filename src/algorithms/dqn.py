"""Deep Q-Network (DQN) algorithm.

Compatible environments:
  - Discrete-action Gym environments (CartPole-v1, MLP network)
  - Atari pixel environments (ALE/Breakout-v5, CNN network)

Architecture:
  - Q-network: MLP or AtariCNN → action values
  - Policy: QValueActor with epsilon-greedy exploration (EGreedyModule)
  - Loss: DQNLoss with double-DQN target network
  - Target update: HardUpdate every N frames
  - Buffer: ReplayBuffer with LazyMemmapStorage
  - Collector: SyncDataCollector (step-based, split_trajs=False)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from src.algorithms.base import BaseAlgorithm, TrainingState
from src.algorithms.utils import last_episode_return
from src.networks.factory import make_network
from src.trainer import TrainerEvent, fire_callbacks


@dataclass
class DQNConfig:
    """Hyperparameters for DQN with double-DQN target network and epsilon-greedy exploration.

    Defaults are tuned for CartPole-v1 (MLP network).
    For Atari pixel environments, override ``network`` to ``cnn_atari`` and increase
    ``replay_buffer.capacity``, ``init_random_frames``, and ``eps_annealing_frames``.
    """

    # Data collection
    frames_per_batch: int = 200       # frames added to replay buffer per collector step
    init_random_frames: int = 5_000   # warm-up frames collected with random policy before training starts

    # Replay buffer
    replay_buffer: dict = field(default_factory=lambda: {
        "capacity": 100_000,   # maximum number of transitions stored
        "batch_size": 128,     # number of transitions sampled per gradient update
    })

    # Optimization
    lr: float = 1e-4                   # Adam learning rate
    gamma: float = 0.99                # discount factor
    max_grad_norm: float = 10.0        # gradient clipping threshold
    target_update_every: int = 1_000   # hard-copy target network every N environment frames

    # Exploration — epsilon-greedy annealing schedule
    eps_start: float = 1.0             # initial exploration probability
    eps_end: float = 0.05              # final exploration probability
    eps_annealing_frames: int = 100_000  # frames over which epsilon is linearly annealed

    # Network architecture (MLP for CartPole; override to cnn_atari for Atari)
    network: dict = field(default_factory=lambda: {
        "architecture": "mlp",
        "hidden_sizes": [128, 128],
        "activation": "relu",
        "layer_norm": False,
    })


class DQNAlgorithm(BaseAlgorithm):
    """DQN with double-DQN target network and epsilon-greedy exploration.

    Args:
        cfg: full Hydra config
        device: resolved torch.device
    """

    def setup(self) -> None:
        from tensordict.nn import TensorDictSequential
        from torchrl.collectors import SyncDataCollector
        from torchrl.data import LazyMemmapStorage, ReplayBuffer, TensorDictReplayBuffer
        from torchrl.data.replay_buffers.samplers import RandomSampler
        from torchrl.modules import EGreedyModule, QValueActor
        from torchrl.objectives import DQNLoss, HardUpdate

        self.acfg = self._build_acfg(DQNConfig())
        acfg = self.acfg
        ecfg = self.cfg.environment

        # --- Environment ---
        from src.environments.factory import make_env
        env_kwargs = OmegaConf.to_container(ecfg, resolve=True)
        env_kwargs.pop("_target_", None)
        self.env = make_env(
            **env_kwargs,
            num_envs=int(self.cfg.trainer.num_envs),
            device=str(self.device),
        )

        obs_shape = tuple(ecfg.obs_shape)
        num_actions = int(ecfg.num_actions)

        # --- Q-network ---
        q_net = make_network(acfg.network, obs_shape, num_actions).to(self.device)

        # Determine observation key (pixels envs use "observation" after transforms)
        obs_key = "observation"
        q_module = TensorDictModule(
            q_net,
            in_keys=[obs_key],
            out_keys=["action_value"],
        )
        self.q_actor = QValueActor(
            module=q_module,
            in_keys=[obs_key],
            spec=self.env.action_spec,
        ).to(self.device)

        # Epsilon-greedy exploration wrapper (used only during collection)
        self.eps_module = EGreedyModule(
            spec=self.env.action_spec,
            annealing_num_steps=int(acfg.eps_annealing_frames),
            eps_init=float(acfg.eps_start),
            eps_end=float(acfg.eps_end),
            action_key="action",
        )
        self.explore_policy = TensorDictSequential(
            self.q_actor,
            self.eps_module,
        ).to(self.device)

        # --- Loss (DQN with target network) ---
        self.loss_module = DQNLoss(
            value_network=self.q_actor,
            loss_function="l2",
            delay_value=True,  # creates a target network clone
        ).to(self.device)
        self.loss_module.make_value_estimator(gamma=float(acfg.gamma))

        # Hard-copy target network update
        self.target_updater = HardUpdate(
            self.loss_module,
            value_network_update_interval=int(acfg.target_update_every),
        )

        # --- Replay buffer ---
        buf_cfg = acfg.replay_buffer
        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(int(buf_cfg.capacity)),
            sampler=RandomSampler(),
            batch_size=int(buf_cfg.batch_size),
        )

        # --- Optimizer ---
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=float(acfg.lr),
        )

        # --- Data collector (uses epsilon-greedy policy) ---
        self.collector = SyncDataCollector(
            create_env_fn=self.env,
            policy=self.explore_policy,
            frames_per_batch=int(acfg.frames_per_batch),
            total_frames=int(self.cfg.trainer.total_frames),
            split_trajs=False,
            device=self.device,
            storing_device=self.device,
        )

    def train(
        self,
        trainer_cfg: DictConfig,
        callbacks: list,
    ) -> dict[str, float]:
        acfg = self.acfg
        init_random_frames = int(acfg.init_random_frames)
        log_every = int(trainer_cfg.log_every_n_steps)

        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            callbacks,
            state={"cfg": self.cfg},
        )

        metrics: dict[str, float] = {}
        for batch in self.collector:
            # Add new transitions to replay buffer
            self.replay_buffer.extend(batch.reshape(-1))
            self._step += batch.numel()

            # Skip updates during warm-up
            if self._step < init_random_frames:
                continue

            # Step epsilon decay
            self.eps_module.step(batch.numel())

            # Gradient update
            sample = self.replay_buffer.sample().to(self.device)
            metrics = self._update(sample)

            # Hard-update target network on schedule
            self.target_updater.step()

            if self._step % log_every < int(acfg.frames_per_batch):
                metrics["reward/last"] = last_episode_return(batch)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        fire_callbacks(TrainerEvent.ON_TRAIN_END, callbacks, state={"cfg": self.cfg})
        return metrics

    def _update(self, batch: TensorDict) -> dict[str, float]:
        acfg = self.acfg

        loss_td = self.loss_module(batch)
        loss = loss_td["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), float(acfg.max_grad_norm)
        )
        self.optimizer.step()

        return {
            "loss/td": loss.item(),
            "q/mean": loss_td.get("pred_value", torch.tensor(0.0)).mean().item(),
        }

    def eval(self, num_episodes: int) -> dict[str, float]:
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        returns = []
        with torch.no_grad(), set_exploration_type(ExplorationType.GREEDY):
            for _ in range(num_episodes):
                td = self.env.reset()
                episode_return = 0.0
                done = False
                while not done:
                    td = self.q_actor(td)
                    td = self.env.step(td)
                    episode_return += td["next", "reward"].item()
                    done = td["next", "done"].item() or td["next", "terminated"].item()
                    td = td["next"]
                returns.append(episode_return)

        t = torch.tensor(returns, dtype=torch.float32)
        return {
            "eval/return_mean": t.mean().item(),
            "eval/return_std": t.std().item(),
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
        }

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=self._step,
            policy_state_dict=self.q_actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            replay_buffer_state={"storage_path": str(self.replay_buffer._storage._path)},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.q_actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        # Replay buffer storage is memory-mapped; path is preserved in checkpoint
        # but re-populating it is non-trivial; skip for resumption scenarios

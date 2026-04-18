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
  - Trainer: StepTrainer (SyncDataCollector, step-based)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks.factory import make_network


@dataclass
class DQNConfig:
    """Hyperparameters for DQN with double-DQN target network and epsilon-greedy exploration.

    Defaults are tuned for CartPole-v1 (MLP network).
    For Atari pixel environments, override ``network`` to ``cnn_atari`` and increase
    ``replay_buffer.capacity``, ``init_random_frames``, and ``eps_annealing_frames``.
    """

    # Data collection
    frames_per_batch: int = 200       # frames added to replay buffer per collector step
    init_random_frames: int = 5_000   # warm-up frames collected before training starts

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
    """DQN with double-DQN target network and epsilon-greedy exploration."""

    def setup(self, env: EnvBase) -> None:
        from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
        from torchrl.data.replay_buffers.samplers import RandomSampler
        from torchrl.modules import EGreedyModule, QValueActor
        from torchrl.objectives import DQNLoss, HardUpdate

        self.acfg = self._build_acfg(DQNConfig())
        acfg = self.acfg
        ecfg = self.cfg.environment

        obs_shape = tuple(ecfg.obs_shape)
        num_actions = int(ecfg.num_actions)

        # --- Q-network ---
        q_net = make_network(acfg.network, obs_shape, num_actions).to(self.device)

        obs_key = "observation"
        q_module = TensorDictModule(
            q_net,
            in_keys=[obs_key],
            out_keys=["action_value"],
        )
        self.q_actor = QValueActor(
            module=q_module,
            in_keys=[obs_key],
            spec=env.action_spec,
        ).to(self.device)

        # Epsilon-greedy exploration wrapper
        self.eps_module = EGreedyModule(
            spec=env.action_spec,
            annealing_num_steps=int(acfg.eps_annealing_frames),
            eps_init=float(acfg.eps_start),
            eps_end=float(acfg.eps_end),
            action_key="action",
        )
        self._explore_policy = TensorDictSequential(
            self.q_actor,
            self.eps_module,
        ).to(self.device)

        # --- Loss (DQN with target network) ---
        self.loss_module = DQNLoss(
            value_network=self.q_actor,
            loss_function="l2",
            delay_value=True,
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

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Collector configuration
    # ------------------------------------------------------------------

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=int(self.acfg.frames_per_batch),
            total_frames=int(self.cfg.trainer.total_frames),
        )

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_batch_collected(self, batch: TensorDict) -> TensorDict:
        """Store transitions in replay buffer (runs even during warmup)."""
        self.replay_buffer.extend(batch.reshape(-1))
        return batch

    def should_skip_update(self, frames_collected: int) -> bool:
        return frames_collected < int(self.acfg.init_random_frames)

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Sample from replay buffer and do one gradient update."""
        sample = self.replay_buffer.sample().to(self.device)

        loss_td = self.loss_module(sample)
        loss = loss_td["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.loss_module.parameters(), float(self.acfg.max_grad_norm)
        )
        self.optimizer.step()

        return {
            "loss/td": loss.item(),
            "q/mean": loss_td.get("pred_value", torch.tensor(0.0)).mean().item(),
        }

    def on_step_complete(self, frames_collected: int) -> None:
        """Decay epsilon and update target network."""
        if frames_collected >= int(self.acfg.init_random_frames):
            self.eps_module.step(int(self.acfg.frames_per_batch))
            self.target_updater.step()

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.q_actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"storage_path": str(self.replay_buffer._storage._path)},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.q_actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)

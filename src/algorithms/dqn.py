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

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks.factory import make_network


class DQNAlgorithm(BaseAlgorithm):
    """DQN with double-DQN target network and epsilon-greedy exploration.

    Defaults are tuned for CartPole-v1 (MLP network).
    For Atari pixel environments override ``network`` to ``cnn_atari`` and increase
    ``replay_buffer["capacity"]``, ``init_random_frames``, and ``eps_annealing_frames``.

    Args:
        cfg: Full Hydra config (trainer, logger, environment sections).
        device: Resolved torch.device; set by the Trainer.
        frames_per_batch: Frames added to replay buffer per collector step.
        init_random_frames: Warm-up frames collected before training starts.
        lr: Adam optimizer learning rate.
        gamma: Discount factor for future rewards.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        updates_per_step: Gradient updates per collector step.
        target_update_every: Hard target-update interval in env frames (ignored when tau > 0).
        tau: Soft update coefficient; 0 = hard update using target_update_every.
        compile: torch.compile the loss module.
        eps_start: Initial epsilon for epsilon-greedy exploration.
        eps_end: Final epsilon after annealing.
        eps_annealing_frames: Frames over which epsilon is linearly annealed.
        replay_buffer: Dict with keys ``capacity``, ``batch_size``, ``prefetch``, ``storage``
            (``"tensor"`` for in-memory GPU storage, ``"mmap"`` for disk-backed).
        network: Dict with keys ``architecture``, ``hidden_sizes``, ``activation``,
            ``layer_norm``. Use ``"mlp"`` for CartPole, ``"cnn_atari"`` for Atari.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device | None = None,
        *,
        frames_per_batch: int = 200,
        init_random_frames: int = 5_000,
        lr: float = 1e-4,  # Adam optimizer learning rate
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,  # gradient clipping threshold
        updates_per_step: int = 1,
        target_update_every: int = 1_000,
        tau: float = 0.0,
        compile: bool = False,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_annealing_frames: int = 100_000,
        replay_buffer: dict | None = None,
        network: dict | None = None,
    ) -> None:
        super().__init__(cfg, device)
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.lr: float = lr  #: Learning rate for the Adam optimizer on the DQN loss.
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.updates_per_step = updates_per_step
        self.target_update_every = target_update_every
        self.tau = tau
        self.compile = compile
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_annealing_frames = eps_annealing_frames
        self._replay_buffer_cfg = replay_buffer or {
            "capacity": 100_000, "batch_size": 128, "prefetch": 0, "storage": "tensor",
        }
        self._network_cfg = network or {
            "architecture": "mlp", "hidden_sizes": [128, 128],
            "activation": "relu", "layer_norm": False,
        }

    def setup(self, env: EnvBase) -> None:
        from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer
        from torchrl.data.replay_buffers.samplers import RandomSampler
        from torchrl.modules import EGreedyModule, QValueActor
        from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate

        ecfg = self.cfg.environment
        obs_shape = tuple(ecfg.obs_shape)
        num_actions = int(ecfg.num_actions)

        # --- Q-network ---
        net_cfg = OmegaConf.create(self._network_cfg)
        q_net = make_network(net_cfg, obs_shape, num_actions).to(self.device)

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
            annealing_num_steps=self.eps_annealing_frames,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
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
        self.loss_module.make_value_estimator(gamma=self.gamma)

        # Target network update (soft or hard)
        if self.tau > 0:
            self.target_updater = SoftUpdate(self.loss_module, tau=self.tau)
        else:
            self.target_updater = HardUpdate(
                self.loss_module,
                value_network_update_interval=self.target_update_every,
            )

        # --- Replay buffer ---
        buf = self._replay_buffer_cfg
        storage_type = buf.get("storage", "tensor")
        prefetch = int(buf.get("prefetch", 0))
        if storage_type == "mmap":
            storage = LazyMemmapStorage(int(buf["capacity"]))
        else:
            storage = LazyTensorStorage(int(buf["capacity"]), device=self.device)
        self.replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=RandomSampler(),
            batch_size=int(buf["batch_size"]),
            prefetch=prefetch if prefetch > 0 else None,
        )

        # --- Optimizer ---
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.lr,
        )

        # --- Optional torch.compile on the loss module ---
        if self.compile:
            self.loss_module = torch.compile(self.loss_module, dynamic=False)

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
            frames_per_batch=self.frames_per_batch,
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
        return frames_collected < self.init_random_frames

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Sample all mini-batches up front, then run sequential gradient updates."""
        n_updates = self.updates_per_step
        mb_size = int(self._replay_buffer_cfg["batch_size"])

        # One big sample call, reshaped into (n_updates, mb_size). Random sampling
        # with replacement makes this equivalent to n_updates separate sample()s.
        samples = self.replay_buffer.sample(n_updates * mb_size).to(self.device)
        samples = samples.reshape(n_updates, mb_size)

        total_loss = 0.0
        total_q = 0.0
        for i in range(n_updates):
            loss_td = self.loss_module(samples[i])
            loss = loss_td["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.loss_module.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.target_updater.step()

            total_loss += loss.item()
            total_q += loss_td.get("pred_value", torch.tensor(0.0)).mean().item()

        return {
            "loss/td": total_loss / n_updates,
            "q/mean": total_q / n_updates,
        }

    def on_step_complete(self, frames_collected: int) -> None:
        """Decay epsilon."""
        if frames_collected >= self.init_random_frames:
            self.eps_module.step(self.frames_per_batch)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.q_actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"storage_path": str(getattr(self.replay_buffer._storage, "scratch_dir", "in-memory"))},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.q_actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)

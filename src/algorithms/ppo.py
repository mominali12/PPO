"""Proximal Policy Optimization (PPO) algorithm.

Compatible environments: continuous-action dm_control environments (humanoid-walk).

Architecture:
  - Actor: MLP → Normal distribution via NormalParamExtractor + TanhNormal
  - Critic: separate MLP → scalar value estimate
  - Loss: ClipPPOLoss (clipped surrogate + value + entropy)
  - Advantage: GAE (Generalized Advantage Estimation)
  - Trainer: StepTrainer (SyncDataCollector over ParallelEnv)
  - No replay buffer (on-policy)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks.factory import make_network


@dataclass
class PPOConfig:
    """Hyperparameters for Proximal Policy Optimization (PPO) with clipped surrogate and GAE.

    Defaults are tuned for continuous-action dm_control environments (e.g. humanoid-walk).
    Override any field in the algorithm YAML or experiment config.
    """

    # Data collection
    frames_per_batch: int = 2_048    # total frames per rollout across all envs

    # PPO update schedule
    epochs_per_batch: int = 10       # number of gradient passes over each collected batch
    minibatch_size: int = 64         # mini-batch size within each epoch

    # PPO loss coefficients
    clip_epsilon: float = 0.2        # clipping parameter for the surrogate objective
    entropy_coef: float = 0.01       # entropy bonus coefficient
    critic_coef: float = 0.5         # value loss coefficient relative to policy loss

    # Advantage estimation (GAE)
    gamma: float = 0.99              # discount factor
    lmbda: float = 0.95              # GAE lambda

    # Optimization
    lr: float = 3e-4                 # Adam learning rate
    max_grad_norm: float = 0.5       # gradient clipping threshold

    # Network architecture — shared backbone for actor and critic
    network: dict = field(default_factory=lambda: {
        "architecture": "mlp",
        "hidden_sizes": [256, 256],
        "activation": "tanh",
        "layer_norm": False,
    })


class PPOAlgorithm(BaseAlgorithm):
    """PPO with clipped surrogate objective, GAE, and parallel environment collection."""

    def setup(self, env: EnvBase) -> None:
        from torchrl.modules import (
            NormalParamExtractor,
            ProbabilisticActor,
            TanhNormal,
            ValueOperator,
        )
        from torchrl.objectives import ClipPPOLoss
        from torchrl.objectives.value import GAE

        self.acfg = self._build_acfg(PPOConfig())
        acfg = self.acfg
        ecfg = self.cfg.environment

        obs_shape = tuple(ecfg.obs_shape)
        num_actions = int(ecfg.num_actions)

        # --- Actor backbone → outputs 2 * num_actions (mean + log_std) ---
        actor_net = make_network(acfg.network, obs_shape, num_actions * 2).to(self.device)
        actor_module = TensorDictModule(
            nn.Sequential(actor_net, NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        action_spec = env.action_spec
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=True,
        ).to(self.device)

        # --- Critic ---
        critic_net = make_network(acfg.network, obs_shape, 1).to(self.device)
        self.critic = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
        ).to(self.device)

        # --- Advantage module (GAE) ---
        self.gae = GAE(
            gamma=float(acfg.gamma),
            lmbda=float(acfg.lmbda),
            value_network=self.critic,
            average_gae=False,
        )

        # --- PPO loss ---
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=float(acfg.clip_epsilon),
            entropy_bonus=True,
            entropy_coeff=float(acfg.entropy_coef),
            critic_coeff=float(acfg.critic_coef),
            normalize_advantage=True,
            loss_critic_type="smooth_l1",
        ).to(self.device)

        # --- Single optimizer for actor + critic ---
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=float(acfg.lr),
        )

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        return self.actor  # stochastic actor is the exploration policy

    # ------------------------------------------------------------------
    # Collector configuration
    # ------------------------------------------------------------------

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=int(self.acfg.frames_per_batch),
            total_frames=int(self.cfg.trainer.total_frames),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Compute GAE, then multi-epoch minibatch PPO updates."""
        acfg = self.acfg
        epochs = int(acfg.epochs_per_batch)
        minibatch_size = int(acfg.minibatch_size)

        # Compute GAE advantages in-place
        with torch.no_grad():
            self.gae(batch)

        # Flatten (num_envs x time) → single batch dimension
        data = batch.reshape(-1)
        batch_size = data.batch_size[0]

        metrics: dict[str, float] = {}
        for _ in range(epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                idx = perm[start : start + minibatch_size]
                if len(idx) < 2:
                    continue
                mb = data[idx]

                loss_td = self.loss_module(mb)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    float(acfg.max_grad_norm),
                )
                self.optimizer.step()

                metrics = {
                    "loss/total": loss.item(),
                    "loss/policy": loss_td["loss_objective"].item(),
                    "loss/value": loss_td["loss_critic"].item(),
                    "loss/entropy": loss_td["loss_entropy"].item(),
                }

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict={
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.actor.load_state_dict(state.policy_state_dict["actor"])
        self.critic.load_state_dict(state.policy_state_dict["critic"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)

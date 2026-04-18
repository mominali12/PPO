"""Proximal Policy Optimization (PPO) algorithm.

Compatible environments: continuous-action dm_control environments (humanoid-walk).

Architecture:
  - Actor: MLP → Normal distribution via NormalParamExtractor + TanhNormal
  - Critic: separate MLP → scalar value estimate
  - Loss: ClipPPOLoss (clipped surrogate + value + entropy)
  - Advantage: GAE (Generalized Advantage Estimation)
  - Collector: SyncDataCollector over ParallelEnv (vectorised)
  - No replay buffer (on-policy)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from src.algorithms.base import BaseAlgorithm, TrainingState
from src.algorithms.utils import last_episode_return
from src.networks.factory import make_network
from src.trainer import TrainerEvent, fire_callbacks


@dataclass
class PPOConfig:
    """Hyperparameters for Proximal Policy Optimization (PPO) with clipped surrogate and GAE.

    Defaults are tuned for continuous-action dm_control environments (e.g. humanoid-walk).
    Override any field in the algorithm YAML or experiment config.
    """

    # Data collection
    frames_per_batch: int = 2_048    # total frames per rollout across all envs (num_envs × steps_per_env)

    # PPO update schedule
    epochs_per_batch: int = 10       # number of gradient passes over each collected batch
    minibatch_size: int = 64         # mini-batch size within each epoch

    # PPO loss coefficients
    clip_epsilon: float = 0.2        # clipping parameter ε for the surrogate objective
    entropy_coef: float = 0.01       # entropy bonus coefficient (encourages exploration)
    critic_coef: float = 0.5         # value loss coefficient relative to policy loss

    # Advantage estimation (GAE)
    gamma: float = 0.99              # discount factor
    lmbda: float = 0.95              # GAE lambda — trades off bias vs. variance in advantage estimates

    # Optimization
    lr: float = 3e-4                 # Adam learning rate (shared optimizer for actor and critic)
    max_grad_norm: float = 0.5       # gradient clipping threshold

    # Network architecture — shared backbone for actor and critic
    network: dict = field(default_factory=lambda: {
        "architecture": "mlp",
        "hidden_sizes": [256, 256],
        "activation": "tanh",
        "layer_norm": False,
    })


class PPOAlgorithm(BaseAlgorithm):
    """PPO with clipped surrogate objective, GAE, and parallel environment collection.

    Args:
        cfg: full Hydra config
        device: resolved torch.device
    """

    def setup(self) -> None:
        from torchrl.collectors import SyncDataCollector
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

        # --- Vectorised environment ---
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

        # --- Actor backbone → outputs 2 * num_actions (mean + log_std) ---
        actor_net = make_network(acfg.network, obs_shape, num_actions * 2).to(self.device)
        actor_module = TensorDictModule(
            nn.Sequential(actor_net, NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        action_spec = self.env.action_spec
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

        # --- Data collector ---
        self.collector = SyncDataCollector(
            create_env_fn=self.env,
            policy=self.actor,
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
        epochs = int(acfg.epochs_per_batch)
        minibatch_size = int(acfg.minibatch_size)
        log_every = int(trainer_cfg.log_every_n_steps)

        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            callbacks,
            state={"cfg": self.cfg},
        )

        metrics: dict[str, float] = {}
        for batch in self.collector:
            # Compute GAE advantages (in-place)
            with torch.no_grad():
                self.gae(batch)

            # Flatten (num_envs × time) → single batch dimension
            data = batch.reshape(-1)
            batch_size = data.batch_size[0]

            for _ in range(epochs):
                # Random minibatch permutation
                perm = torch.randperm(batch_size, device=self.device)
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start : start + minibatch_size]
                    if len(idx) < 2:
                        continue
                    mb = data[idx]
                    metrics = self._update(mb)

            self._step += batch.numel()

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
        loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            float(acfg.max_grad_norm),
        )
        self.optimizer.step()

        return {
            "loss/total": loss.item(),
            "loss/policy": loss_td["loss_objective"].item(),
            "loss/value": loss_td["loss_critic"].item(),
            "loss/entropy": loss_td["loss_entropy"].item(),
        }

    def eval(self, num_episodes: int) -> dict[str, float]:
        from torchrl.envs.utils import ExplorationType, set_exploration_type

        # Use a single-env copy for evaluation
        from src.environments.factory import make_env
        from omegaconf import OmegaConf

        ecfg = self.cfg.environment
        env_kwargs = OmegaConf.to_container(ecfg, resolve=True)
        env_kwargs.pop("_target_", None)
        env_kwargs["num_envs"] = 1
        eval_env = make_env(**env_kwargs, device=str(self.device))

        returns = []
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = eval_env.reset()
                episode_return = 0.0
                done = False
                while not done:
                    td = self.actor(td)
                    td = eval_env.step(td)
                    episode_return += td["next", "reward"].sum().item()
                    done = td["next", "done"].any().item() or td["next", "terminated"].any().item()
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

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=self._step,
            policy_state_dict={
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            optimizer_state_dict=self.optimizer.state_dict(),
            replay_buffer_state=None,
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.actor.load_state_dict(state.policy_state_dict["actor"])
        self.critic.load_state_dict(state.policy_state_dict["critic"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)

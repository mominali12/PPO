"""REINFORCE (Monte-Carlo Policy Gradient) algorithm.

Compatible environments: discrete-action Gym environments (e.g. CartPole-v1).

Architecture:
  - Policy: MLP → Categorical distribution (discrete actions)
  - No value baseline (vanilla REINFORCE)
  - Training: episode-by-episode rollouts via env.rollout()
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
class ReinforceConfig:
    """Hyperparameters for REINFORCE (Monte-Carlo Policy Gradient).

    Defaults are tuned for discrete-action environments like CartPole-v1.
    Override any field in the algorithm YAML or experiment config.
    """

    # Optimization
    lr: float = 1e-3             # Adam learning rate
    gamma: float = 0.99          # discount factor for Monte-Carlo returns
    max_grad_norm: float = 0.5   # gradient clipping threshold (inf = disabled)
    normalize_returns: bool = True  # standardize returns before policy gradient loss

    # Network architecture — MLP for discrete-action environments
    network: dict = field(default_factory=lambda: {
        "architecture": "mlp",
        "hidden_sizes": [64, 64],
        "activation": "tanh",
        "layer_norm": False,
    })


class ReinforceAlgorithm(BaseAlgorithm):
    """Vanilla REINFORCE with Monte-Carlo returns, episode-level rollouts.

    Args:
        cfg: full Hydra config
        device: resolved torch.device
    """

    def setup(self) -> None:
        """Build env, policy, and optimizer."""
        from torchrl.modules import ProbabilisticActor
        from torchrl.modules.distributions import OneHotCategorical

        self.acfg = self._build_acfg(ReinforceConfig())
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

        # --- Policy network ---
        policy_net = make_network(acfg.network, obs_shape, num_actions)
        policy_net = policy_net.to(self.device)

        policy_module = TensorDictModule(
            policy_net,
            in_keys=["observation"],
            out_keys=["logits"],
        )
        self.actor = ProbabilisticActor(
            module=policy_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=OneHotCategorical,
            return_log_prob=True,
        ).to(self.device)

        # --- Optimizer ---
        self.optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(acfg.lr),
        )

    def train(
        self,
        trainer_cfg: DictConfig,
        callbacks: list,
    ) -> dict[str, float]:
        acfg = self.acfg
        total_frames = int(trainer_cfg.total_frames)
        log_every = int(trainer_cfg.log_every_n_steps)

        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            callbacks,
            state={"cfg": self.cfg},
        )

        metrics: dict[str, float] = {}
        while self._step < total_frames:
            # Collect one episode; cap at remaining budget to respect total_frames
            remaining = total_frames - self._step
            rollout = self.env.rollout(
                max_steps=remaining,
                policy=self.actor,
                auto_reset=True,
            )

            rollout = self._compute_returns(rollout, gamma=float(acfg.gamma))
            metrics = self._update(rollout)

            prev_step = self._step
            self._step += rollout.batch_size[0]

            # Fire callbacks when we cross a log_every boundary
            if prev_step // log_every < self._step // log_every:
                metrics["reward/last"] = last_episode_return(rollout)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        fire_callbacks(TrainerEvent.ON_TRAIN_END, callbacks, state={"cfg": self.cfg})
        return metrics

    def _update(self, rollout: TensorDict) -> dict[str, float]:
        acfg = self.acfg

        returns = rollout.get("advantage").reshape(-1)       # [T]
        log_probs = rollout.get("action_log_prob").reshape(-1)  # [T]

        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), float(acfg.max_grad_norm))
        self.optimizer.step()

        return {"loss/policy": loss.item()}

    def eval(self, num_episodes: int) -> dict[str, float]:
        from torchrl.envs.utils import set_exploration_type, ExplorationType

        returns = []
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            for _ in range(num_episodes):
                td = self.env.reset()
                episode_return = 0.0
                done = False
                while not done:
                    td = self.actor(td)
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

    def _compute_returns(self, rollout: TensorDict, gamma: float) -> TensorDict:
        """Compute discounted Monte-Carlo returns and write them as 'advantage'."""
        rewards = rollout.get(("next", "reward")).reshape(-1)  # [T]
        T = rewards.shape[0]

        returns = torch.zeros(T, dtype=torch.float32, device=rewards.device)
        G = 0.0
        for t in reversed(range(T)):
            G = rewards[t].item() + gamma * G
            returns[t] = G

        if self.acfg.normalize_returns and T > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        rollout.set("advantage", returns)
        return rollout

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=self._step,
            policy_state_dict=self.actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            replay_buffer_state=None,
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)

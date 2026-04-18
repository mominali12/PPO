"""REINFORCE (Monte-Carlo Policy Gradient) algorithm.

Compatible environments: discrete-action Gym environments (e.g. CartPole-v1).

Architecture:
  - Policy: MLP → Categorical distribution (discrete actions)
  - No value baseline (vanilla REINFORCE)
  - Trainer: EpisodicTrainer (full episode rollouts)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, TrainingState
from src.networks.factory import make_network


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
    """Vanilla REINFORCE with Monte-Carlo returns, episode-level rollouts."""

    def setup(self, env: EnvBase) -> None:
        from torchrl.modules import ProbabilisticActor
        from torchrl.modules.distributions import OneHotCategorical

        self.acfg = self._build_acfg(ReinforceConfig())
        acfg = self.acfg
        ecfg = self.cfg.environment

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

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        return self.actor  # stochastic actor IS the exploration policy

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process a full episode: compute returns, then policy gradient update."""
        acfg = self.acfg

        batch = self._compute_returns(batch, gamma=float(acfg.gamma))

        returns = batch.get("advantage").reshape(-1)
        log_probs = batch.get("action_log_prob").reshape(-1)

        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), float(acfg.max_grad_norm))
        self.optimizer.step()

        return {"loss/policy": loss.item()}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_returns(self, rollout: TensorDict, gamma: float) -> TensorDict:
        """Compute discounted Monte-Carlo returns and write them as 'advantage'."""
        rewards = rollout.get(("next", "reward")).reshape(-1)
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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,  # Trainer sets the real step
            policy_state_dict=self.actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)

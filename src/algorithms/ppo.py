"""
Implementation of PPO in the structure of this repository following the Blogpost from S.Huang.
The defaults are defined such that they match the implementation at:
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
The Implementation details from the blog post are (the implementation details are referred to below in the code):

[1]: Vectorized environments: BaseTrainer owns this (setup by Environment). Configured via "num_envs" in the trainer
    yaml. During rollout "fixed trajectory segments" in these environments are collected (including automatic resets for
    terminated env). During Learning PPO learns from this data.
[2]: Orthogonal initialization of weights and constant initialization of biases. Realized in the factory (networks.py)
    by the helper _layer_init_mlp_ppo()/ ADD ATARI ONE HERE.
[3]: Adams epsilon parameter (small parameter in denominator for numerical stability) set to 1e-5 (instead of 1e-8,
    which is PyTorch default)
[4]: Learning rate annealing. For Atari a linear decay from 2.5e-4 -> 0 is implemented.
[5]: Generalized Advantage Estimation. The advantage measures: "how much better was this action than the critic expected?"
    A_t = return_t - V(s_t) for timestep t.
    Then, during learning, the probability of this action is increased if the advantag is >0 and decreased if <0.
    Return can be estimated in different ways:
        - One-step TD advantage:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) -> low variance but bias
        - Monte Carlo advantage (with reward discount gamma):
            A_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ... - V(s_t) -> less bias high var.
        - NOTE: if episode is done, there is no next state/value. Therefore, the binary done=1/0 removes them.
        - GAE interpolated between them with gae_lambda (bias/variance tradeoff parameter):
            - Compute TD error: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            - Accumulate backwards:
                A_t = delta_t + gamma * gae_lambda * (1 -done_t) * A_{t+1}
    With gae_lambda = 0 only one-step TD errors are used, for gae_lambda = 1 close to MC returns.
    For PPO lambda = 0.95. All this is included in the self.adv_module here, where the ppo_monolith.py
    explicitly calculated this.
[6]: PPO collects one rollout batch (so N_envs * N_steps steps), then reuses that data for several gradient updates.
    This is better sample efficiency. PPO optimizes over several epochs over the same rollout. The minibatches are
    created from shuffled indices. This reduces the correleation minibatch gradients.
    PPO is still on-policy. The Buffer just holds the current rollout.
[7]: Normalization of advantages on the minibatch level.
[8]: Clipped surrogate objective. Main PPO idea: prevent the policy from changing too much from one update to the next,
    even though PPO reuses same rollout for many epochs. The clipped objective limits limits the policy drift
    without requiring a complicated trust-region optimizer.
[9]: Value Loss Clipping. Before the actor is clipped so the policy does not change too much.
    The critic can also change too much. Value loss clipping limits how far the new value predic
"""
from typing import Callable
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase

from torch.distributions import Categorical # instead of OneHotCategorical
from torchrl.envs import EnvBase
import torch
import torch.nn as nn
from torchrl.data import TensorSpec, TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.envs.utils import ExplorationType
from torchrl.objectives.value import GAE
import functools
from src.algorithms.base import BaseAlgorithm, TrainingState, CollectorConfig

from src.networks import make_mlp_ppo_actor, make_mlp_ppo_critic

class PPO(BaseAlgorithm):
    """

    """
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        actor_network: Callable[[tuple[int, ...], int], nn.Module] | None = None,
        critic_network: Callable[[tuple[int, ...], int], nn.Module] | None = None,
        actor_critic_network: Callable[[tuple[int, ...], int], tuple[nn.Module, nn.Module]] | None = None,
        obs_key: str = "observation",
        # ------ Hyperparameters -------
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_epochs: int = 4,
        num_minibatches: int = 4,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        anneal_lr: bool = True,
        target_kl: float | None = None,
        frames_per_batch: int = 512, # corresponds to num_steps * num_envs from monolith
        max_frames_per_traj: int = -1,
        total_frames: int = 25_000,
    ) -> None:
        """
        Initializer that stores only hyperparameters and factories.

        Args:
            device: Passed to abstract base class
            actor_network: configured in config.yml partially. The number and size of hidden layers is prefilled in the
                provided Callable. If None these arguments will be assigned defaults. The size of action and obs.-space
                depends on the environment and the args will be provided in setup().
            critic_network: configured in config.yml partially. Follows the same call signature as the actor factory.
            obs_key: Key for the observation tensor in the TensorDict.
        """
        super().__init__(device)

        # ======= Used for e.g. Cartpole (mlp networks)=====
        # Default for the actor network
        if actor_network is None:
            self._actor_factory = functools.partial(make_mlp_ppo_actor, num_cells=[64,64],activation_class=nn.Tanh)
        else:
            self._actor_factory = actor_network

        # Default for the critic network
        if critic_network is None:
            self._critic_factory = functools.partial(make_mlp_ppo_critic, num_cells=[64,64],activation_class=nn.Tanh)
        else:
            self._critic_factory = critic_network

        # ======== Used for e.g. Atari Breakout (cnn networks)=====
        self._actor_critic_factory = actor_critic_network

        # The key under which observations / for Atari Games e.g. Pixels are stored in TensorDicts or torchRL
        # environments, respectively.
        self._obs_key: str = obs_key

        # Store hyperparameters
        self.lr = lr # Default 2.5e-4 (see Implementation Detail [4]).
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.num_minibatches = num_minibatches
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.anneal_lr = anneal_lr
        self.target_kl = target_kl
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self.total_frames = total_frames

        # Iteration counter for Implementation detail [4]
        self._num_updates_done = 0


    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        """
        Setup of the algorithm set by the trainer. Note the BaseTrainer passes the environment creating make_env Callable.

        Args:
            make_env: creates a temporary environment to inspect its properties to correctly configure the algorithm.
                The real training env is created and owned by the Trainer (see setup() of BaseTrainer).

        Returns:
            The environment of type EnvBase (torch-RL type; this environment expects TensorDicts).

        """
        temporary_env: EnvBase = make_env()
        # Shape of the observation space, for Atari this will be the X x Y sized pixel
        obs_spec = temporary_env.observation_spec[self._obs_key]
        observation_shape: tuple[int, ...] = tuple(
            obs_spec.shape[len(temporary_env.batch_size):]
        )
        # Specifications of action space (shape, dtype, bounds, discrete/continuous)
        action_spec: TensorSpec = temporary_env.action_spec
        # Number of actions;
        num_actions: int = int(action_spec.space.n)

        # Create the actor and the critic wrapped in TensorDictModules (pass them the env-specific shapes).
        # This allows that the network reads from one Tensor in the TensorDict (specified by in_keys) and
        # outputs into another (out_keys).

        # .... Actor
        if self._actor_critic_factory is not None:
            # Branch for CNN based Envs (atari)
            actor_net, critic_net = self._actor_critic_factory(
                observation_shape,
                num_actions,
            )
            actor_net = actor_net.to(self.device)
            critic_net = critic_net.to(self.device)
        else:
            # Branch for MLP based envs (e.g. cartpole)
            actor_net: nn.Module = self._actor_factory(observation_shape, num_actions).to(self.device)
            critic_net: nn.Module = self._critic_factory(observation_shape, num_actions).to(self.device)

        self.actor_module: TensorDictModule = TensorDictModule(
            actor_net,
            in_keys=[self._obs_key],
            out_keys=["logits"],
        )

        # In the monolithic implementation and with the use of plain torch the actor does:
        # logits = self.actor(x)
        # probs = Categorical(logits=logits)
        # action = probs.sample()
        # log_prob = probs.log_prob(action)
        # entropy = probs.entropy() # -> this is done in the LOSS part
        # Using torchRL we wrap the actor module in a ProbabilisticActor that transforms the logits into
        # probabilities, samples actions (key: action in td), writes log-probs (key: sample_log_prob):
        self.actor = ProbabilisticActor(
            module=self.actor_module,
            spec=action_spec,
            in_keys=["logits"], # input for calculating the probabilities
            distribution_class=Categorical, # instead of one hot encoding (actions are integer class indices)
            return_log_prob=True,
            log_prob_key="sample_log_prob",
            default_interaction_type=ExplorationType.RANDOM
        ).to(self.device)

        # .... Critic

        # The TensorDict wrapper for a value function is given by ValueOperator. It reads observation and
        # writes the state value under the standard key "state_value" (needed by GAE and PPO losses)
        self.critic = ValueOperator(
            module=critic_net,
            in_keys=[self._obs_key],
        ).to(self.device)

        # .... GAE - advantage module
        # Implementation Detail [5]. Note set average_gae to False to avoid normalization within this
        # module. This is later done in Implementation Detail [XX].
        # Calling self.adv_module(batch) represents the full reverse-time loop in ppo_monolith.py.
        self.adv_module: TensorDictModuleBase = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=False, # avoid normalization here.
            device=self.device,
        )

        # .... On-Policy minibatch buffer:
        mini_batch_size: int = self.frames_per_batch // self.num_minibatches

        # Implementation Detail [6]. Initialize the buffer for the data of one rollout. Sampling
        # from this produces randomly shuffled data in minibatches.
        self.data_buffer = TensorDictReplayBuffer(
            # lazy means allocation on first insert, capacity is exactly one PPO rollout
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            # This sampler matches the monolith's minibatch shuffling logic. Each transition appears
            # once per epoch, in random minibatch order.
            sampler=SamplerWithoutReplacement(),
            batch_size=mini_batch_size,
        )

        # .... Optimizer
        # Do not duplicate parameters for actor and critic (for CNN case they share some)
        params = list({id(p): p for p in list(self.actor.parameters()) + list(self.critic.parameters())}.values())

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
            params, # extend actor parameters by critic parameters
            lr=self.lr,
            # Implementation Detail [3]: the value of the eps-parameter for Adam.
            eps=1e-5,)
        self._optim_params = params




    def get_policy(self) -> TensorDictModule:
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        return self.actor

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    def step(self, batch: TensorDict) -> dict[str, float]:
        """
        One PPO iteration: GAE -> repeated minibatch PPO updates

        Mirrows ppo_monolith.py:
            - compute advantages / returns from the rollout
            - recompute current-policy log-probs for old actions
            - clipped policy loss
            - clipped value loss
            - entropy bonus
            - global grad clipping
        """
        # Implementation Detail [4]: Linear decay of the learning rate.
        if self.anneal_lr:
            num_updates = max(1, self.total_frames // self.frames_per_batch)
            frac = 1.0 - self._num_updates_done / num_updates
            lr_now = frac * self.lr
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_now

        batch: TensorDict = batch.to(self.device)
        # Compte GAE first -> then flatten for minibatches

        # Compute fixed old values, advantages, and value targets before updating.
        # GAE writes "advantage", "value_target", and "state_value" in the tensordict batch
        with torch.no_grad():
            # This includes the full reverse "time" loop in ppo_monolith.py
            batch = self.adv_module(batch)

        # flatten the batch (each tensor in the dict): go from [T,N,...] -> [T*N,...] with T frames per env and N
        # number of envs
        batch = batch.reshape(-1)


        # Detach rollout tensors. PPO treats collected log-probs, values, returns,
        # and advantages as fixed targets during the update. They are removed from the autograd
        # graph
        batch = batch.detach()

        # Empty and refill the buffer on every update. Load one rollout:
        self.data_buffer.empty()
        self.data_buffer.extend(batch)

        policy_losses: list[torch.Tensor] = []
        value_losses: list[torch.Tensor] = []
        entropy_losses: list[torch.Tensor] = []
        approx_kls: list[torch.Tensor] = []
        clipfracs: list[torch.Tensor] = []

        stop_early: bool = False

        for _epoch in range(self.num_epochs):
            # Implementation Detail [6]: The TensorDictReplayBuffer was initialized with SamplerWithoutReplacement()
            # This means that the minibatches are sampled randomly and samples are not repeated.
            for mb in self.data_buffer:
                mb = mb.to(self.device)

                # Current policy distribution for the old collected observations instead of
                # calling self.actor(mb) (this would sample a new action)
                policy_td: TensorDict = self.actor_module(mb.select(self._obs_key))
                # Probability distribution
                dist = Categorical(logits=policy_td["logits"])

                action = mb["action"].squeeze(-1).long() # [B, 1] -> [B]
                new_log_prob = dist.log_prob(action)
                entropy = dist.entropy()

                old_log_prob = mb["sample_log_prob"].squeeze(-1)

                logratio = new_log_prob - old_log_prob
                # Important for Implementation Detail [8] below
                # ratio = pi_new(action | state) / pi_old(action | state)
                # ratio = 1: new policy gives same prob. as old policy
                # ratio = 1.2: new policy makes action 20% more likely
                # ratio = 0.7: new policy makes action 30% less likely
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_coef).float().mean()

                advantages = mb["advantage"].squeeze(-1) # arrives as [B, 1] we want [B]

                # Implementation Detail [7]: Advantage normalization on the minibatch data.
                if self.norm_adv:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # Implementation Detail [8]: Clipped surrogate policy objective. Implemented exactly as in monolith.
                # The ratio is limited to [0.8,1.2] (default values)
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio,
                    1.0 - self.clip_coef,
                    1.0 + self.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() # conservative

                # Current critic value for the old observations
                value_td = self.critic(mb.select(self._obs_key))
                new_value = value_td["state_value"].squeeze(-1)

                returns = mb["value_target"].squeeze(-1)
                old_value = mb["state_value"].squeeze(-1)

                old_value = old_value.detach()

                # CleanRL-style clipped value loss.
                if self.clip_vloss:
                    v_loss_unclipped = (new_value - returns) ** 2
                    v_clipped = old_value + torch.clamp(
                        new_value - old_value,
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((new_value - returns) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss


                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self._optim_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                policy_losses.append(pg_loss.detach())
                value_losses.append(v_loss.detach())
                entropy_losses.append(entropy_loss.detach())
                approx_kls.append(approx_kl.detach())
                clipfracs.append(clipfrac.detach())

                if self.target_kl is not None and approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break
        with torch.no_grad():
            values = batch["state_value"]
            returns = batch["value_target"]
            explained_var = self._explained_variance(values, returns)

        self._num_updates_done += 1  # update iteration counter
        return {
            "train/policy_loss": torch.stack(policy_losses).mean().item(),
            "train/value_loss": torch.stack(value_losses).mean().item(),
            "train/entropy": torch.stack(entropy_losses).mean().item(),
            "train/approx_kl": torch.stack(approx_kls).mean().item(),
            "train/clipfrac": torch.stack(clipfracs).mean().item(),
            "train/explained_variance": explained_var,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }

    @staticmethod
    def _explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        y_pred = y_pred.detach().reshape(-1)
        y_true = y_true.detach().reshape(-1)
        var_y = torch.var(y_true)
        if var_y == 0:
            return float("nan")
        return (1.0 - torch.var(y_true - y_pred) / var_y).item()


    def _get_training_state(self) -> TrainingState:
        """
        For writing checkpoint
        """
        return TrainingState(
            step=0,
            policy_state_dict={
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"num_updates_done": self._num_updates_done},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        """
        For loading checkpoint
        """
        self.actor.load_state_dict(state.policy_state_dict["actor"])
        self.critic.load_state_dict(state.policy_state_dict["critic"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra is not None:
            self._num_updates_done = int(state.extra.get("num_updates_done", 0))

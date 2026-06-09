"""
This module is currently a one-file implementation of Huang's 2022 Blog Post and video series. Will be refactored
later to fit into the structure of this repo.
"""
import argparse
import os
import time
import numpy as np
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch._C import device
from torch.distributions.categorical import Categorical
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from torchrl.data.llm import reward
from torchrl.objectives.value import advantages


def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, video_folder="videos")#, episode_trigger=lambda t: t % 100 == 0)
        # Deprecated in modern gymnasium versions. Use envs.reset(seed)
        # env.seed(seed)
        # env.action_space.seed(seed)
        # env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std = np.sqrt(2), bias_const = 0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(),64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,1),std=1)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,64)),
            nn.Tanh(),
            layer_init(nn.Linear(64,envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name',type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment.")
    parser.add_argument('--gym-id', type=str, default='CartPole-v1',help="the id of the gym env.")
    parser.add_argument('--learning-rate',type=float, default=2.5e-4, help="learning rate of optimizer")
    parser.add_argument("--seed",type=int,default=1,help="seed of experiment")
    parser.add_argument("--total-timesteps",type=int,default=25000, help="total timesteps of exp.")
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda',type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?',const=True, help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track',type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?',const=True, help='if toggled, the experiment is tracked with wandb')
    parser.add_argument('--wandb-project-name',type=str, default='CleanRL',help="wandb project name")
    parser.add_argument('--wandb-entity',type=str,default=None,help='the entity (team) of wandbs project')
    parser.add_argument('--capture-video',type=lambda x:bool(strtobool(x)), default=False,
                        nargs='?',const=True,help='if video shall be recorded.')

    # Algorithm specific args
    parser.add_argument('--n-envs',type=int,default=4, help='number of environments')
    parser.add_argument('--num-steps',type=int,default=128, help='the number of steps to run in each env per policy rollout')
    parser.add_argument('--anneal_lr',type=lambda  x:bool(strtobool(x)),default=True,nargs='?',const=True, help="annealing of the learning rate")
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True, help="gae is enabled by default")
    parser.add_argument('--gamma',type=float,default=0.99,help='discount factor')
    parser.add_argument('--gae-lambda',type=float,default=0.95,help='lambda parameter')
    parser.add_argument('--num-minibatches',type=int, default=4,help='the number of minibatches')
    parser.add_argument('--update-epochs',type=int,default=4,help='the K epochs to update the policy')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)),default=True,nargs='?',const=True, help="Toggle advantage normalization or not")
    parser.add_argument('--clip-coef',type=float, default=0.2,help="the surrogate clipping coefficient")
    parser.add_argument('--clip-vloss',type=lambda x:bool(strtobool(x)), default=True,nargs='?',const=True, help="Toggle clip value loss or not")
    parser.add_argument('--ent-coef',type=float, default=0.01,help="the entropy regularization coefficient for actor")
    parser.add_argument('--vf-coef',type=float, default=0.5,help="the value function coefficient for critic")
    parser.add_argument('--max-grad-norm',type=float,default=0.5,help="the max norm of the gradient")
    parser.add_argument('--target-kl',type=float, default=None, help='the target KL divergence threshold') # 0.015 is default value in OpenAI spinning up
    args = parser.parse_args()
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

if __name__=="__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparamers",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )

    # Try not to modify SEEDING
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id,seed=args.seed + i,idx=i,capture_video=args.capture_video,
                                              run_name=run_name ) for i in range(args.n_envs)])
    assert isinstance(envs.single_action_space,gym.spaces.Discrete), "only discrete Action Spaces supported."

    agent = Agent(envs=envs).to(device)
    optimizer = optim.Adam(agent.parameters(),lr=args.learning_rate,eps=1e-5)

    # Setup storage
    obs = torch.zeros((args.num_steps, args.n_envs) + envs.single_observation_space.shape).to(device) # use tuple addition -> eg. (num_steps, n_envs, 4)
    actions = torch.zeros((args.num_steps, args.n_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.n_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.n_envs)).to(device)
    truncations = torch.zeros((args.num_steps, args.n_envs)).to(device)
    terminations = torch.zeros((args.num_steps, args.n_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.n_envs)).to(device)
    values = torch.zeros((args.num_steps, args.n_envs)).to(device)

    # DO not modify
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset(seed=[args.seed + i for i in range(args.n_envs)])[0]).to(device)
    next_done = torch.zeros(args.n_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the learning rate if instructed to do so
        if args.anneal_lr:
            # linearly decrease the learning rate from 1 to 0.
            frac = 1.0 - (update - 1) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]['lr'] = lrnow

            # Policy loop
            for step in range(0, args.num_steps):
                # increment global step counter
                global_step += 1 * args.n_envs
                obs[step] = next_obs
                dones[step] = next_done

                # Algorithmic Logic: action logic
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(x=next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = log_prob

                # Do not modify: execute the game and log data
                next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
                done = np.logical_or(terminated,truncated)
                rewards[step] = torch.as_tensor(reward, device=device).view(-1)
                next_obs = torch.as_tensor(next_obs,dtype=torch.float32, device=device)
                next_done = torch.as_tensor(done,dtype=torch.float32, device=device).view(-1)

                # Output writing
                if "episode" in info:
                    episode_info = info["episode"]
                    mask = info.get("_episode", np.ones_like(episode_info["r"],dtype=bool))

                    for episodic_return, episodic_length in zip(
                        episode_info["r"][mask],
                        episode_info["l"][mask]
                    ):
                        print(f"global step: {global_step}",end="")
                        print(f" episodic return: {episodic_return}")
                        if args.track:
                            wandb.log(
                                {
                                    "charts/episodic_return": float(episodic_return),
                                    "charts/episodic_length": float(episodic_length),
                                },
                                step=global_step,
                            )

            # 5th
            # After rollout has been done
            # bootstrap reward if not done
            with torch.no_grad():
                # To compute advantages/returns one needs the value of the next state after the final collected step
                # for bootstrapping
                next_value = agent.get_value(x=next_obs).reshape(1, -1) # usually (4,1) -> (1,4) (to match the advantage loop style)
                if args.gae:
                    # Implementation Detail [5]: Generalized Advantage Estimation
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    # loop backwards through time: neccessary because advantage at time t depends on advantage at time t+1
                    for t in reversed(range(0, args.num_steps)):
                        if t == args.num_steps - 1:
                            # final rollout step
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value # use value from above for bootstrapping
                        else:
                            # all the other earlier steps
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        # one step TD error
                        # actual reward + discounted next values - predicted current value
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        # GAE smooths the deltas over time
                        # lastgaelam carries the future advantage backward through the rollout
                        advantages[t] = lastgaelam = (
                                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                        )
                    # returns = advantages + values, because advantage = return - value
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(0, args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = values[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and the value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                # Implementation Detail [6]: minibatches with randomly shuffled data.
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds],
                                                                                    b_actions.long()[mb_inds])
                    # ratio of logprobs (new) and the old logprobs from policy rollout
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    # Debug variables
                    with torch.no_grad():
                        # calculate approx kullback leiber
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio -1)-logratio).mean()
                        clipfracs += [((ratio -1.0).abs() > args.clip_coef).float().mean()]

                    # Implementation detail [7]: advantage normalization
                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Implementation detail [8]: clip coefficient (papers objective as min of positives)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1+ args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # 9th: value loss clipping
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue -b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        # usually: mean squared error of predicted values and empirical returns
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # 10th entropy loss
                    entropy_loss = entropy.mean()
                    # minimize policy loss and value loss but maximize entropy loss -> encourage exploration
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    # 11th global gradient clipping
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                # early stopping at batch level
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            # explained variance
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # NOT MODIDYY: record rewards for plotting resources
            if args.track:
                wandb.log(
                    {
                        "charts/learning_rate": optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/policy_loss": pg_loss.item(),
                        "losses/entropy_loss": entropy_loss.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                        "losses/explained_variance": explained_var,
                        "charts/SPS": int(global_step / (time.time() - start_time)),
                    },
                    step=global_step,
                )

    envs.close()

    if args.track and args.capture_video:
        for video_path in glob.glob(f"videos/*.mp4"):
            wandb.log({"videos": wandb.Video(video_path, format="mp4")})
    writer.close()
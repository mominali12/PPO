"""
This module is currently a one-file implementation of Huang's 2022 Blog Post and video series. Will be refactored
later to fit into the structure of this repo.
"""
import argparse
import os
import time
import numpy as np
import random
import torch
from distutils.util import strtobool
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--track',type=lambda x:bool(strtobool(x)), default=True,
                        nargs='?',const=True, help='if toggled, the experiment is tracked with wandb')
    parser.add_argument('--wandb-project-name',type=str, default='CleanRL',help="wandb project name")
    parser.add_argument('--wandb-entity',type=str,default=None,help='the entity (team) of wandbs project')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
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

    device = torch.device("cude" if torch.cuda.is_available() and args.cuda else "cpu")

    for i in range(100):
        writer.add_scalar("test_loss",i*2,global_step=i)

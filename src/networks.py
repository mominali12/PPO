"""Network factories used by ``configs/algorithm/*.yaml``.

Each factory takes ``(obs_shape, action_dim)`` positionally and keeps the
rest as keyword-only args, so a Hydra ``_partial_`` config can pre-bind the
kwargs while the algorithm's ``setup()`` supplies the runtime shape and
action count. ``action_dim`` is the discrete action count for value-based
algorithms (DQN) and the continuous action vector size for actor/critic
algorithms (DDPG).
"""
from __future__ import annotations

import math
from typing import Sequence, Type
import numpy as np
import torch
import torch.nn as nn
from torchrl.modules import ConvNet, MLP


def _layer_init_mlp_ppo(module: nn.Module,*,final_weight_std: float,
                        hidden_weight_std: float = math.sqrt(2),bias_const: float = 0.0) -> nn.Module:
    """
    Implementation detail [2] of PPO. Orthogonal initialization of weights and constant initialization of biases.
    """
    linear_layers = [m for m in module.modules() if isinstance(m, nn.Linear)]

    for layer in linear_layers[:-1]:
        nn.init.orthogonal_(layer.weight, hidden_weight_std)
        nn.init.constant_(layer.bias, bias_const)

    if linear_layers:
        final_layer = linear_layers[-1]
        nn.init.orthogonal_(final_layer.weight, final_weight_std)
        nn.init.constant_(final_layer.bias, bias_const)

    return module

def make_mlp_ppo_actor(
        obs_shape: Sequence[int],
        num_actions: int,
        *, # all arguments after this must be passed by keyword
        num_cells: Sequence[int],
        activation_class: Type[nn.Module] = nn.Tanh,
) -> nn.Module:
    """
    Creates the multilayer perceptron (MLP) for the actor of PPO. MLP is the torchRL substitute of torch's
    nn.Sequential(...). However, the num_cells attribute makes it easier to add more layers without extra code.

    Args:
        obs_shape: the shape of a single observation space of the vect. environment.
        num_actions: the number of actions (single_action_space.n).
        num_cells: the length of the sequence is the number of hidden layers and the values of the sequence
            are the number of cells for the layers.
        activation_class: the type of activation function. Default tanh for PPO implementation.
    Returns:
        the actor network, with orthogonal initialization of weights and constant initialization of biases.
    """
    mlp: nn.Module = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=num_actions,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return _layer_init_mlp_ppo(module=mlp, final_weight_std=0.01)


def make_mlp_ppo_critic(
        obs_shape: Sequence[int],
        num_actions: int,
        *, # all arguments after this must be passed by keyword
        num_cells: Sequence[int],
        activation_class: Type[nn.Module] = nn.Tanh,
) -> nn.Module:
    """
    Creates the multilayer perceptron (MLP) for the critic of PPO.

    Args:
        obs_shape: the shape of a single observation space of the vect. environment.
        num_cells: the length of the sequence is the number of hidden layers and the values of the sequence
            are the number of cells for the layers.
        activation_class: the type of activation function. Default tanh for PPO implementation.
    Returns:
        the critic network, mapping observations to a single scalar V(s).
    """
    del num_actions
    mlp: nn.Module = MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )
    return _layer_init_mlp_ppo(module=mlp, final_weight_std=1.0)


def NaturePPOActor(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    num_cells_mlp: Sequence[int] = (512,),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Nature-DQN style CNN policy head for pixel-observation PPO.

    Maps stacked Atari frames ``obs_shape`` to categorical action logits.
    """
    return _nature_cnn_mlp(
        obs_shape=obs_shape,
        out_features=num_actions,
        num_cells_cnn=num_cells_cnn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_cells_mlp=num_cells_mlp,
        activation_class=activation_class,
    )


def NaturePPOCritic(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    num_cells_mlp: Sequence[int] = (512,),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """Nature-DQN style CNN value head for pixel-observation PPO."""
    del num_actions  # signature parity with the actor factory
    return _nature_cnn_mlp(
        obs_shape=obs_shape,
        out_features=1,
        num_cells_cnn=num_cells_cnn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_cells_mlp=num_cells_mlp,
        activation_class=activation_class,
    )


def make_mlp_q_net(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """Plain MLP Q-network. Flattens ``obs_shape`` to ``in_features``."""
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=num_actions,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG deterministic actor.

    Returns an MLP mapping the flattened observation to ``action_dim``
    unbounded outputs. The algorithm wraps this with ``TanhModule`` to
    rescale to the action spec, so this factory must NOT apply tanh itself.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=action_dim,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_ddpg_critic(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for a DDPG state-action value (critic).

    Returns an MLP mapping the concatenated ``[obs, action]`` vector to a
    single Q-value. ``ValueOperator`` concatenates inputs along the last
    dim before calling the module.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)) + int(action_dim),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_actor(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C stochastic actor.

    Returns an MLP mapping the flattened observation to ``2 * action_dim``
    outputs. The algorithm chains it with ``NormalParamExtractor`` to split
    the output into ``loc`` and (positive) ``scale`` for a TanhNormal policy.
    """
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=2 * int(action_dim),
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def make_mlp_a2c_value(
    obs_shape: Sequence[int],
    action_dim: int,
    *,
    num_cells: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """MLP body for an A2C state-value critic.

    Takes ``(obs_shape, action_dim)`` for signature parity with the actor
    factory; ``action_dim`` is unused — the critic estimates V(s) only.
    Returns an MLP mapping the flattened observation to a single value.
    """
    del action_dim  # signature parity with actor factory
    return MLP(
        in_features=int(math.prod(obs_shape)),
        out_features=1,
        num_cells=list(num_cells),
        activation_class=activation_class,
    )


def NatureDQN(
    obs_shape: Sequence[int],
    num_actions: int,
    *,
    num_cells_cnn: Sequence[int] = (32, 64, 64),
    kernel_sizes: Sequence[int] = (8, 4, 3),
    strides: Sequence[int] = (4, 2, 1),
    num_cells_mlp: Sequence[int] = (512,),
    activation_class: Type[nn.Module] = nn.ReLU,
) -> nn.Module:
    """ConvNet -> MLP Q-network from Mnih et al. 2015 (\"Nature DQN\")."""
    return _nature_cnn_mlp(
        obs_shape=obs_shape,
        out_features=num_actions,
        num_cells_cnn=num_cells_cnn,
        kernel_sizes=kernel_sizes,
        strides=strides,
        num_cells_mlp=num_cells_mlp,
        activation_class=activation_class,
    )


def _nature_cnn_mlp(
    *,
    obs_shape: Sequence[int],
    out_features: int,
    num_cells_cnn: Sequence[int],
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    num_cells_mlp: Sequence[int],
    activation_class: Type[nn.Module],
) -> nn.Module:
    """Shared ConvNet -> MLP builder for Atari-style pixel observations."""
    cnn = ConvNet(
        activation_class=activation_class,
        num_cells=list(num_cells_cnn),
        kernel_sizes=list(kernel_sizes),
        strides=list(strides),
    )
    with torch.no_grad():
        cnn_out = cnn(torch.zeros(1, *obs_shape))
    mlp = MLP(
        in_features=cnn_out.shape[-1],
        out_features=out_features,
        num_cells=list(num_cells_mlp),
        activation_class=activation_class,
    )
    return nn.Sequential(cnn, mlp)

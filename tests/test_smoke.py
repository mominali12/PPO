"""Smoke test: one full training cycle of DQN on CartPole.

Loads the experiment config, applies minimal-frame overrides so the run
finishes in a few seconds, and asserts that ``_train()`` returns a non-empty
metrics dict without raising.

Run with:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


BASE_OVERRIDES = [
    "logger=[]",
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "checkpoint.save_dir=/tmp/hydra_smoke_tests/checkpoints",
    "checkpoint.save_last=false",
    "checkpoint.save_every_n_steps=999999999",
    "hydra.run.dir=/tmp/hydra_smoke_tests",
]


def _dqn_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
    ]

def _ppo_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=1024",
        "trainer.num_envs=4",
        "trainer.log_every_n_steps=256",
        "algorithm.frames_per_batch=256",
        "algorithm.total_frames=1024",
        "algorithm.num_epochs=2",
        "algorithm.num_minibatches=4",
    ]

def test_smoke_ppo_cartpole():
    cfg = load_experiment_cfg("ppo/cartpole", _ppo_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ppo_breakout_overrides() -> list[str]:
    # Tiny pixel-observation PPO run: just enough to verify CNN actor/critic,
    # categorical Atari actions, GAE, and the manual PPO minibatch path.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=128",
        "trainer.num_envs=1",
        "trainer.log_every_n_steps=64",
        "trainer.accelerator=cpu",
        "algorithm.frames_per_batch=64",
        "algorithm.total_frames=128",
        "algorithm.num_epochs=1",
        "algorithm.num_minibatches=2",
    ]


def test_smoke_ppo_breakout():
    """PPO on ALE/Breakout-v5: pixel obs, CNN actor/critic, categorical actions."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("ppo/breakout", _ppo_breakout_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_smoke_dqn_cartpole():
    """DQN on CartPole-v1: discrete actions, MLP Q-network, replay buffer."""
    cfg = load_experiment_cfg("dqn/cartpole", _dqn_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _dqn_pong_overrides() -> list[str]:
    # Same shape as the cartpole overrides: 600 frames in 100-frame batches,
    # init_random_frames=100 so we hit the gradient path. Shrinks the 1M
    # replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_dqn_pong():
    """DQN on ALE/Pong-v5: pixel obs, NatureDQN CNN, eval-env split."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("dqn/pong", _dqn_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ddpg_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    # Shrink the 1M replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.replay_buffer.storage.max_size=500",
        "algorithm.exploration_noise.annealing_num_steps=600",
    ]


def test_smoke_ddpg_halfcheetah():
    """DDPG on HalfCheetah-v4: continuous actions, MLP actor/critic, OU noise."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("ddpg/halfcheetah", _ddpg_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _a2c_overrides() -> list[str]:
    # 600 frames in 120-frame rollouts: 5 collections, 6 mini-batches each
    # (mini_batch_size=20). On-policy: no replay buffer, no warm-up.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=120",
        "algorithm.mini_batch_size=20",
    ]


def test_smoke_a2c_halfcheetah():
    """A2C on HalfCheetah-v4: continuous actions, stochastic actor + GAE."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("a2c/halfcheetah", _a2c_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0

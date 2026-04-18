"""Smoke tests: one full update cycle for every defined experiment.

Each test loads the experiment config, applies minimal-run overrides,
and calls _train(). The test passes if no exception is raised and
the returned metrics dict is non-empty.

Run with:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


# ---------------------------------------------------------------------------
# Common overrides that apply to every experiment
# ---------------------------------------------------------------------------
BASE_OVERRIDES = [
    "logger=[]",                          # no logging during tests
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "checkpoint.save_dir=/tmp/hydra_smoke_tests/checkpoints",
    "checkpoint.save_last=false",
    "checkpoint.save_every_n_steps=999999999",
    "hydra.run.dir=/tmp/hydra_smoke_tests",
]


def _reinforce_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=2000",
    ]


def _dqn_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=400",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=0",
        "algorithm.replay_buffer.capacity=400",
        "algorithm.replay_buffer.batch_size=32",
        "algorithm.eps_annealing_frames=200",
    ]


def _ppo_overrides() -> list[str]:
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=256",
        "trainer.num_envs=1",
        "algorithm.frames_per_batch=64",
        "algorithm.epochs_per_batch=1",
        "algorithm.minibatch_size=32",
        "environment.normalize_obs=false",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_smoke_reinforce_cartpole():
    """REINFORCE on CartPole: discrete actions, MLP policy."""
    cfg = load_experiment_cfg("reinforce/cartpole", _reinforce_overrides())
    from src.train import _train
    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_smoke_dqn_cartpole():
    """DQN on CartPole: discrete actions, MLP Q-network, replay buffer."""
    cfg = load_experiment_cfg("dqn/cartpole", _dqn_overrides())
    from src.train import _train
    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_smoke_dqn_atari_breakout():
    """DQN on Atari Breakout: pixel obs, CNN Q-network, frame stacking."""
    cfg = load_experiment_cfg("dqn/atari_breakout", _dqn_overrides())
    from src.train import _train
    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_smoke_ppo_dmc_humanoid():
    """PPO on DMControl humanoid: continuous actions, MLP actor-critic, GAE."""
    cfg = load_experiment_cfg("ppo/dmc_humanoid", _ppo_overrides())
    from src.train import _train
    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0

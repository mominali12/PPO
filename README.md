<div align="center">

# Torchrl Hydra Template

A clean template to kickstart your deep reinforcement learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/raphaelschwinger/torchrl-hydra-template/generate) to initialize new repository.

_Suggestions are always welcome!_

</div>

Reproducibility and rapid iteration are critical in reinforcement learning research. A well-structured project lets you:

- Swap algorithms, environments, and hyperparameters with a single config change.
- Reproduce results reliably across machines and collaborators.
- Spend less time on boilerplate and more time on research.

This template combines [TorchRL](https://github.com/pytorch/rl) and [Hydra](https://github.com/facebookresearch/hydra)
into an opinionated yet flexible scaffold for RL experimentation. TorchRL provides composable, GPU-friendly building
blocks for environments, replay buffers, losses, and data collection, while Hydra handles hierarchical configuration
with command-line overrides, making it straightforward to launch sweeps, compare algorithms, and version-control
every experiment setting.

Out of the box the template ships working implementations for common algorithms (REINFORCE, DQN, PPO, ...) and can be
extended to new algorithms or environments with minimal wiring, thanks to its modular config-driven design.

This project builds on the ideas pioneered by
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by @ashleve and further refined in
[yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template) by
@gorodnitskiy. Their work on combining structured Hydra configs with clean training pipelines served as the
foundation; this template adapts that philosophy to the reinforcement learning setting with TorchRL.

## Quick start

```shell
# clone template
git clone https://github.com/raphaelschwinger/torchrl-hydra-template
cd torchrl-hydra-template

# install requirements
uv sync

# activate virtual environment
source .venv/bin/activate

# run an experiment configured in configs/experiment e.g.:
python src/train.py experiment=reinforce/cartpole
```

## Main technologies

**[TorchRL](https://github.com/pytorch/rl)** — A PyTorch-native library for reinforcement learning that provides modular, composable primitives for environments, replay buffers, data collectors, and loss modules. It leverages [`TensorDict`](https://github.com/pytorch/tensordict) as a universal data carrier, making it easy to swap components without rewriting glue code, and supports GPU-accelerated batched simulation out of the box.

**[Hydra](https://github.com/facebookresearch/hydra)** — A configuration framework by Meta Research that lets you compose hierarchical configs from multiple YAML files and override any parameter from the command line. This makes it trivial to launch hyperparameter sweeps, compare algorithm variants, and keep every experiment setting version-controlled without touching Python code.

## Project structure

```
├── configs/                    <- Hydra configuration (compose-based)
│   ├── algorithm/              <- Per-algorithm hyperparameters + network architecture
│   │   ├── reinforce.yaml      <- REINFORCE (MLP policy, Monte-Carlo returns)
│   │   ├── dqn.yaml            <- DQN (Q-network, replay buffer, epsilon-greedy)
│   │   └── ppo.yaml            <- PPO (actor-critic, GAE, clipped surrogate)
│   ├── environment/            <- Environment construction kwargs
│   │   ├── cartpole.yaml
│   │   ├── atari_breakout.yaml
│   │   └── dmc_humanoid.yaml
│   ├── logger/                 <- Logger backend configs (wandb, tensorboard)
│   ├── experiment/             <- Composed algorithm × environment overrides
│   │   └── [algo]/[env].yaml   <- e.g. dqn/atari_breakout.yaml
│   ├── train.yaml              <- Top-level train defaults list
│   └── eval.yaml               <- Top-level eval defaults list
│
├── src/                        <- Source code
│   ├── algorithms/             <- RL algorithm implementations
│   │   ├── base.py             <- BaseAlgorithm (setup / train / eval / checkpoint)
│   │   ├── reinforce.py
│   │   ├── dqn.py
│   │   └── ppo.py
│   ├── callbacks/              <- Training callbacks
│   │   ├── logger.py           <- WandBLogger, TensorBoardLogger
│   │   ├── checkpoint.py       <- CheckpointCallback (full state, interval + last)
│   │   └── progress.py         <- ProgressCallback (tqdm CLI bar)
│   ├── environments/
│   │   └── factory.py          <- make_env: Gymnasium + dm_control + transforms
│   ├── networks/
│   │   └── factory.py          <- MLP, AtariCNN (dispatched from algorithm config)
│   ├── utils/                  <- device resolution, seeding, callback builders
│   ├── train.py                <- Training entry point
│   └── eval.py                 <- Evaluation entry point
│
├── tests/
│   └── test_smoke.py           <- One training cycle for every defined experiment
│
├── pyproject.toml              <- Dependencies + build config (uv / hatchling)
├── uv.lock                     <- Dependency lock file
└── README.md
```

## Concept

We follow the classic RL information flow of an agent interacting with an environment and organise our code accordingly:

- **Environment**: The environment defines and makes accessible to the agent:
    - name
    - observation shape
    - action shape
    - reward shape
    - total steps the agent can interact with

- **Algorithm**: The algorithm that learns the policy of the agent:
    - `setup` — initializes networks, loss modules, optimizer, and data collector. Can optionally receive a checkpoint to resume from.
    - `train` — entry point for the training loop
    - `eval` — runs the agent without further training

- **Trainer**: The component managing the training loop:
    - **Callbacks** used during training:
        - `ProgressCallback` — tqdm progress bar in the CLI
        - `CheckpointCallback` — saves full training state (policy + optimizer + replay buffer) every N steps and at the end
        - `WandBLogger` / `TensorBoardLogger` — metric logging (activate any combination via config)

## Configuration

Network architecture is embedded directly in each algorithm config — no separate config group. This keeps algorithm and network tightly coupled and avoids a combinatorial explosion of compatibility concerns:

```yaml
# configs/algorithm/dqn.yaml (excerpt)
network:
  architecture: mlp        # "mlp" | "cnn_atari"
  hidden_sizes: [128, 128]
  activation: relu
```

Override in an experiment config when needed (e.g. switching to CNN for Atari):

```yaml
# configs/experiment/dqn/atari_breakout.yaml (excerpt)
algorithm:
  network:
    architecture: cnn_atari
    conv_channels: [32, 64, 64]
    conv_kernels: [8, 4, 3]
    conv_strides: [4, 2, 1]
    fc_hidden: [512]
```

### Algorithm config dataclasses

Each algorithm file defines a typed `@dataclass` (e.g. `ReinforceConfig`, `DQNConfig`, `PPOConfig`) that lives alongside the algorithm class. The dataclass serves three purposes:

1. **Typed defaults** — every hyperparameter has an explicit Python default so the algorithm is runnable without any YAML.
2. **Inline documentation** — comments next to each field explain what it controls and any tuning guidance, right where the parameter is consumed.
3. **Discoverability** — a reader opening `reinforce.py` immediately sees all knobs without having to cross-reference a YAML file.

The YAML files remain the authoritative source of values for running experiments. At `setup()` time, `BaseAlgorithm._build_acfg()` merges the two:

```
dataclass defaults  ←  overridden by  →  configs/algorithm/*.yaml  ←  overridden by  →  experiment config / CLI
```

This means a new algorithm only needs a minimal YAML (just `_target_:`) during prototyping — the dataclass provides all defaults — and production configs can be written with full confidence that every tunable parameter is documented in one place.

## Device selection

Device configuration follows PyTorch Lightning conventions:

```shell
# CPU (default)
python src/train.py experiment=reinforce/cartpole

# Single GPU
python src/train.py experiment=dqn/atari_breakout trainer.accelerator=gpu trainer.devices=[0]

# Second GPU
python src/train.py experiment=ppo/dmc_humanoid trainer.accelerator=gpu trainer.devices=[1]
```

## Logging

Pass a list of loggers — any combination of `wandb` and `tensorboard` is valid:

```shell
# Both simultaneously
python src/train.py experiment=dqn/cartpole 'logger=[wandb,tensorboard]'

# TensorBoard only
python src/train.py experiment=reinforce/cartpole 'logger=[tensorboard]'

# No logging
python src/train.py experiment=reinforce/cartpole logger=[]
```

## Smoke tests

```shell
pytest tests/test_smoke.py -v
```

Each test loads the full experiment config, applies minimal-frame overrides (CPU, no logging, small buffer), and asserts that one complete training cycle runs without error.

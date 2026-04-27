"""Environment factory: builds a TorchRL TransformedEnv from Hydra config params.

Usage (from algorithm setup):
    from hydra.utils import instantiate
    env = instantiate(cfg.environment, device=str(self.device))

Or directly:
    from src.environments.factory import make_env
    env = make_env(**OmegaConf.to_container(cfg.environment, resolve=True), device="cpu")
"""
from __future__ import annotations

from functools import partial
from typing import Sequence


def make_env(
    name: str,
    backend: str,
    obs_shape: Sequence[int],
    num_actions: int,
    num_envs: int = 1,
    frame_stack: int = 1,
    grayscale: bool = False,
    resize: Sequence[int] | None = None,
    clip_rewards: bool = False,
    normalize_obs: bool = False,
    device: str = "cpu",
    task: str | None = None,
    max_episode_steps: int | None = None,
    **kwargs,
):
    """Build a (possibly vectorised) TransformedEnv.

    Args:
        name: environment name (e.g. "CartPole-v1", "ALE/Breakout-v5", "humanoid")
        backend: "gymnasium" or "dm_control"
        obs_shape: expected observation shape after preprocessing (for validation)
        num_actions: number of actions (discrete or continuous dim)
        num_envs: number of parallel envs (>1 → ParallelEnv)
        frame_stack: number of frames to stack (CatFrames)
        grayscale: convert RGB to grayscale
        resize: [H, W] to resize pixel observations
        clip_rewards: clip rewards to {-1, 0, +1} (standard Atari)
        normalize_obs: apply running mean/std normalisation to observations
        device: target device string
        task: dm_control task string (e.g. "walk")
        **kwargs: extra env kwargs forwarded to the base env constructor

    Returns:
        TransformedEnv (single env or wrapped in ParallelEnv)
    """
    if backend == "gymnasium":
        env_fn = partial(
            _make_gymnasium_env,
            name=name,
            grayscale=grayscale,
            resize=resize,
            frame_stack=frame_stack,
            clip_rewards=clip_rewards,
            normalize_obs=normalize_obs,
            device=device,
        )
    elif backend == "dm_control":
        env_fn = partial(
            _make_dmcontrol_env,
            name=name,
            task=task or "walk",
            normalize_obs=normalize_obs,
            device=device,
        )
    elif backend == "envpool":
        return _make_envpool_env(
            name=name,
            num_envs=num_envs,
            clip_rewards=clip_rewards,
            normalize_obs=normalize_obs,
            device=device,
            max_episode_steps=max_episode_steps,
        )
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: 'gymnasium', 'dm_control', 'envpool'."
        )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv
        return ParallelEnv(num_envs, env_fn)
    else:
        return env_fn()


def _make_gymnasium_env(
    name: str,
    grayscale: bool,
    resize: Sequence[int] | None,
    frame_stack: int,
    clip_rewards: bool,
    normalize_obs: bool,
    device: str,
):
    from torchrl.envs import GymEnv, TransformedEnv
    from torchrl.envs.transforms import (
        CatFrames,
        Compose,
        GrayScale,
        RewardClipping,
        ToTensorImage,
    )

    # Determine if this is a pixel-based env
    pixel_obs = grayscale or resize is not None

    base_env = GymEnv(name, device=device, from_pixels=pixel_obs)

    transforms = []

    if pixel_obs:
        transforms.append(ToTensorImage(in_keys=["pixels"], out_keys=["pixels"]))

    if grayscale:
        transforms.append(GrayScale(in_keys=["pixels"], out_keys=["pixels"]))

    if resize is not None:
        from torchrl.envs.transforms import Resize
        h, w = resize
        transforms.append(Resize(h, w, in_keys=["pixels"], out_keys=["pixels"]))

    if frame_stack > 1:
        transforms.append(
            CatFrames(N=frame_stack, dim=-3, in_keys=["pixels"], out_keys=["observation"])
        )
    elif pixel_obs:
        # Rename pixels → observation for a uniform key across all envs
        from torchrl.envs.transforms import RenameTransform
        transforms.append(RenameTransform(["pixels"], ["observation"]))

    if clip_rewards:
        transforms.append(RewardClipping(-1.0, 1.0))

    if normalize_obs:
        from torchrl.envs.transforms import ObservationNorm
        transforms.append(ObservationNorm(in_keys=["observation"]))

    from torchrl.envs.transforms import StepCounter
    transforms.append(StepCounter())

    if transforms:
        from torchrl.envs.transforms import Compose
        return TransformedEnv(base_env, Compose(*transforms))
    return base_env


def _patch_envpool_reset_mask(env):
    """Squeeze a trailing singleton from the ``_reset`` mask before envpool sees it.

    Once we unsqueeze done to ``[num_envs, 1]`` (see ``_make_envpool_env``),
    torchrl's ``maybe_reset`` forwards ``_reset`` with the same trailing
    dim, but envpool's ``_reset`` does ``self.obs[reset_workers]`` where
    ``self.obs`` has shape ``[num_envs]`` — so a ``[N, 1]`` mask raises
    ``IndexError``. Squeezing it keeps both sides happy.
    """
    _orig = env._reset

    def _reset(td, **kwargs):
        if td is not None:
            r = td.get("_reset", None)
            if r is not None and r.ndim > 1 and r.shape[-1] == 1:
                td = td.clone(False)
                td.set("_reset", r.squeeze(-1))
        return _orig(td, **kwargs)

    env._reset = _reset
    return env


def _make_envpool_env(
    name: str,
    num_envs: int,
    clip_rewards: bool,
    normalize_obs: bool,
    device: str,
    max_episode_steps: int | None = None,
):
    """envpool-backed vectorised env via torchrl's ``MultiThreadedEnv``.

    Two non-obvious details:

    * ``MultiThreadedEnvWrapper._get_action_spec`` hardcodes
      ``categorical_action_encoding=True``, so actions are scalar ints as the
      rest of the pipeline expects — no config knob needed.
    * Reward/done/terminated/truncated are emitted with shape ``[num_envs]``
      rather than torchrl's standard ``[num_envs, 1]``. Without the trailing
      singleton, ``DQNLoss`` raises ``"All input tensors (value, reward and
      done states) must share a unique shape"``. We fix this with an
      ``UnsqueezeTransform`` on those keys, plus a tiny patch to the base
      env's ``_reset`` so the now-2D ``_reset`` mask gets squeezed back to 1D
      before indexing envpool's internal obs buffer.
    """
    from torchrl.envs import MultiThreadedEnv, TransformedEnv
    from torchrl.envs.transforms import (
        Compose,
        ObservationNorm,
        RewardClipping,
        UnsqueezeTransform,
    )

    env = _patch_envpool_reset_mask(
        MultiThreadedEnv(
            num_workers=num_envs,
            env_name=name,
            device=device,
        )
    )

    # StepCounter is omitted here: its internal step_count is 1D while the
    # unsqueeze makes the _reset mask 2D, and it can't expand across them.
    # Last-episode logging reads "next.done" directly, so nothing depends on
    # step_count for this env path.
    transforms: list = [
        UnsqueezeTransform(
            dim=-1,
            in_keys=["reward", "done", "terminated", "truncated"],
            in_keys_inv=[],
        ),
    ]
    if clip_rewards:
        transforms.append(RewardClipping(-1.0, 1.0))
    if normalize_obs:
        transforms.append(ObservationNorm(in_keys=["observation"]))

    return TransformedEnv(env, Compose(*transforms))


def _make_dmcontrol_env(
    name: str,
    task: str,
    normalize_obs: bool,
    device: str,
):
    from torchrl.envs import DMControlEnv, TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter

    base_env = DMControlEnv(name, task, device=device)

    obs_keys = list(base_env.observation_spec.keys())
    from torchrl.envs.transforms import CatTensors
    transforms: list = [
        DoubleToFloat(in_keys=obs_keys),
        CatTensors(in_keys=obs_keys, out_key="observation", del_keys=True),
    ]

    if normalize_obs:
        from torchrl.envs.transforms import ObservationNorm
        transforms.append(ObservationNorm(in_keys=["observation"]))

    transforms.append(StepCounter())

    return TransformedEnv(base_env, Compose(*transforms))

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from loguru import logger as log

from roboverse_learn.il.runners.default_eval_runner import DefaultEvalRunner


class NoiseProbeEvalRunner(DefaultEvalRunner):
    """Eval runner that perturbs observation history for robustness probing.

    This runner is opt-in and keeps the default evaluation path untouched.
    """

    def _init_policy(self, default_runner, **kwargs):
        super()._init_policy(default_runner, **kwargs)

        self.probe_enable = bool(kwargs.get("probe_enable", False))
        self.probe_sigma = float(kwargs.get("probe_sigma", 0.0))
        self.probe_transform = str(kwargs.get("probe_transform", "noise"))
        self.probe_mode = str(kwargs.get("probe_mode", "continuous"))
        self.probe_step = int(kwargs.get("probe_step", 30))
        self.probe_seed = kwargs.get("probe_seed", None)

        valid_probe_transforms = {"noise", "reverse", "shuffle_time", "zero", "repeat_last"}
        if self.probe_transform not in valid_probe_transforms:
            raise ValueError(f"Unsupported probe_transform: {self.probe_transform}")
        if self.probe_mode not in {"continuous", "one_shot"}:
            raise ValueError(f"Unsupported probe_mode: {self.probe_mode}")

        self._probe_fired = False
        self._noise_gen = None
        if self.probe_seed is not None:
            self._noise_gen = torch.Generator(device="cpu")
            self._noise_gen.manual_seed(int(self.probe_seed))

        if self.probe_enable:
            log.info(
                "History probe enabled: transform={}, mode={}, sigma={}, step={}, seed={}",
                self.probe_transform,
                self.probe_mode,
                self.probe_sigma,
                self.probe_step,
                self.probe_seed,
            )

        # Closed-loop action shock config: perturb executed actions, not just inputs.
        self.action_shock_enable = bool(kwargs.get("action_shock_enable", False))
        self.action_shock_sigma = float(kwargs.get("action_shock_sigma", 0.0))
        self.action_shock_mode = str(kwargs.get("action_shock_mode", "one_shot"))
        self.action_shock_step = int(kwargs.get("action_shock_step", 30))
        self.action_shock_seed = kwargs.get("action_shock_seed", None)
        self.action_shock_clip = bool(kwargs.get("action_shock_clip", True))

        if self.action_shock_mode not in {"continuous", "one_shot"}:
            raise ValueError(f"Unsupported action_shock_mode: {self.action_shock_mode}")

        self._action_shock_fired = False
        self._action_rng = np.random.default_rng(self.action_shock_seed)

        if self.action_shock_enable:
            log.info(
                "Action shock enabled: mode={}, sigma={}, step={}, seed={}, clip={}",
                self.action_shock_mode,
                self.action_shock_sigma,
                self.action_shock_step,
                self.action_shock_seed,
                self.action_shock_clip,
            )

    def reset(self):
        self._probe_fired = False
        self._action_shock_fired = False
        super().reset()

    def _should_apply_probe(self) -> bool:
        if not self.probe_enable:
            return False
        if self.probe_transform == "noise" and self.probe_sigma <= 0:
            return False
        if self.probe_mode == "continuous":
            return True
        if self.probe_mode == "one_shot":
            return (not self._probe_fired) and (self.step >= self.probe_step)
        return False

    def _sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self._noise_gen is None:
            return torch.randn_like(x) * self.probe_sigma
        noise_cpu = torch.randn(x.shape, generator=self._noise_gen, dtype=x.dtype, device="cpu")
        return noise_cpu.to(device=x.device) * self.probe_sigma

    def _history_permutation(self, n_steps: int, device: torch.device) -> torch.Tensor:
        if self._noise_gen is None:
            return torch.randperm(n_steps, device=device)
        perm = torch.randperm(n_steps, generator=self._noise_gen, device="cpu")
        return perm.to(device=device)

    def _apply_history_probe(self, agent_pos: torch.Tensor) -> torch.Tensor:
        if self.probe_transform == "noise":
            return agent_pos + self._sample_noise(agent_pos)
        if self.probe_transform == "reverse":
            return torch.flip(agent_pos, dims=[1])
        if self.probe_transform == "shuffle_time":
            return agent_pos[:, self._history_permutation(agent_pos.shape[1], agent_pos.device), ...]
        if self.probe_transform == "zero":
            return torch.zeros_like(agent_pos)
        if self.probe_transform == "repeat_last":
            return agent_pos[:, -1:, ...].expand_as(agent_pos).clone()
        raise ValueError(f"Unsupported probe_transform: {self.probe_transform}")

    def predict_action(self, observaton=None):
        if observaton is not None:
            self.obs.append(observaton)
        obs = self._get_n_steps_obs()

        if self._should_apply_probe() and "agent_pos" in obs:
            obs = dict(obs)
            obs["agent_pos"] = self._apply_history_probe(obs["agent_pos"])
            if self.probe_mode == "one_shot":
                self._probe_fired = True

        with torch.no_grad():
            action_chunk = (
                self.policy.predict_action(obs)["action"].detach().to(torch.float32)
            )
            action_chunk = action_chunk.transpose(0, 1)
        return action_chunk

    def _should_apply_action_shock(self) -> bool:
        if not self.action_shock_enable or self.action_shock_sigma <= 0:
            return False
        if self.action_shock_mode == "continuous":
            return True
        if self.action_shock_mode == "one_shot":
            # self.step is incremented in BaseEvalRunner.get_action before return.
            current_step = max(0, self.step - 1)
            return (not self._action_shock_fired) and (current_step >= self.action_shock_step)
        return False

    def _joint_limits(self, joint_name: str):
        lim = self.scenario.robots[0].joint_limits.get(joint_name, None)
        if lim is None:
            return None
        try:
            if isinstance(lim, (list, tuple)) and len(lim) >= 2:
                return float(lim[0]), float(lim[1])
            if hasattr(lim, "__len__") and len(lim) >= 2:
                return float(lim[0]), float(lim[1])
        except Exception:
            return None
        return None

    def _apply_action_shock(self, actions):
        robot_name = self.scenario.robots[0].name
        for env_idx, action_env in enumerate(actions):
            if robot_name not in action_env:
                continue
            dof = action_env[robot_name].get("dof_pos_target", {})
            for jn, val in list(dof.items()):
                shock = float(self._action_rng.normal(loc=0.0, scale=self.action_shock_sigma))
                new_val = float(val) + shock
                if self.action_shock_clip:
                    lims = self._joint_limits(jn)
                    if lims is not None:
                        lo, hi = lims
                        new_val = float(np.clip(new_val, lo, hi))
                dof[jn] = float(new_val)
        return actions

    def get_action(self, obs):
        actions = super().get_action(obs)
        if self._should_apply_action_shock():
            actions = self._apply_action_shock(actions)
            if self.action_shock_mode == "one_shot":
                self._action_shock_fired = True
            log.debug(
                "Applied action shock at policy_step={} mode={} sigma={}",
                max(0, self.step - 1),
                self.action_shock_mode,
                self.action_shock_sigma,
            )
        return actions

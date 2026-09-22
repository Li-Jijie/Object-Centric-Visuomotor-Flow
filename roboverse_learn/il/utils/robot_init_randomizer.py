"""Robot initial state perturbation for spatial generalization evaluation.

Independent from visual DR and object position offset. Adds Gaussian
noise to the robot's initial joint positions before env reset, testing
whether the policy can adapt from perturbed starting configurations.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from loguru import logger as log


class RobotInitRandomizer:
    """Add Gaussian noise to robot initial joint positions.

    Usage:
        rand = RobotInitRandomizer(sigma=0.03, per_demo=True, seed=42)
        modified_states = rand.apply(init_states, demo_indices=[0, 1, 2])
    """

    def __init__(
        self,
        sigma: float = 0.03,
        per_demo: bool = True,
        seed: int | None = None,
        clip: float | None = None,
    ):
        self.sigma = float(sigma)
        self.per_demo = per_demo
        self._rng = np.random.default_rng(seed)
        self.clip = float(clip) if clip is not None else None
        self._global_noise: dict[str, dict[str, float]] | None = None
        self._demo_noises: dict[int, dict[str, dict[str, float]]] = {}

    @property
    def enabled(self) -> bool:
        return self.sigma > 1e-8

    def _sample_noise(self, robot_state: dict) -> dict[str, dict[str, float]]:
        """Sample per-joint noise. dof_pos is a dict {joint_name: scalar_value}."""
        noise: dict[str, dict[str, float]] = {}
        dof_pos = robot_state.get("dof_pos")
        if dof_pos is not None and isinstance(dof_pos, dict) and len(dof_pos) > 0:
            n_dof = len(dof_pos)
            n = self._rng.normal(0.0, self.sigma, size=n_dof)
            if self.clip is not None:
                n = np.clip(n, -self.clip, self.clip)
            noise["dof_pos"] = dict(zip(dof_pos.keys(), [float(v) for v in n]))
        return noise

    def _get_noise(self, demo_idx: int, robot_state: dict) -> dict[str, dict[str, float]]:
        if not self.per_demo:
            if self._global_noise is None:
                self._global_noise = self._sample_noise(robot_state)
            return self._global_noise
        if demo_idx not in self._demo_noises:
            self._demo_noises[demo_idx] = self._sample_noise(robot_state)
        return self._demo_noises[demo_idx]

    def apply(
        self,
        init_states: list[dict],
        demo_indices: Sequence[int] | None = None,
    ) -> list[dict]:
        """Return a deep-copied list of init_states with robot init noise applied.

        Original list is NOT mutated.
        """
        if not self.enabled:
            return init_states

        if demo_indices is None:
            demo_indices = list(range(len(init_states)))

        modified = list(init_states)
        for _i, demo_idx in enumerate(demo_indices):
            if demo_idx >= len(modified):
                continue
            state = copy.deepcopy(modified[demo_idx])
            if "robots" in state:
                for robot_name, robot_state in state["robots"].items():
                    noise = self._get_noise(demo_idx, robot_state)
                    if "dof_pos" in noise:
                        dof_pos: dict = robot_state["dof_pos"]
                        for jname, nval in noise["dof_pos"].items():
                            if jname in dof_pos:
                                dof_pos[jname] = dof_pos[jname] + nval
                        max_noise = max(abs(v) for v in noise["dof_pos"].values())
                        log.debug(
                            f"[RobotInit] demo{demo_idx} {robot_name}: "
                            f"sigma={self.sigma:.3f}, n_joints={len(dof_pos)}, max_dof_noise={max_noise:.4f}"
                        )
            modified[demo_idx] = state

        log.info(
            f"[RobotInitRandomizer] Applied noise sigma={self.sigma:.4f}, "
            f"per_demo={self.per_demo}, clip={self.clip} | {len(demo_indices)} demos"
        )
        return modified

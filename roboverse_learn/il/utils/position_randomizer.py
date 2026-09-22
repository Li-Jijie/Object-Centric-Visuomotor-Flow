"""Object position randomization for spatial generalization evaluation.

Independent from visual domain randomization (level 0-3). Applies
per-demo or global position offsets to task objects before env reset.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from loguru import logger as log


class PositionRandomizer:
    """Apply random XY(Z) offsets to task object initial positions.

    Usage:
        rand = PositionRandomizer(
            x_range=(-0.1, 0.1),
            y_range=(-0.1, 0.1),
            z_range=(0.0, 0.0),
            obj_names=None,       # None = all objects
            per_demo=True,
            seed=42,
        )
        modified_states = rand.apply(init_states, demo_indices=[0, 1, 2])
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-0.05, 0.05),
        y_range: tuple[float, float] = (-0.05, 0.05),
        z_range: tuple[float, float] = (0.0, 0.0),
        obj_names: Sequence[str] | None = None,
        per_demo: bool = True,
        seed: int | None = None,
    ):
        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.z_range = tuple(z_range)
        self.obj_names = list(obj_names) if obj_names else None
        self.per_demo = per_demo
        self._rng = np.random.default_rng(seed)
        self._global_offset: tuple[float, float, float] | None = None
        self._demo_offsets: dict[int, tuple[float, float, float]] = {}

    @property
    def enabled(self) -> bool:
        """Whether any offset range is non-zero."""
        return (
            abs(self.x_range[1] - self.x_range[0]) > 1e-8
            or abs(self.y_range[1] - self.y_range[0]) > 1e-8
            or abs(self.z_range[1] - self.z_range[0]) > 1e-8
        )

    def _sample_offset(self) -> tuple[float, float, float]:
        dx = float(self._rng.uniform(*self.x_range))
        dy = float(self._rng.uniform(*self.y_range))
        dz = float(self._rng.uniform(*self.z_range))
        return dx, dy, dz

    def _get_offset(self, demo_idx: int) -> tuple[float, float, float]:
        if not self.per_demo:
            if self._global_offset is None:
                self._global_offset = self._sample_offset()
            return self._global_offset
        if demo_idx not in self._demo_offsets:
            self._demo_offsets[demo_idx] = self._sample_offset()
        return self._demo_offsets[demo_idx]

    def _should_offset_obj(self, obj_name: str) -> bool:
        if self.obj_names is None or len(self.obj_names) == 0:
            return True
        return obj_name in self.obj_names

    def apply(
        self,
        init_states: list[dict],
        demo_indices: Sequence[int] | None = None,
    ) -> list[dict]:
        """Return a (shallow-copied) list of init_states with position offsets applied.

        Args:
            init_states: List of per-demo initial state dicts.
            demo_indices: Which demo indices to modify. None = all.

        Returns:
            New list with modified states. Original list is NOT mutated.
        """
        if not self.enabled:
            return init_states

        if demo_indices is None:
            demo_indices = list(range(len(init_states)))

        modified = list(init_states)  # shallow copy the list
        for i, demo_idx in enumerate(demo_indices):
            if demo_idx >= len(modified):
                continue
            dx, dy, dz = self._get_offset(demo_idx)
            state = copy.deepcopy(modified[demo_idx])
            if "objects" in state:
                for obj_name, obj_state in state["objects"].items():
                    if not self._should_offset_obj(obj_name):
                        continue
                    obj_state["pos"] = torch.tensor(
                        [
                            float(obj_state["pos"][0]) + dx,
                            float(obj_state["pos"][1]) + dy,
                            float(obj_state["pos"][2]) + dz,
                        ],
                        dtype=obj_state["pos"].dtype,
                        device=obj_state["pos"].device,
                    )
            modified[demo_idx] = state

        if self.per_demo and len(demo_indices) > 0:
            offsets_str = ", ".join(
                f"demo{d}: ({self._get_offset(d)[0]:+.3f}, {self._get_offset(d)[1]:+.3f}, {self._get_offset(d)[2]:+.3f})"
                for d in demo_indices
            )
        else:
            dx, dy, dz = self._get_offset(demo_indices[0] if demo_indices else 0)
            offsets_str = f"global: ({dx:+.3f}, {dy:+.3f}, {dz:+.3f})"
        log.info(
            f"[PositionRandomizer] Applied offsets: {offsets_str} | "
            f"ranges: x={self.x_range}, y={self.y_range}, z={self.z_range}"
        )
        return modified

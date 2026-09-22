from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from loguru import logger as log


@dataclass
class Args:
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaacsim"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = True
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 250
    """Maximum number of steps to collect"""
    render_mode: Literal["raytracing", "pathtracing"] = "raytracing"
    """IsaacSim render mode"""
    gpu_id: int = 0
    """GPU ID to use"""

    # Domain Randomization options
    level: Literal[0, 1, 2, 3] = 0
    """Randomization level: 0=None, 1=Scene+Material, 2=+Light, 3=+Camera"""
    scene_mode: Literal[0, 1, 2, 3] = 0
    """Scene mode: 0=Manual, 1=USD Table, 2=USD Scene, 3=Full USD"""
    randomization_seed: int | None = None
    """Seed for reproducible randomization. If None, uses random seed"""

    # Evaluation probe options (default off; keeps clean evaluation unchanged)
    probe_enable: bool = False
    """Enable observation-history perturbation probe during evaluation"""
    probe_sigma: float = 0.0
    """Std of Gaussian noise added to agent_pos history when probe is enabled"""
    probe_transform: Literal["noise", "reverse", "shuffle_time", "zero", "repeat_last"] = "noise"
    """How to perturb stacked agent_pos history during probe evaluation"""
    probe_mode: Literal["continuous", "one_shot"] = "continuous"
    """continuous: add noise every step; one_shot: inject only once at probe_step"""
    probe_step: int = 30
    """Step index for one-shot perturbation injection"""
    probe_seed: int | None = None
    """Optional random seed for reproducible probe noise"""

    # Object position generalization options (default off; independent from visual DR)
    pos_offset_enable: bool = False
    """Enable object position offset during evaluation for spatial generalization testing"""
    pos_offset_x_min: float = -0.05
    """Minimum X-axis offset in meters"""
    pos_offset_x_max: float = 0.05
    """Maximum X-axis offset in meters"""
    pos_offset_y_min: float = -0.05
    """Minimum Y-axis offset in meters"""
    pos_offset_y_max: float = 0.05
    """Maximum Y-axis offset in meters"""
    pos_offset_z_min: float = 0.0
    """Minimum Z-axis offset in meters"""
    pos_offset_z_max: float = 0.0
    """Maximum Z-axis offset in meters"""
    pos_offset_seed: int | None = 42
    """Optional random seed for reproducible position offsets. If None, uses random seed"""
    pos_offset_obj_names: str = ""
    """Comma-separated object names to apply position offset to. Empty = all task objects"""
    pos_offset_per_demo: bool = True
    """If True, sample a new offset per demo. If False, use a single offset for all demos"""

    # Mid-rollout object perturbation (history/current-observation conflict)
    mid_obj_perturb_enable: bool = False
    """Teleport task object(s) during rollout after history has accumulated"""
    mid_obj_perturb_step: int = 40
    """Policy step at which to apply the mid-rollout object displacement"""
    mid_obj_perturb_x: float = 0.10
    """X displacement in meters for mid-rollout object perturbation"""
    mid_obj_perturb_y: float = 0.0
    """Y displacement in meters for mid-rollout object perturbation"""
    mid_obj_perturb_z: float = 0.0
    """Z displacement in meters for mid-rollout object perturbation"""
    mid_obj_perturb_obj_names: str = ""
    """Comma-separated object names to perturb. Empty = all task objects"""
    mid_obj_perturb_zero_vel: bool = True
    """Zero perturbed object root/joint velocities after teleport"""

    # Robot initial state perturbation (default off; independent from visual DR and pos offset)
    robot_init_noise_enable: bool = False
    """Enable robot initial joint position noise for start-configuration generalization testing"""
    robot_init_noise_sigma: float = 0.0
    """Std of Gaussian noise added to robot initial joint positions (radians)"""
    robot_init_noise_per_demo: bool = True
    """If True, sample new noise per demo. If False, same noise for all demos"""
    robot_init_noise_seed: int | None = 42
    """Optional random seed for reproducible robot init noise"""
    robot_init_noise_clip: float | None = None
    """Optional clip range for noise. If None, no clipping"""

    # Closed-loop action shock options (default off)
    action_shock_enable: bool = False
    """Enable perturbation on executed actions during evaluation (closed-loop shock test)"""
    action_shock_sigma: float = 0.0
    """Std of Gaussian shock added to joint target action"""
    action_shock_mode: Literal["continuous", "one_shot"] = "one_shot"
    """continuous: perturb every step; one_shot: inject only once at action_shock_step"""
    action_shock_step: int = 30
    """Step index for one-shot action shock injection"""
    action_shock_seed: int | None = None
    """Optional random seed for reproducible action shock"""
    action_shock_clip: bool = True
    """Clip shocked joint targets to joint limits"""

    def __post_init__(self):
        log.info(f"Args: {self}")

from roboverse_learn.il.configs.base_config import BasePolicyCfg

Pose = None
get_curobo_models = None
try:
    from curobo.types.math import Pose
except ImportError:
    pass
try:
    from metasim.utils.kinematics import get_curobo_models
except ImportError:
    pass

import sys

import torch
from loguru import logger as log
from metasim.scenario.scenario import ScenarioCfg
from roboverse_learn.il.utils.pytorch_util import dict_apply


def _quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)


def _quat_invert(q: torch.Tensor) -> torch.Tensor:
    # q format: [w, x, y, z]
    qn = _quat_normalize(q)
    wxyz = qn.clone()
    wxyz[..., 1:] = -wxyz[..., 1:]
    return wxyz


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Hamilton product, q format: [w, x, y, z]
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    out = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )
    return _quat_normalize(out)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Rotate vector v by quaternion q (both batched)
    qn = _quat_normalize(q)
    qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_multiply(_quat_multiply(qn, qv), _quat_invert(qn))[..., 1:]


def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    # q format: [w, x, y, z], returns XYZ intrinsic euler
    qn = _quat_normalize(q)
    w, x, y, z = qn.unbind(dim=-1)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    rx = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    ry = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    rz = torch.atan2(t3, t4)
    return torch.stack([rx, ry, rz], dim=-1)


def _euler_xyz_to_quat(e: torch.Tensor) -> torch.Tensor:
    # e format: [...,3], returns quat [w,x,y,z]
    ex, ey, ez = e.unbind(dim=-1)
    cx, sx = torch.cos(ex * 0.5), torch.sin(ex * 0.5)
    cy, sy = torch.cos(ey * 0.5), torch.sin(ey * 0.5)
    cz, sz = torch.cos(ez * 0.5), torch.sin(ez * 0.5)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    return _quat_normalize(torch.stack([w, x, y, z], dim=-1))


def _build_curobo_pose(pos: torch.Tensor, quat: torch.Tensor):
    """
    Build a cuRobo Pose with lazy import fallback.
    Some environments fail the module-level Pose import but succeed later.
    """
    global Pose
    if Pose is None:
        _pose_cls = None
        try:
            from curobo.types.math import Pose as _Pose  # newer/expected layout
            _pose_cls = _Pose
        except Exception:
            try:
                from curobo.types import Pose as _Pose  # fallback for alt layouts
                _pose_cls = _Pose
            except Exception as e:
                raise ImportError(
                    "Unable to import Pose from curobo (tried curobo.types.math.Pose and curobo.types.Pose)."
                ) from e
        Pose = _pose_cls
    try:
        return Pose(pos, quat)
    except TypeError:
        # Some variants require keyword args.
        return Pose(position=pos, quaternion=quat)


class BaseEvalRunner:
    """Base class to run a policy, based on the policyCFG it preprocesses the observation to
    match the policy's input requirements and postprocesses the action into joint space to match the policy's action type
    """

    def __init__(self, runner, scenario: ScenarioCfg, num_envs: int = 1, **kwargs):
        self.num_envs = num_envs
        self.scenario = scenario
        self.policy_cfg: BasePolicyCfg = None
        self.step = 0
        self.device = kwargs.get("device", "cuda:0")
        self.task_name = kwargs.get("task_name")
        self._init_policy(runner, **kwargs)
        self.robot_ik = None
        self.curobo_n_dof = None
        self.action_cache = []
        self._ee_key_debug_printed = False
        self.__post_init__()

    def _init_policy(self, **kwargs):
        """Method for subclasses to inherit to load in the policy"""
        raise NotImplementedError

    def __post_init__(self):
        if self.policy_cfg.action_config.action_type == "ee":
            if get_curobo_models is None:
                raise ImportError(
                    "EE action eval requires curobo + metasim kinematics. "
                    "Failed to import get_curobo_models."
                )
            self.robot_fk_model, self.robot_fk_fn, self.robot_ik = get_curobo_models(
                self.scenario.robots[0]
            )
            self.curobo_n_dof = len(self.robot_ik.robot_config.cspace.joint_names)
            self.ee_n_dof = len(self.scenario.robots[0].gripper_open_q)

        if self.policy_cfg.action_config.temporal_agg:
            self.all_time_actions = torch.zeros(
                [
                    self.num_envs,
                    self.scenario.episode_length,
                    self.scenario.episode_length
                    + self.policy_cfg.action_config.action_chunk_steps,
                    self.policy_cfg.action_config.action_dim,
                ],
                device=self.device,
            )
            self.k = 0.01

    def _get_root_pose_from_obs(self, obs):
        """Return robot root pose (pos, quat wxyz). Fallback to identity if absent."""
        if "robot_root_state" in obs:
            robot_root_state = obs["robot_root_state"]
            return robot_root_state[:, 0:3], robot_root_state[:, 3:7]
        pos = torch.zeros(self.num_envs, 3, device=self.device)
        quat = torch.zeros(self.num_envs, 4, device=self.device)
        quat[:, 0] = 1.0
        return pos, quat

    def _get_ee_state_from_obs(self, obs):
        """Find EE state from various possible keys/layouts and normalize to [B,8] as [pos(3), quat(4), grip(1)]."""
        def _maybe_get(d, key):
            return d[key] if isinstance(d, dict) and key in d else None

        def _search_nested(d, candidates):
            if not isinstance(d, dict):
                return None, None
            for k in candidates:
                if k in d:
                    return k, d[k]
            for k, v in d.items():
                if isinstance(v, dict):
                    kk, vv = _search_nested(v, candidates)
                    if vv is not None:
                        return f"{k}.{kk}", vv
            return None, None

        candidates = [
            "robot_ee_state",
            "ee_state",
            "robot_tcp_state",
            "tcp_state",
            "ee_pose",
            "tcp_pose",
        ]

        hit_key, hit_val = None, None
        # 1) direct + nested exact lookup
        hit_key, hit_val = _search_nested(obs, candidates)

        # 2) fuzzy nested lookup (contains both ee/tcp and state/pose)
        if hit_val is None and isinstance(obs, dict):
            stack = [("", obs)]
            while stack:
                prefix, cur = stack.pop()
                if not isinstance(cur, dict):
                    continue
                for k, v in cur.items():
                    lk = str(k).lower()
                    path = f"{prefix}.{k}" if prefix else str(k)
                    if isinstance(v, dict):
                        stack.append((path, v))
                    elif torch.is_tensor(v):
                        if ("ee" in lk or "tcp" in lk) and ("state" in lk or "pose" in lk):
                            hit_key, hit_val = path, v
                            break
                if hit_val is not None:
                    break

        if hit_val is None:
            # Last-resort fallback: derive EE state from joint_qpos via FK.
            if "joint_qpos" in obs and hasattr(self, "robot_fk_fn"):
                q = obs["joint_qpos"]
                if not torch.is_tensor(q):
                    q = torch.as_tensor(q, device=self.device, dtype=torch.float32)
                q = q.to(self.device)
                if q.dim() == 1:
                    q = q.unsqueeze(0)
                q_fk = q[:, : self.curobo_n_dof]
                ee_pos, ee_quat = self.robot_fk_fn(q_fk)
                # Grip from last ee_n_dof joints (binary open/close proxy)
                if q.shape[-1] >= self.ee_n_dof and self.ee_n_dof > 0:
                    grip_j = q[:, -self.ee_n_dof :]
                    open_q = torch.as_tensor(
                        self.scenario.robots[0].gripper_open_q,
                        device=self.device,
                        dtype=grip_j.dtype,
                    ).view(1, -1)
                    close_q = torch.as_tensor(
                        self.scenario.robots[0].gripper_close_q,
                        device=self.device,
                        dtype=grip_j.dtype,
                    ).view(1, -1)
                    denom = (open_q - close_q).clamp_min(1e-6)
                    open_ratio = ((grip_j - close_q) / denom).clamp(0.0, 1.0)
                    grip = (open_ratio.mean(dim=-1, keepdim=True) >= 0.5).to(grip_j.dtype)
                else:
                    grip = torch.ones(q.shape[0], 1, device=self.device)
                ee_fk = torch.cat([ee_pos, ee_quat, grip], dim=-1)
                if not self._ee_key_debug_printed:
                    log.info(
                        f"[EE OBS] derived from joint_qpos via FK, shape={tuple(ee_fk.shape)}"
                    )
                    self._ee_key_debug_printed = True
                return ee_fk
            if not self._ee_key_debug_printed:
                top_keys = list(obs.keys()) if isinstance(obs, dict) else [type(obs).__name__]
                log.error(f"[EE OBS] Missing EE key. top-level obs keys: {top_keys}")
                self._ee_key_debug_printed = True
            raise KeyError("Missing EE state in obs. Expected one of robot_ee_state/ee_state (or nested ee/tcp state).")

        ee = hit_val
        if not torch.is_tensor(ee):
            ee = torch.as_tensor(ee, device=self.device, dtype=torch.float32)
        if ee.dim() == 1:
            ee = ee.unsqueeze(0)
        ee = ee.to(self.device)

        # Normalize to [B,8]: [pos(3), quat(4), grip(1)]
        if ee.shape[-1] >= 8:
            if not self._ee_key_debug_printed:
                log.info(f"[EE OBS] using key={hit_key}, shape={tuple(ee.shape)} (quat-format)")
                self._ee_key_debug_printed = True
            return ee[:, :8]
        if ee.shape[-1] == 7:
            # Assume [pos(3), rpy(3), grip(1)] and convert to quaternion
            pos = ee[:, 0:3]
            rpy = ee[:, 3:6]
            grip = ee[:, 6:7]
            quat = _euler_xyz_to_quat(rpy)
            if not self._ee_key_debug_printed:
                log.info(f"[EE OBS] using key={hit_key}, shape={tuple(ee.shape)} (rpy-format -> quat)")
                self._ee_key_debug_printed = True
            return torch.cat([pos, quat, grip], dim=-1)

        if not self._ee_key_debug_printed:
            log.error(f"[EE OBS] key={hit_key} has unsupported shape={tuple(ee.shape)}")
            self._ee_key_debug_printed = True
        raise KeyError(f"Unsupported EE state shape: {tuple(ee.shape)}")

    def process_obs(self, obs):
        """
        Processes the observation to be used by the policy, according to the observation the policy is configured to use.
        """
        obs = dict_apply(
            obs,
            lambda x: x.to(device=self.device) if isinstance(x, torch.Tensor) else x,
        )
        obs_dict = {}

        if self.policy_cfg.obs_config.norm_image:
            obs_dict["head_cam"] = obs["rgb"].permute(0, 3, 1, 2) / 255.0
        else:
            obs_dict["head_cam"] = obs["rgb"]
        if self.policy_cfg.obs_config.obs_type == "joint_pos":
            obs_dict["agent_pos"] = obs["joint_qpos"]
        if self.policy_cfg.obs_config.obs_type == "ee":
            robot_ee_state = self._get_ee_state_from_obs(obs)
            robot_pos, robot_quat = self._get_root_pose_from_obs(obs)
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = _quat_apply(_quat_invert(robot_quat), curr_ee_pos - robot_pos)
            curr_ee_quat_local = _quat_multiply(_quat_invert(robot_quat), curr_ee_quat)

            if self.policy_cfg.obs_config.ee_cfg.gripper_rep == "q_pos":
                gripper_state = obs["joint_qpos"][:, -2:]
            else:
                gripper_state = robot_ee_state[:, -1]

            if self.policy_cfg.obs_config.ee_cfg.rotation_rep == "quaternion":
                curr_ee_rot_local = curr_ee_quat_local
            else:
                curr_ee_rot_local = _quat_to_euler_xyz(curr_ee_quat_local)

            obs_dict["agent_pos"] = torch.cat(
                [curr_ee_pos_local, curr_ee_rot_local, gripper_state], dim=1
            )

        if self.policy_cfg.obs_config.obs_padding > 0:
            padding_len = (
                self.policy_cfg.obs_config.obs_padding - obs_dict["agent_pos"].shape[1]
            )
            padding = torch.zeros(self.num_envs, padding_len, device=self.device)
            obs_dict["agent_pos"] = torch.cat([obs_dict["agent_pos"], padding], dim=1)

        assert obs_dict["agent_pos"].shape == (
            self.num_envs,
            self.policy_cfg.obs_config.obs_dim,
        )
        # flush unused keys
        obs_dict = {
            k: v
            for k, v in obs_dict.items()
            if k in self.policy_cfg.obs_config.obs_keys
        }
        return obs_dict

    def action_to_dict(self, curr_action: torch.Tensor):
        """
        Converts action tensor to dict with joint keys
        """
        action_nested_list = curr_action.tolist()  # bulk GPU-> CPU transfer; elements of Action must be python float
        actions = [
            {
                self.scenario.robots[0].name: {
                    "dof_pos_target": {
                        joint_name: action_nested_list[i][index]
                        for index, joint_name in enumerate(
                            sorted(self.scenario.robots[0].joint_limits.keys())
                        )
                    }
                }
            }
            for i in range(self.num_envs)
        ]
        return actions

    def get_temporal_agg_action(self, action_chunk):
        """
        Implements temporal ensembline, as in Aloha ACT. Takes in a current prediction chunk and returns a single ensembled action
        """
        assert action_chunk.shape == (
            self.policy_cfg.action_config.action_chunk_steps,
            self.num_envs,
            self.policy_cfg.action_config.action_dim,
        )

        # Put envs dimension first
        self.all_time_actions[
            :,
            self.step,
            self.step : self.step + self.policy_cfg.action_config.action_chunk_steps,
        ] = action_chunk.transpose(0, 1)

        actions_for_curr_step = self.all_time_actions[:, :, self.step]

        actions_populated = torch.all(
            torch.all(actions_for_curr_step != 0, dim=2), dim=0
        )
        actions_for_curr_step = actions_for_curr_step[:, actions_populated]

        time_indices = torch.arange(
            actions_for_curr_step.shape[1],
            device=actions_for_curr_step.device,
            dtype=torch.float,
        )
        exp_weights = torch.exp(self.k * time_indices)
        exp_weights = exp_weights / exp_weights.sum()

        weighted_actions = actions_for_curr_step * exp_weights.unsqueeze(-1).unsqueeze(
            0
        )

        raw_action = weighted_actions.sum(dim=1)

        return raw_action

    def get_action(self, obs):
        """Returns a single action to be directly executed. For action chunking policies it either uses an previsouly
        predicted action chunk, or if it has exausted all of those actions, it queries the model for a new chunk and returns the first one
        """
        # Always update observation history for policies that need continuous observation history
        # This is critical for action-to-action flow policies like VITA
        processed_obs = self.process_obs(obs)
        self.update_obs(processed_obs)
        
        if len(self.action_cache) > 0:
            curr_action = self.action_cache.pop(0)
        else:
            action_chunk = self.predict_action(
                None  # Don't pass obs again since we already updated it
            )  # shape: (action_chunk_steps, num_envs, action_dim)
            if self.policy_cfg.action_config.temporal_agg:
                curr_action = self.get_temporal_agg_action(action_chunk)
                curr_action = self.process_action([curr_action], obs)[0]
            else:
                qpos_action = self.process_action(action_chunk, obs)
                assert (
                    len(qpos_action) == self.policy_cfg.action_config.action_chunk_steps
                ), (
                    f"Expected {self.policy_cfg.action_config.action_chunk_steps} actions, got {len(qpos_action)}"
                )
                self.action_cache = qpos_action
                curr_action = self.action_cache.pop(0)

        self.step += 1
        assert curr_action.shape == (
            self.num_envs,
            len(self.scenario.robots[0].joint_limits.keys()),
        ), (
            f"Expected num_envs X n_dof : {self.num_envs} X {len(self.scenario.robots[0].joint_limits.keys())}, got {curr_action.shape} instead"
        )

        actions = self.action_to_dict(curr_action)
        return actions

    def predict_action(self, obs):
        raise NotImplementedError

    def _solve_ik(self, action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q):
        """Solves IK for the given action end-effector action, in either delta or absolute control"""
        assert action.shape == (
            self.num_envs,
            self.policy_cfg.action_config.action_dim,
        ), (
            f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {action.shape} instead"
        )
        if self.policy_cfg.action_config.ee_cfg.rotation_rep == "quaternion":
            ee_quat_action = action[:, 3:7]
            quat_norm = torch.norm(ee_quat_action, dim=1, keepdim=True)
            ee_quat_action = ee_quat_action / (quat_norm + 1e-5)
        else:
            ee_quat_action = _euler_xyz_to_quat(action[:, 3:6])

        if self.policy_cfg.action_config.delta:
            ee_pos_target = curr_ee_pos_local + action[:, :3]
            ee_quat_target = _quat_multiply(curr_ee_quat_local, ee_quat_action)
        else:
            ee_pos_target = action[:, :3]
            ee_quat_target = ee_quat_action

        # Solve IK
        seed_config = (
            curr_robot_q[:, : self.curobo_n_dof]
            .unsqueeze(1)
            .tile([1, self.robot_ik._num_seeds, 1])
        )
        result = self.robot_ik.solve_batch(
            _build_curobo_pose(ee_pos_target.cuda(0), ee_quat_target.cuda(0)),
            seed_config=seed_config.cuda(0),
        )

        if self.policy_cfg.action_config.ee_cfg.gripper_rep == "strength":
            gripper_pos = 1 - action[:, -1]
            gripper_widths = torch.zeros(
                self.num_envs, self.ee_n_dof, device=self.device
            )
            for i in range(self.num_envs):
                if gripper_pos[i] < 0.5:
                    gripper_widths[i] = torch.tensor(
                        self.scenario.robots[0].gripper_close_q, device=self.device
                    )
                else:
                    gripper_widths[i] = torch.tensor(
                        self.scenario.robots[0].gripper_open_q, device=self.device
                    )
        else:
            gripper_widths = action[:, -self.ee_n_dof :]

        q = curr_robot_q.clone()
        ik_succ = result.success.squeeze(1).to(self.device)
        if (~ik_succ).any():
            log.warning(f"IK failed: {ik_succ}")
            log.info("Trying to POS delta: ", action[:, :3])

        q[ik_succ, : self.curobo_n_dof] = result.solution.to(self.device)[
            ik_succ, 0
        ].clone()
        q[:, -self.ee_n_dof :] = gripper_widths
        return q

    def process_action(self, action_chunk, obs):
        """
        Processes a chunk of actions into joint positions.
        """
        action_chunk = [a.to(self.device) for a in action_chunk]
        for a in action_chunk:
            assert a.shape == (
                self.num_envs,
                self.policy_cfg.action_config.action_dim,
            ), (
                f"Expected num_envs X action_dim : {self.num_envs} X {self.policy_cfg.action_config.action_dim}, got {a.shape} instead"
            )
        if self.policy_cfg.action_config.action_type == "joint_pos":
            qpos_action_chunk = action_chunk
        elif self.policy_cfg.action_config.action_type == "ee":
            qpos_action_chunk = []
            robot_ee_state = self._get_ee_state_from_obs(obs).to(self.device)
            robot_pos, robot_quat = self._get_root_pose_from_obs(obs)
            curr_ee_pos, curr_ee_quat = robot_ee_state[:, 0:3], robot_ee_state[:, 3:7]
            curr_ee_pos_local = _quat_apply(_quat_invert(robot_quat), curr_ee_pos - robot_pos)
            curr_ee_quat_local = _quat_multiply(_quat_invert(robot_quat), curr_ee_quat)
            curr_robot_q = obs["joint_qpos"].to(self.device)
            for action in action_chunk:
                target_qpos = self._solve_ik(
                    action, curr_ee_pos_local, curr_ee_quat_local, curr_robot_q
                )
                qpos_action_chunk.append(target_qpos)

        if self.policy_cfg.action_config.interpolate_chunk:
            return self._interpolate_chunk(
                obs["joint_qpos"].to(self.device), qpos_action_chunk
            )
        else:
            return qpos_action_chunk

    def _interpolate_chunk(self, curr_qpos, qpos_action_chunk):
        """Smoothly interpolates between the current state and final predicted action of the chunk"""
        last_action = qpos_action_chunk[-1]
        assert curr_qpos.shape == last_action.shape, (
            f"Expected {curr_qpos.shape} and {last_action.shape} to be the same, got {curr_qpos.shape} and {last_action.shape} instead"
        )

        return [
            curr_qpos
            + (last_action - curr_qpos)
            * (i + 1)
            / self.policy_cfg.action_config.action_chunk_steps
            for i in range(self.policy_cfg.action_config.action_chunk_steps)
        ]

    def update_obs(self, current_obs):
        """Update observation history. Override in subclass if needed."""
        pass  # Default implementation does nothing; subclasses should override

    def reset(self):
        self.action_cache = []
        self.step = 0

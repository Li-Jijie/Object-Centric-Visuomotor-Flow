import argparse
import json
import logging
import os
import shutil

import imageio.v2 as iio
import numpy as np
import torch
import zarr
from tqdm import tqdm

try:
    from pytorch3d import transforms
except ImportError:
    pass

try:
    from scipy.spatial.transform import Rotation as _SciRotation
except ImportError:
    _SciRotation = None


def _as_np_float(arr_like):
    return np.asarray(arr_like, dtype=np.float32)


def _build_real_tcp7(vec, gripper_norm=None):
    """
    Build a 7D TCP vector: [x, y, z, r1, r2, r3, gripper].
    Supports source vector dim 6 or 7.
    """
    v = _as_np_float(vec).reshape(-1)
    if v.shape[0] == 7:
        return v
    if v.shape[0] == 6:
        g = 1.0 if gripper_norm is None else float(gripper_norm)
        return np.concatenate([v, np.array([g], dtype=np.float32)], axis=0)
    raise ValueError(f"Unsupported TCP dim={v.shape[0]}, expected 6 or 7")


def _rotvec_delta_np(curr_rotvec, target_rotvec):
    """Relative rotvec delta satisfying R_target ~= R_delta * R_curr."""
    curr = _as_np_float(curr_rotvec).reshape(3)
    target = _as_np_float(target_rotvec).reshape(3)
    if _SciRotation is None:
        # Fallback is only an approximation, but keeps values on the principal
        # branch instead of producing 2*pi wrap jumps from raw subtraction.
        diff = target - curr
        return ((diff + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)
    r_curr = _SciRotation.from_rotvec(curr)
    r_target = _SciRotation.from_rotvec(target)
    return (r_target * r_curr.inv()).as_rotvec().astype(np.float32)


def _euler_xyz_to_quat_np(euler_xyz):
    """Convert XYZ Euler (rad) to quaternion [w, x, y, z]."""
    e = _as_np_float(euler_xyz).reshape(-1)
    if e.shape[0] != 3:
        raise ValueError(f"Expected 3D euler, got shape={e.shape}")
    rx, ry, rz = float(e[0]), float(e[1]), float(e[2])
    cx, sx = np.cos(rx * 0.5), np.sin(rx * 0.5)
    cy, sy = np.cos(ry * 0.5), np.sin(ry * 0.5)
    cz, sz = np.cos(rz * 0.5), np.sin(rz * 0.5)
    # XYZ intrinsic (equiv. extrinsic ZYX)
    w = cx * cy * cz + sx * sy * sz
    x = sx * cy * cz - cx * sy * sz
    y = cx * sy * cz + sx * cy * sz
    z = cx * cy * sz - sx * sy * cz
    q = np.array([w, x, y, z], dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def _quat_mul_np(q1, q2):
    """Quaternion multiply for [w, x, y, z]."""
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def _quat_inv_np(q):
    w, x, y, z = [float(v) for v in q]
    n2 = w * w + x * x + y * y + z * z
    if n2 < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([w, -x, -y, -z], dtype=np.float32) / n2


def _build_sim_ee9_from_ee_state(vec):
    """
    Build 9D EE vector compatible with eval runner ee(quaternion,q_pos):
    [x, y, z, qw, qx, qy, qz, gripper_l, gripper_r]
    From ee_state [x,y,z,rx,ry,rz,g] (or already-9D passthrough).
    """
    v = _as_np_float(vec).reshape(-1)
    if v.shape[0] == 9:
        return v
    if v.shape[0] != 7:
        raise ValueError(f"Unsupported ee_state dim={v.shape[0]}, expected 7 or 9")
    pos = v[:3]
    quat = _euler_xyz_to_quat_np(v[3:6])
    g = float(v[6])
    grip2 = np.array([g, g], dtype=np.float32)
    return np.concatenate([pos, quat, grip2], axis=0)


def main():
    parser = argparse.ArgumentParser(description="Process Meta Data To ZARR For Diffusion Policy.")
    parser.add_argument(
        "--task_name",
        type=str,
        default="StackCube_franka",
        help="The name of the task (e.g., StackCube_franka)",
    )
    parser.add_argument(
        "--expert_data_num",
        type=int,
        default=200,
        help="Number of episodes to process (e.g., 200)",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="~/RoboVerse/data_isaaclab/demo/StackCube/robot-franka",
        help="The path of metadata",
    )
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=1,
        help="The downsample ratio of metadata",
    )

    parser.add_argument(
        "--observation_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The observation space to use (e.g., joint_pos, ee)",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="joint_pos",
        choices=["joint_pos", "ee"],
        help="The action space to use (e.g., joint_pos, ee)",
    )

    parser.add_argument("--delta_ee", type=int, choices=[0, 1], default=0)

    parser.add_argument(
        "--joint_pos_padding",
        type=int,
        default=0,
        help="If > 0, pad joint positions to this length when using joint_pos observation/action space",
    )
    parser.add_argument(
        "--require_task_mask",
        action="store_true",
        help="Fail if any demo is missing task_mask.npy (avoid silent all-white fallback).",
    )

    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    load_dir = args.metadata_dir
    downsample_ratio = args.downsample_ratio

    print("Metadata load dir:", load_dir)

    # Scan success folder to find all existing demo indices
    success_dir = os.path.join(load_dir)
    demo_indices = []

    if os.path.exists(success_dir) and os.path.isdir(success_dir):
        for item in os.listdir(success_dir):
            if item.startswith("demo_") and os.path.isdir(os.path.join(success_dir, item)):
                try:
                    demo_idx = int(item.split("_")[1])
                    demo_dir = os.path.join(success_dir, item)
                    # Check if metadata.json exists
                    if os.path.isfile(os.path.join(demo_dir, "metadata.json")):
                        demo_indices.append(demo_idx)
                except (ValueError, IndexError):
                    continue

        demo_indices.sort()
        print(f"Found {len(demo_indices)} demos in success folder: {demo_indices[:10]}{'...' if len(demo_indices) > 10 else ''}")

        # Use actual number of demos if requested number is not enough
        if len(demo_indices) < num:
            print(f"Requested {num} demos but only {len(demo_indices)} available. Using all available demos.")
            num = len(demo_indices)
        else:
            # Use first num demos
            demo_indices = demo_indices[:num]
    else:
        # Fallback to old behavior if success folder doesn't exist
        print(f"Success folder not found at {success_dir}, falling back to sequential processing")
        demo_indices = list(range(num))

    export_task_name = task_name
    # Naming guardrail: when exporting ee-delta data, append "_delta" by default
    # unless the task name already encodes a delta suffix.
    if (
        args.observation_space == "ee"
        and args.action_space == "ee"
        and int(args.delta_ee) == 1
        and (not export_task_name.endswith("_delta"))
        and (not export_task_name.endswith("_delta1"))
    ):
        export_task_name = f"{export_task_name}_delta"
        print(f"[Naming] delta_ee=1 detected, export task_name auto-updated to: {export_task_name}")

    save_dir = f"data_policy/{export_task_name}_{len(demo_indices)}.zarr"
    print("ZARR save dir:", save_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    # Batch processing settings
    batch_size = 100
    head_camera_arrays = []
    task_mask_arrays = []
    action_arrays = []
    state_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
    missing_task_mask_demo_count = 0
    fallback_white_frame_count = 0

    if args.joint_pos_padding > 0 and args.observation_space == "ee" and args.action_space == "ee":
        logging.warning("Padding is not supported for ee observation and action spaces.")

    for current_ep, demo_idx in enumerate(tqdm(demo_indices, desc=f"Processing {len(demo_indices)} MetaData")):
        demo_id = str(demo_idx).zfill(4)
        # Try success folder first, then fallback to root load_dir
        demo_dir = os.path.join(success_dir, f"demo_{demo_id}")
        if not os.path.isdir(demo_dir):
            demo_dir = os.path.join(load_dir, f"demo_{demo_id}")

        if not os.path.isdir(demo_dir):
            print(f"Skipping episode {demo_idx} as directory {demo_dir} does not exist.")
            continue

         # Check metadata.json
        metadata_path = os.path.join(demo_dir, "metadata.json")
        if not os.path.isfile(metadata_path):
          print(f"Skipping episode {demo_idx} as metadata.json does not exist.")
          continue
        else:
            with open(os.path.join(demo_dir, "metadata.json"), encoding="utf-8") as f:
                # print("metadata load dir:", demo_dir)
                metadata = json.load(f)

        if args.observation_space == "joint_pos" or args.action_space == "joint_pos":
            # Support both "joint_qpos" (sim) and "state_joint" (real) naming conventions
            if "joint_qpos" in metadata:
                state_key_joint = "joint_qpos"
                action_key_joint = "joint_qpos_target"
            elif "state_joint" in metadata:
                state_key_joint = "state_joint"
                action_key_joint = "action_joint"
            else:
                raise KeyError(
                    f"joint_pos requested but metadata has no joint keys. "
                    f"Expected one of ['joint_qpos','state_joint'], got keys: {list(metadata.keys())}"
                )
        else:
            state_key_joint = None
            action_key_joint = None

        if args.observation_space == "ee" or args.action_space == "ee":
            # Real-data TCP keys (preferred for UR3 real demos)
            has_real_tcp = ("state_tcp" in metadata) and ("action_tcp" in metadata)
            # Sim-data EE keys (legacy)
            has_sim_ee_legacy = all(
                k in metadata
                for k in ["robot_root_state", "robot_ee_state", "robot_ee_state_target"]
            )
            # Sim-data EE keys (new/simple): absolute EE state per frame (usually 7D)
            has_sim_ee_simple = ("ee_state" in metadata)
            if not has_real_tcp and not has_sim_ee_legacy and not has_sim_ee_simple:
                raise KeyError(
                    f"ee requested but no supported keys found. "
                    f"Need real tcp keys ['state_tcp','action_tcp'] or sim ee keys "
                    f"['robot_root_state','robot_ee_state','robot_ee_state_target'] "
                    f"or ['ee_state']. "
                    f"Got keys: {list(metadata.keys())}"
                )
        else:
            has_real_tcp = False
            has_sim_ee_legacy = False
            has_sim_ee_simple = False

        if state_key_joint is not None:
            data_length = len(metadata[state_key_joint])
        elif has_real_tcp:
            data_length = len(metadata["state_tcp"])
        elif has_sim_ee_simple:
            data_length = len(metadata["ee_state"])
        else:
            data_length = len(metadata["robot_ee_state"])

        rgbs = iio.mimread(os.path.join(demo_dir, "rgb.mp4"))
        task_mask_seq = None
        task_mask_path = os.path.join(demo_dir, "task_mask.npy")
        if os.path.isfile(task_mask_path):
            try:
                task_mask_seq = np.load(task_mask_path)
            except Exception as e:
                print(f"Warning: failed to load {task_mask_path}: {e}")
                task_mask_seq = None
        else:
            missing_task_mask_demo_count += 1
            if args.require_task_mask:
                raise FileNotFoundError(f"Missing required task mask: {task_mask_path}")

        def _resize_mask_nearest(mask_2d: np.ndarray, h: int, w: int) -> np.ndarray:
            src_h, src_w = mask_2d.shape[:2]
            if src_h == h and src_w == w:
                return mask_2d
            ys = np.linspace(0, src_h - 1, h).round().astype(np.int32)
            xs = np.linspace(0, src_w - 1, w).round().astype(np.int32)
            return mask_2d[np.ix_(ys, xs)]

        for i, rgb in enumerate(rgbs):
            if i % downsample_ratio != 0:
                continue

            # you can change state and action here
            if args.observation_space == "joint_pos":
                state = metadata[state_key_joint][i]
                # Apply padding if specified and using joint_pos
                if args.joint_pos_padding > 0 and len(state) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(state))
                    state = np.concatenate([state, padding])
            elif args.observation_space == "ee":
                if has_real_tcp:
                    g = metadata.get("gripper_norm", None)
                    g_i = g[i] if g is not None and i < len(g) else None
                    state = _build_real_tcp7(metadata["state_tcp"][i], gripper_norm=g_i)
                elif has_sim_ee_simple:
                    # Simple sim format: ee_state=[pos3,euler3,gripper1] -> convert to eval-compatible 9D.
                    state = _build_sim_ee9_from_ee_state(metadata["ee_state"][i])
                else:
                    if "transforms" not in globals():
                        raise ImportError("pytorch3d is required for sim ee conversion.")
                    robot_pos, robot_quat = (
                        torch.tensor(metadata["robot_root_state"][i][0:3]),
                        torch.tensor(metadata["robot_root_state"][i][3:7]),
                    )

                    # Convert both current and next EE state into local coordinates
                    local_ee_pos = transforms.quaternion_apply(
                        transforms.quaternion_invert(robot_quat),
                        torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                    )
                    local_ee_quat = transforms.quaternion_multiply(
                        transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state"][i][3:7])
                    )

                    gripper_state = metadata[state_key_joint][i][-2:]
                    state = np.concatenate([local_ee_pos, local_ee_quat, gripper_state])
                    assert state.shape == (9,)
            else:
                raise ValueError(f"Unknown observation space: {args.observation_space}")

            if args.action_space == "joint_pos":
                action = metadata[action_key_joint][i]
                # Apply padding if specified and using joint_pos
                if args.joint_pos_padding > 0 and len(action) < args.joint_pos_padding:
                    padding = np.zeros(args.joint_pos_padding - len(action))
                    action = np.concatenate([action, padding])
            elif args.action_space == "ee":
                if has_real_tcp:
                    g = metadata.get("gripper_norm", None)
                    g_i = g[i] if g is not None and i < len(g) else None
                    action_abs = _build_real_tcp7(metadata["action_tcp"][i], gripper_norm=g_i)
                    if not args.delta_ee:
                        action = action_abs
                    else:
                        state_abs = _build_real_tcp7(metadata["state_tcp"][i], gripper_norm=g_i)
                        action = action_abs.copy()
                        action[:3] = action_abs[:3] - state_abs[:3]
                        action[3:6] = _rotvec_delta_np(state_abs[3:6], action_abs[3:6])
                elif has_sim_ee_simple:
                    # Use next-frame ee_state as action target (last frame reuses itself).
                    j = min(i + 1, data_length - 1)
                    state_abs = _build_sim_ee9_from_ee_state(metadata["ee_state"][i])
                    next_abs = _build_sim_ee9_from_ee_state(metadata["ee_state"][j])
                    if not args.delta_ee:
                        action = next_abs
                    else:
                        action = next_abs.copy()
                        # delta position
                        action[:3] = next_abs[:3] - state_abs[:3]
                        # delta quaternion: q_delta = inv(q_curr) * q_next
                        q_curr = state_abs[3:7]
                        q_next = next_abs[3:7]
                        q_delta = _quat_mul_np(_quat_inv_np(q_curr), q_next)
                        qn = np.linalg.norm(q_delta)
                        if qn > 1e-8:
                            q_delta = q_delta / qn
                        else:
                            q_delta = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                        action[3:7] = q_delta
                        # gripper keeps target absolute width (same convention as legacy sim branch)
                        action[7:9] = next_abs[7:9]
                else:
                    if "transforms" not in globals():
                        raise ImportError("pytorch3d is required for sim ee conversion.")
                    robot_pos, robot_quat = (
                        torch.tensor(metadata["robot_root_state"][i][0:3]),
                        torch.tensor(metadata["robot_root_state"][i][3:7]),
                    )

                    # Convert both current and next EE state into local coordinates
                    local_ee_pos = transforms.quaternion_apply(
                        transforms.quaternion_invert(robot_quat),
                        torch.tensor(metadata["robot_ee_state"][i][0:3]) - robot_pos,
                    )
                    local_next_ee_pos = transforms.quaternion_apply(
                        transforms.quaternion_invert(robot_quat),
                        torch.tensor(metadata["robot_ee_state_target"][i][0:3]) - robot_pos,
                    )

                    local_ee_quat = transforms.quaternion_multiply(
                        transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state"][i][3:7])
                    )
                    local_next_ee_quat = transforms.quaternion_multiply(
                        transforms.quaternion_invert(robot_quat), torch.tensor(metadata["robot_ee_state_target"][i][3:7])
                    )
                    gripper_action = metadata[action_key_joint][i][-2:]

                    if not args.delta_ee:
                        action = np.concatenate([local_next_ee_pos, local_next_ee_quat, gripper_action])
                    else:
                        # Compute the delta in local coordinates
                        local_ee_delta_pos = local_next_ee_pos - local_ee_pos
                        local_ee_delta_quat = transforms.quaternion_multiply(
                            transforms.quaternion_invert(local_ee_quat), local_next_ee_quat
                        )
                        action = np.concatenate([local_ee_delta_pos, local_ee_delta_quat, gripper_action])

                    assert action.shape == (9,), f"Action shape is {action.shape}, expected (9,)"
            else:
                raise ValueError(f"Unknown action space: {args.action_space}")

            action = list(action)
            # Append data to batch arrays
            if task_mask_seq is not None and i < len(task_mask_seq):
                task_mask = np.asarray(task_mask_seq[i]).astype(np.uint8)
            else:
                task_mask = np.ones(rgb.shape[:2], dtype=np.uint8) * 255
                fallback_white_frame_count += 1
            if task_mask.ndim > 2:
                task_mask = task_mask.squeeze()
            if task_mask.shape != rgb.shape[:2]:
                task_mask = _resize_mask_nearest(task_mask, rgb.shape[0], rgb.shape[1])

            head_camera_arrays.append(rgb)
            task_mask_arrays.append(task_mask)
            state_arrays.append(state)
            action_arrays.append(action)
            total_count += 1

        episode_ends_arrays.append(total_count)

        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == len(demo_indices):
            # Convert arrays to NumPy and format head_camera
            head_camera_arrays = np.array(head_camera_arrays)
            head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW
            # print(head_camera_arrays)
            task_mask_arrays = np.array(task_mask_arrays, dtype=np.uint8)
            action_arrays = np.array(action_arrays)
            state_arrays = np.array(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)

            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "head_camera",
                    shape=(0, *head_camera_arrays.shape[1:]),
                    chunks=(batch_size, *head_camera_arrays.shape[1:]),
                    dtype=head_camera_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "task_mask",
                    shape=(0, *task_mask_arrays.shape[1:]),
                    chunks=(batch_size, *task_mask_arrays.shape[1:]),
                    dtype=task_mask_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )

            # Append data to ZARR datasets
            zarr_data["head_camera"].append(head_camera_arrays)
            zarr_data["task_mask"].append(task_mask_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)

            print(f"Batch {current_batch + 1} written with {len(head_camera_arrays)} samples.")

            # Clear arrays for next batch
            head_camera_arrays = []
            task_mask_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1

    if missing_task_mask_demo_count > 0 or fallback_white_frame_count > 0:
        print(
            "[TaskMask Summary] "
            f"missing_demo_masks={missing_task_mask_demo_count}, "
            f"fallback_white_frames={fallback_white_frame_count}"
        )

    # Save metadata to a JSON file
    metadata = {
        "observation_space": args.observation_space,
        "action_space": args.action_space,
        "delta_ee": args.delta_ee,
        "joint_pos_padding": args.joint_pos_padding,
        "task_name": args.task_name,
        "num_episodes": len(demo_indices),
        "downsample_ratio": args.downsample_ratio,
    }

    # Save metadata to zarr group
    for key, value in metadata.items():
        zarr_meta.attrs[key] = value

    # Also save as a separate JSON file for easier access
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()

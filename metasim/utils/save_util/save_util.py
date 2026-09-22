"""Sub-module containing utilities for saving data."""

from __future__ import annotations

import json
import os
import pickle as pkl
from collections import deque
from typing import Any

import imageio as iio
import numpy as np
import torch

from metasim.types import DictEnvState
from metasim.utils.io_util import write_16bit_depth_video
from metasim.utils.kinematics import get_ee_state_from_list


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    return (depth - depth.min()) / (depth.max() - depth.min())


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _as_valid_vec3(x, *, allow_zero: bool = True) -> np.ndarray | None:
    arr = _to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size < 3:
        return None
    v = arr[:3]
    if not np.all(np.isfinite(v)):
        return None
    if not allow_zero and float(np.linalg.norm(v)) <= 1e-6:
        return None
    return v


def _as_valid_intrinsics(x) -> np.ndarray | None:
    arr = _to_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size != 9:
        return None
    k = arr.reshape(3, 3)
    if not np.all(np.isfinite(k)):
        return None
    return k


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _get_camera_basis(camera_cfg: dict[str, Any]):
    cam_pos = _to_numpy(camera_cfg.get("pos"))
    cam_look_at = _to_numpy(camera_cfg.get("look_at"))
    cam_intr = _to_numpy(camera_cfg.get("intrinsics"))
    if cam_pos is None or cam_look_at is None or cam_intr is None:
        return None

    cam_pos = np.asarray(cam_pos, dtype=np.float32).reshape(-1)[:3]
    cam_look_at = np.asarray(cam_look_at, dtype=np.float32).reshape(-1)[:3]
    cam_intr = np.asarray(cam_intr, dtype=np.float32).reshape(3, 3)

    forward = _safe_normalize(cam_look_at - cam_pos)
    up_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_world))) > 0.99:
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _safe_normalize(np.cross(forward, up_world))
    up = _safe_normalize(np.cross(right, forward))
    return cam_pos, right, up, forward, cam_intr


def _project_world_to_pixel(point_xyz: np.ndarray, camera_cfg: dict[str, Any]):
    basis = _get_camera_basis(camera_cfg)
    if basis is None:
        return None
    cam_pos, right, up, forward, cam_intr = basis

    p = np.asarray(point_xyz, dtype=np.float32).reshape(-1)[:3]
    rel = p - cam_pos
    x_cam = float(np.dot(rel, right))
    y_cam = float(np.dot(rel, up))
    z_cam = float(np.dot(rel, forward))
    if z_cam <= 1e-6:
        return None

    fx, fy = float(cam_intr[0, 0]), float(cam_intr[1, 1])
    cx, cy = float(cam_intr[0, 2]), float(cam_intr[1, 2])
    u = fx * (x_cam / z_cam) + cx
    v = cy - fy * (y_cam / z_cam)
    return u, v, z_cam


def _get_task_keywords(task_name: str | None, task_desc: str | None) -> list[str]:
    name = (task_name or "").lower()
    desc = (task_desc or "").lower()
    text = f"{name} {desc}"
    if "close_box" in text or ("close" in text and "box" in text):
        return ["box", "lid", "cover", "flap", "base"]
    if "pick_cube" in text or "stack_cube" in text:
        return ["cube", "block"]
    if "drawer" in text:
        return ["drawer", "handle"]
    # fallback
    return [tok for tok in text.split() if len(tok) >= 3]


def _pick_target_object_key(object_keys: list[str], keywords: list[str]) -> str | None:
    if not object_keys:
        return None
    lower_map = {k: k.lower() for k in object_keys}
    # Prefer non-robot, non-background objects.
    ban = ["robot", "franka", "panda", "table", "floor", "wall", "world", "light", "camera", "ground"]
    candidates = [k for k in object_keys if not any(tok in lower_map[k] for tok in ban)]
    if not candidates:
        candidates = list(object_keys)
    for kw in keywords:
        for k in candidates:
            kl = lower_map[k]
            if kw in kl:
                return k
    return sorted(candidates)[0]


def _label_to_text(label: Any) -> str:
    if isinstance(label, str):
        return label.lower()
    if isinstance(label, bytes):
        try:
            return label.decode("utf-8").lower()
        except Exception:
            return str(label).lower()
    if isinstance(label, dict):
        parts = [f"{k}:{_label_to_text(v)}" for k, v in label.items()]
        return " ".join(parts).lower()
    return str(label).lower()


def _is_robot_label(label_text: str) -> bool:
    robot_keywords = ["robot", "franka", "panda", "arm", "gripper", "finger", "wrist", "link"]
    return any(tok in label_text for tok in robot_keywords)


def _extract_seg_and_labels(camera_state: dict[str, Any]) -> tuple[np.ndarray | None, dict[Any, Any] | None]:
    seg = None
    id2label = None
    # Prefer geom-level segmentation because labels are much more semantic
    # (body/geom names), while instance_id is often just obj_<id>.
    if "instance_seg" in camera_state and "instance_seg_id2label" in camera_state:
        seg = _to_numpy(camera_state.get("instance_seg"))
        id2label = camera_state.get("instance_seg_id2label")
    elif "instance_id_seg" in camera_state and "instance_id_seg_id2label" in camera_state:
        seg = _to_numpy(camera_state.get("instance_id_seg"))
        id2label = camera_state.get("instance_id_seg_id2label")
    if seg is None:
        return None, None
    seg = np.asarray(seg).squeeze()
    if id2label is not None and not isinstance(id2label, dict):
        id2label = dict(id2label)
    return seg, id2label


def _instance_debug_info(
    camera_state: dict[str, Any],
    shape_hw: tuple[int, int] | None,
    seed_uv: tuple[float, float] | None = None,
    topk: int = 8,
) -> dict[str, Any]:
    info: dict[str, Any] = {"has_instance": False}
    seg, id2label = _extract_seg_and_labels(camera_state)
    if seg is None:
        return info
    if shape_hw is not None and seg.shape != shape_hw:
        info["shape_mismatch"] = {"seg": list(seg.shape), "expected": list(shape_hw)}
        return info

    unique_ids = [int(v) for v in np.unique(seg).tolist() if int(v) >= 0]
    area_by_id = {sid: int((seg == sid).sum()) for sid in unique_ids}
    ranked = sorted(unique_ids, key=lambda sid: area_by_id[sid], reverse=True)
    top_ids = ranked[:topk]
    top_items = []
    for sid in top_ids:
        label = id2label.get(sid) if isinstance(id2label, dict) else None
        top_items.append({"id": int(sid), "area": int(area_by_id[sid]), "label": _label_to_text(label)})

    info["has_instance"] = True
    info["unique_count"] = int(len(unique_ids))
    info["top_ids"] = top_items
    if seed_uv is not None:
        u, v = seed_uv
        h, w = seg.shape
        if 0 <= u < w and 0 <= v < h:
            cy = max(0, min(h - 1, int(round(v))))
            cx = max(0, min(w - 1, int(round(u))))
            sid = int(seg[cy, cx])
            info["seed_id"] = int(sid)
            if isinstance(id2label, dict):
                info["seed_label"] = _label_to_text(id2label.get(sid))
    return info


def _mask_from_instance(
    camera_state: dict[str, Any],
    keywords: list[str],
    shape_hw: tuple[int, int],
    target_object: str | None = None,
    seed_uv: tuple[float, float] | None = None,
) -> np.ndarray | None:
    seg, id2label = _extract_seg_and_labels(camera_state)
    if seg is None:
        return None

    if seg.shape != shape_hw:
        return None

    unique_ids = [int(v) for v in np.unique(seg).tolist() if int(v) >= 0]
    if not unique_ids:
        return None

    area_by_id = {sid: int((seg == sid).sum()) for sid in unique_ids}
    label_text_by_id: dict[int, str] = {}
    # NOTE: do not blacklist "world" because many valid IsaacSim labels are full paths
    # like "/World/envs/env_0/.../box_base".
    blacklist = ["collision", " col", "site", "sensor", "light", "camera"]
    if id2label is not None:
        for key, label in id2label.items():
            try:
                sid = int(key)
            except Exception:
                continue
            text = _label_to_text(label)
            # keep behavior close to hdf5_to_mask.py: skip non-semantic geoms
            if any(bad in text for bad in blacklist):
                continue
            label_text_by_id[sid] = text

    robot_ids = {sid for sid, text in label_text_by_id.items() if _is_robot_label(text)}
    candidate_ids = [sid for sid in unique_ids if sid not in robot_ids]
    if not candidate_ids:
        candidate_ids = unique_ids

    selected_ids = []
    target_tokens: list[str] = []
    if target_object:
        target_tokens = [tok for tok in target_object.lower().replace("-", "_").split("_") if tok]

    # First pass: strong target-object match (e.g., box_base).
    if target_tokens:
        for sid in candidate_ids:
            text = label_text_by_id.get(sid, "")
            if text and all(tok in text for tok in target_tokens):
                selected_ids.append(sid)

    # Second pass: keyword match.
    for sid in candidate_ids:
        if sid in selected_ids:
            continue
        text = label_text_by_id.get(sid, "")
        # same refined matching style as your old script:
        # keyword split by spaces, each token must appear in label text.
        if any(all(tok in text for tok in kw.lower().split()) for kw in keywords):
            selected_ids.append(sid)
    if not selected_ids and seed_uv is not None:
        u, v = seed_uv
        h, w = shape_hw
        if 0 <= u < w and 0 <= v < h:
            cy = max(0, min(h - 1, int(round(v))))
            cx = max(0, min(w - 1, int(round(u))))
            sid = int(seg[cy, cx])
            if sid in candidate_ids and sid >= 0:
                selected_ids = [sid]
            else:
                # MuJoCo-style robust fallback: pick dominant instance id around seed.
                # This avoids selecting background id at the exact projected center.
                r = 4
                y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                patch = seg[y0:y1, x0:x1].reshape(-1)
                vals, cnts = np.unique(patch, return_counts=True)
                local_rank = sorted(
                    [
                        (int(vv), int(cc))
                        for vv, cc in zip(vals.tolist(), cnts.tolist())
                        if int(vv) in candidate_ids and int(vv) >= 0
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                img_area = shape_hw[0] * shape_hw[1]
                for sid2, _ in local_rank:
                    area2 = area_by_id.get(sid2, 0)
                    if 20 <= area2 <= int(0.8 * img_area):
                        selected_ids = [sid2]
                        break

    if not selected_ids:
        img_area = shape_hw[0] * shape_hw[1]
        sorted_ids = sorted(candidate_ids, key=lambda sid: area_by_id.get(sid, 0), reverse=True)
        for sid in sorted_ids:
            # Skip near-full-frame blobs (table/background artifacts).
            area = area_by_id.get(sid, 0)
            if 20 <= area <= int(0.6 * img_area):
                selected_ids = [sid]
                break
        # If all candidates are near-full-frame, treat instance seg as unreliable
        # and let downstream depth/projection fallback handle mask generation.
        if not selected_ids:
            return None

    if not selected_ids:
        return None

    # Keep only ids with reasonable area and merge multi-part object ids.
    img_area = shape_hw[0] * shape_hw[1]
    filtered_ids = [sid for sid in selected_ids if 20 <= area_by_id.get(sid, 0) <= int(0.8 * img_area)]
    if filtered_ids:
        selected_ids = filtered_ids

    mask = np.zeros(seg.shape, dtype=np.uint8)
    for sid in selected_ids:
        mask[seg == sid] = 255
    # Safety check: avoid returning almost full-frame masks from noisy segmentation.
    area_ratio = float((mask > 0).sum()) / float(mask.size)
    if area_ratio > 0.85:
        return None
    if (mask > 0).sum() < 20:
        return None
    return mask


def _erase_robot_from_instance_mask(mask: np.ndarray, camera_state: dict[str, Any]) -> np.ndarray:
    seg, id2label = _extract_seg_and_labels(camera_state)
    if seg is None or id2label is None or seg.shape != mask.shape:
        return mask
    out = mask.copy()
    for key, label in id2label.items():
        try:
            sid = int(key)
        except Exception:
            continue
        if _is_robot_label(_label_to_text(label)):
            out[seg == sid] = 0
    return out


def _get_effective_camera_cfg(camera_state: dict[str, Any], fallback_cfg: dict[str, Any] | None) -> dict[str, Any] | None:
    if fallback_cfg is None and ("cam_pos" not in camera_state or "cam_intr" not in camera_state):
        return None

    cfg: dict[str, Any] = {}
    if fallback_cfg is not None:
        cfg.update(fallback_cfg)

    cam_pos = _as_valid_vec3(camera_state.get("cam_pos"), allow_zero=False) if "cam_pos" in camera_state else None
    cam_intr = _as_valid_intrinsics(camera_state.get("cam_intr")) if "cam_intr" in camera_state else None
    cam_look_at = _as_valid_vec3(camera_state.get("cam_look_at")) if "cam_look_at" in camera_state else None

    # Some backends export placeholder cam_pos=[0,0,0] and empty cam_look_at.
    # Prefer fallback camera when runtime values look invalid.
    if cam_pos is not None:
        cfg["pos"] = cam_pos

    if cam_intr is not None:
        cfg["intrinsics"] = cam_intr

    if cam_look_at is not None:
        cfg["look_at"] = cam_look_at

    pos = _as_valid_vec3(cfg.get("pos"), allow_zero=False)
    intr = _as_valid_intrinsics(cfg.get("intrinsics"))
    look_at = _as_valid_vec3(cfg.get("look_at"))

    if pos is None or intr is None:
        return None
    # If look_at is missing, default to world origin (common for static scene cameras).
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    # Avoid degenerate basis when look_at ~= pos.
    if float(np.linalg.norm(look_at - pos)) <= 1e-6:
        look_at = pos + np.array([-1.0, 0.0, -1.0], dtype=np.float32)

    return {"pos": pos, "intrinsics": intr, "look_at": look_at}


def _mask_from_center(shape_hw: tuple[int, int], radius: int) -> np.ndarray:
    h, w = shape_hw
    u = (w - 1) / 2.0
    v = (h - 1) / 2.0
    yy, xx = np.ogrid[:h, :w]
    rr2 = (xx - u) ** 2 + (yy - v) ** 2
    return (rr2 <= float(radius * radius)).astype(np.uint8) * 255


def _mask_from_projected_object(
    env_state: DictEnvState,
    target_object: str | None,
    camera_cfg: dict[str, Any] | None,
    shape_hw: tuple[int, int],
    radius: int,
) -> np.ndarray | None:
    if target_object is None or camera_cfg is None:
        return None
    if "objects" not in env_state or target_object not in env_state["objects"]:
        return None
    obj_pos = _to_numpy(env_state["objects"][target_object].get("pos"))
    if obj_pos is None:
        return None

    proj = _project_world_to_pixel(obj_pos, camera_cfg)
    if proj is None:
        return None
    u, v, _ = proj

    h, w = shape_hw
    if not (0 <= u < w and 0 <= v < h):
        return None

    yy, xx = np.ogrid[:h, :w]
    rr2 = (xx - u) ** 2 + (yy - v) ** 2
    mask = (rr2 <= float(radius * radius)).astype(np.uint8) * 255
    return mask


def _collect_projected_points_for_objects(
    env_state: DictEnvState,
    object_names: list[str],
    camera_cfg: dict[str, Any] | None,
    shape_hw: tuple[int, int],
) -> list[tuple[float, float, float]]:
    if camera_cfg is None:
        return []
    objects = env_state.get("objects", {})
    if not objects:
        return []
    h, w = shape_hw
    points: list[tuple[float, float, float]] = []
    for name in object_names:
        obj_state = objects.get(name)
        if obj_state is None:
            continue

        world_points = []
        if "pos" in obj_state:
            world_points.append(_to_numpy(obj_state.get("pos")))
        body_dict = obj_state.get("body", {})
        if isinstance(body_dict, dict):
            for body_state in body_dict.values():
                if isinstance(body_state, dict) and "pos" in body_state:
                    world_points.append(_to_numpy(body_state.get("pos")))

        for p in world_points:
            if p is None:
                continue
            proj = _project_world_to_pixel(p, camera_cfg)
            if proj is None:
                continue
            u, v, z = float(proj[0]), float(proj[1]), float(proj[2])
            if 0 <= u < w and 0 <= v < h:
                points.append((u, v, z))
    return points


def _mask_from_projected_points(
    shape_hw: tuple[int, int],
    points_uv: list[tuple[float, float, float]],
    point_radius: int,
    bbox_pad: int,
) -> np.ndarray | None:
    if not points_uv:
        return None
    h, w = shape_hw
    yy, xx = np.ogrid[:h, :w]
    out = np.zeros((h, w), dtype=bool)
    rr = max(2, int(point_radius))
    for u, v, _ in points_uv:
        rr2 = (xx - u) ** 2 + (yy - v) ** 2
        out |= rr2 <= float(rr * rr)

    # Do NOT fill an axis-aligned bbox here: that causes a fixed rectangle mask.
    # Instead, connect projected support points softly around their centroid.
    us = np.array([p[0] for p in points_uv], dtype=np.float32)
    vs = np.array([p[1] for p in points_uv], dtype=np.float32)
    cu, cv = float(us.mean()), float(vs.mean())
    rr_center = max(2, int(0.8 * rr))
    rr2_center = (xx - cu) ** 2 + (yy - cv) ** 2
    out |= rr2_center <= float(rr_center * rr_center)

    # Keep bbox_pad as a soft expansion factor only (backward-compatible arg).
    if int(bbox_pad) > 0:
        extra = max(0, int(bbox_pad // 10))
        if extra > 0:
            out = _binary_dilate(out, iterations=extra)

    out = _binary_close(out, iterations=1)
    return (out.astype(np.uint8) * 255)


def _largest_connected_from_seed(binary_mask: np.ndarray, seed_y: int, seed_x: int) -> np.ndarray:
    h, w = binary_mask.shape
    if not (0 <= seed_y < h and 0 <= seed_x < w):
        return np.zeros_like(binary_mask, dtype=bool)
    if not binary_mask[seed_y, seed_x]:
        return np.zeros_like(binary_mask, dtype=bool)

    visited = np.zeros_like(binary_mask, dtype=bool)
    out = np.zeros_like(binary_mask, dtype=bool)
    q = deque()
    q.append((seed_y, seed_x))
    visited[seed_y, seed_x] = True
    out[seed_y, seed_x] = True
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]
    while q:
        y, x = q.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            if binary_mask[ny, nx]:
                out[ny, nx] = True
                q.append((ny, nx))
    return out


def _binary_dilate(mask_bool: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask_bool, dtype=bool)
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            | p[:-2, 1:-1]
            | p[2:, 1:-1]
            | p[1:-1, :-2]
            | p[1:-1, 2:]
            | p[:-2, :-2]
            | p[:-2, 2:]
            | p[2:, :-2]
            | p[2:, 2:]
        )
    return out


def _binary_erode(mask_bool: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = np.asarray(mask_bool, dtype=bool)
    for _ in range(max(0, int(iterations))):
        p = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            p[1:-1, 1:-1]
            & p[:-2, 1:-1]
            & p[2:, 1:-1]
            & p[1:-1, :-2]
            & p[1:-1, 2:]
            & p[:-2, :-2]
            & p[:-2, 2:]
            & p[2:, :-2]
            & p[2:, 2:]
        )
    return out


def _binary_close(mask_bool: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask_bool, iterations=iterations), iterations=iterations)


def _erase_robot_projection(mask: np.ndarray, env_state: DictEnvState, camera_cfg: dict[str, Any], radius: int = 12) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    robots = env_state.get("robots", {})
    for _, robot_state in robots.items():
        points = []
        if "pos" in robot_state:
            points.append(_to_numpy(robot_state["pos"]))
        body_dict = robot_state.get("body", {})
        if isinstance(body_dict, dict):
            for body_state in body_dict.values():
                if isinstance(body_state, dict) and "pos" in body_state:
                    points.append(_to_numpy(body_state["pos"]))
        for p in points:
            if p is None:
                continue
            proj = _project_world_to_pixel(p, camera_cfg)
            if proj is None:
                continue
            u, v, _ = proj
            if not (0 <= u < w and 0 <= v < h):
                continue
            yy, xx = np.ogrid[:h, :w]
            rr2 = (xx - u) ** 2 + (yy - v) ** 2
            out[rr2 <= float(radius * radius)] = 0
    return out


def _mask_from_depth_guided_object(
    env_state: DictEnvState,
    camera_state: dict[str, Any],
    target_object: str | None,
    camera_cfg: dict[str, Any] | None,
    shape_hw: tuple[int, int],
    radius: int,
    depth_tol: float = 0.03,
) -> np.ndarray | None:
    if target_object is None or camera_cfg is None:
        return None
    depth = _to_numpy(camera_state.get("depth"))
    if depth is None:
        return None
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.shape != shape_hw:
        return None

    h, w = shape_hw
    objects = env_state.get("objects", {})
    if not objects:
        return None

    # Use multiple related object seeds (e.g. box_base + box_lid) to avoid
    # returning only one visible face when depth-only segmentation is used.
    names: list[str] = []
    target_tokens = [tok for tok in target_object.lower().replace("-", "_").split("_") if len(tok) >= 3]
    if target_object in objects:
        names.append(target_object)
    for name in objects.keys():
        if name == target_object:
            continue
        lname = name.lower()
        if any(tok in lname for tok in target_tokens):
            names.append(name)
    if not names:
        names = [target_object]
    names = names[:4]

    img_area = h * w
    max_reasonable_area = int(0.35 * img_area)
    use_proj_prior = os.getenv("METASIM_TASK_MASK_USE_PROJ_PRIOR", "1").strip().lower() not in {"0", "false", "no"}
    prior_mask = None
    prior_bool = None
    if use_proj_prior:
        prior_points = _collect_projected_points_for_objects(
            env_state=env_state,
            object_names=names,
            camera_cfg=camera_cfg,
            shape_hw=shape_hw,
        )
        prior_mask = _mask_from_projected_points(
            shape_hw=shape_hw,
            points_uv=prior_points,
            point_radius=max(6, int(radius * 0.7)),
            bbox_pad=max(12, int(radius * 1.6)),
        )
        prior_bool = (prior_mask > 0) if prior_mask is not None else None
        if prior_bool is not None:
            prior_bool = _binary_dilate(prior_bool, iterations=2)

    best_union = np.zeros((h, w), dtype=bool)

    for obj_name in names:
        obj_state = objects.get(obj_name)
        if obj_state is None or "pos" not in obj_state:
            continue
        proj = _project_world_to_pixel(_to_numpy(obj_state["pos"]), camera_cfg)
        if proj is None:
            continue
        u, v, z_proj = proj
        if not (0 <= u < w and 0 <= v < h):
            continue
        cx, cy = int(round(u)), int(round(v))
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))

        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        patch = depth[y0:y1, x0:x1]
        patch_valid = patch[patch > 0]
        if patch_valid.size > 0:
            z_seed = float(np.median(patch_valid))
            mad = float(np.median(np.abs(patch_valid - z_seed)))
        else:
            z_seed = float(z_proj)
            mad = 0.0

        base_tol = float(np.clip(max(float(depth_tol), 0.02, 2.5 * mad), 0.02, 0.12))
        tol_candidates = [
            base_tol,
            min(0.14, base_tol * 1.4),
            min(0.18, base_tol * 1.9),
        ]
        rr_candidates = [max(int(radius * 2), 16), max(int(radius * 3), 24)]

        best_local = np.zeros((h, w), dtype=bool)
        for rr in rr_candidates:
            yy, xx = np.ogrid[:h, :w]
            local = ((xx - u) ** 2 + (yy - v) ** 2) <= float(rr * rr)
            if prior_bool is not None:
                local = local & prior_bool
            for tol in tol_candidates:
                depth_ok = (depth > 0) & (np.abs(depth - z_seed) <= float(tol))
                candidate = local & depth_ok
                connected = _largest_connected_from_seed(candidate, cy, cx)
                if int(connected.sum()) < 20:
                    continue
                connected = _binary_close(connected, iterations=1)
                connected = _largest_connected_from_seed(connected, cy, cx)
                area = int(connected.sum())
                if area < 20 or area > max_reasonable_area:
                    continue
                if area > int(best_local.sum()):
                    best_local = connected

        if int(best_local.sum()) > 0:
            best_union = best_union | best_local

    best_area = int(best_union.sum())
    if prior_bool is not None and int(prior_bool.sum()) > 0:
        prior_area = int(prior_bool.sum())
        # If depth mask is small, only do local expansion constrained by prior.
        # Avoid full prior union, which tends to create fixed rectangular blobs.
        if best_area >= 20 and best_area < max(20, int(0.4 * prior_area)):
            expanded = _binary_dilate(best_union, iterations=2)
            refined = expanded & prior_bool
            refined = _binary_close(refined, iterations=1)
            if int(refined.sum()) > best_area:
                best_union = refined
                best_area = int(best_union.sum())

    if best_area < 20:
        fallback = prior_mask
        if fallback is None:
            fallback = _mask_from_projected_object(env_state, target_object, camera_cfg, shape_hw, radius=max(radius, 18))
        if fallback is None:
            return None
        return fallback

    return (best_union.astype(np.uint8) * 255)


def save_demo(
    save_dir: str,
    demo: list[DictEnvState],
    robot_config,
    task_desc="",
    task_name: str | None = None,
    generate_task_mask: bool = False,
    task_mask_radius: int = 18,
    allow_center_fallback: bool = False,
    task_mask_debug: bool = False,
    camera_cfg: dict[str, Any] | None = None,
):
    """Save a list-state demo sequence and metadata (incl. full EE states)."""
    os.makedirs(save_dir, exist_ok=True)

    robot_name = next(iter(demo[0]["robots"].keys()))
    camera_name = next(iter(demo[0]["cameras"].keys()))

    rgb_frames = []
    depth_frames = []
    task_mask_frames = []
    task_mask_sources = []
    metadata = {
        "depth_min": [],
        "depth_max": [],
        "cam_pos": [],
        "cam_look_at": [],
        "cam_intr": [],
        "cam_extr": [],
        "joint_qpos_target": [],
        "joint_qpos": [],
        "robot_root_state": [],
        "ee_state": [],
        "task_desc": [],
        "task_name": task_name,
        "task_mask_enabled": generate_task_mask,
        "task_mask_radius": int(task_mask_radius),
        "task_mask_allow_center_fallback": bool(allow_center_fallback),
        "task_mask_debug_enabled": bool(task_mask_debug),
        "task_mask_target_object": None,
        "task_mask_keywords": [],
        "task_mask_sources": [],
        "task_mask_area": [],
        "task_mask_debug": [],
    }
    keywords = _get_task_keywords(task_name, task_desc)
    target_object = _pick_target_object_key(list(demo[0].get("objects", {}).keys()), keywords)
    if target_object:
        keywords.extend([tok for tok in target_object.lower().replace("-", "_").split("_") if len(tok) >= 2])
        # de-duplicate but keep order
        keywords = list(dict.fromkeys(keywords))
    metadata["task_mask_target_object"] = target_object
    metadata["task_mask_keywords"] = keywords

    for t, env_state in enumerate(demo):
        robot_state = env_state["robots"][robot_name]
        camera_state = env_state["cameras"][camera_name]

        if "rgb" in camera_state:
            rgb_frames.append(camera_state["rgb"].cpu().numpy())
        if "depth" in camera_state:
            depth_np = camera_state["depth"].cpu().numpy()
            depth_frames.append(_normalize_depth(depth_np))
            metadata["depth_min"].append(float(depth_np.min()))
            metadata["depth_max"].append(float(depth_np.max()))

        metadata["cam_pos"].append(camera_state.get("cam_pos", []).tolist() if "cam_pos" in camera_state else [])
        metadata["cam_look_at"].append(
            camera_state.get("cam_look_at", []).tolist() if "cam_look_at" in camera_state else []
        )
        metadata["cam_intr"].append(camera_state.get("cam_intr", []).tolist() if "cam_intr" in camera_state else [])
        metadata["cam_extr"].append(camera_state.get("cam_extr", []).tolist() if "cam_extr" in camera_state else [])

        metadata["joint_qpos"].append([robot_state["dof_pos"][k] for k in sorted(robot_state["dof_pos"].keys())])

        if next(iter(demo[0]["robots"].values())).get("dof_pos_target", None) is not None:
            if t < len(demo) - 1:
                next_robot_state = demo[t + 1]["robots"][robot_name]
                target_dof_pos = [
                    next_robot_state["dof_pos_target"][k] for k in sorted(next_robot_state["dof_pos_target"].keys())
                ]
            else:
                target_dof_pos = [
                    robot_state["dof_pos_target"][k] for k in sorted(robot_state["dof_pos_target"].keys())
                ]
        else:
            target_dof_pos = None
        metadata["joint_qpos_target"].append(target_dof_pos)

        root_state_flat = torch.cat([
            robot_state["pos"],
            robot_state["rot"],
            robot_state["vel"],
            robot_state["ang_vel"],
        ])
        metadata["robot_root_state"].append(root_state_flat.tolist())

        if generate_task_mask:
            shape_hw = None
            if "rgb" in camera_state:
                rgb_np = _to_numpy(camera_state["rgb"])
                shape_hw = (int(rgb_np.shape[0]), int(rgb_np.shape[1]))
            elif "depth" in camera_state:
                depth_np = _to_numpy(camera_state["depth"])
                shape_hw = (int(depth_np.shape[0]), int(depth_np.shape[1]))

            mask = None
            source = "none"
            effective_camera_cfg = _get_effective_camera_cfg(camera_state, camera_cfg)
            seed_uv = None
            seed_reason = "ok"
            proj = None
            obj_state = None
            obj_pos = None
            target_for_seed = target_object
            objects_now = env_state.get("objects", {})
            if target_for_seed is not None:
                obj_state = objects_now.get(target_for_seed)
            if obj_state is None and objects_now:
                target_for_seed = _pick_target_object_key(list(objects_now.keys()), keywords)
                obj_state = objects_now.get(target_for_seed) if target_for_seed is not None else None
            if obj_state is not None and "pos" in obj_state:
                obj_pos = _to_numpy(obj_state["pos"])
            else:
                seed_reason = "target_missing_or_no_pos"

            if effective_camera_cfg is None:
                seed_reason = "camera_cfg_invalid"
                # Build a minimal camera cfg for debugging/fallback projection.
                fallback_cam = {
                    "pos": _as_valid_vec3((camera_cfg or {}).get("pos"), allow_zero=False),
                    "intrinsics": _as_valid_intrinsics((camera_cfg or {}).get("intrinsics")),
                    "look_at": _as_valid_vec3((camera_cfg or {}).get("look_at")),
                }
                if fallback_cam["pos"] is None:
                    fallback_cam["pos"] = _as_valid_vec3(camera_state.get("cam_pos"), allow_zero=False)
                if fallback_cam["intrinsics"] is None:
                    fallback_cam["intrinsics"] = _as_valid_intrinsics(camera_state.get("cam_intr"))
                if fallback_cam["look_at"] is None:
                    fallback_cam["look_at"] = _as_valid_vec3(obj_pos) if obj_pos is not None else np.array(
                        [0.0, 0.0, 0.0], dtype=np.float32
                    )
                if fallback_cam["pos"] is not None and fallback_cam["intrinsics"] is not None:
                    effective_camera_cfg = fallback_cam
                    seed_reason = "camera_cfg_fallback"

            if obj_pos is not None and effective_camera_cfg is not None:
                proj = _project_world_to_pixel(obj_pos, effective_camera_cfg)
                if proj is not None:
                    seed_uv = (float(proj[0]), float(proj[1]))
                    seed_reason = "ok"
                else:
                    seed_reason = "projection_failed"
            frame_debug: dict[str, Any] | None = None
            if task_mask_debug:
                frame_debug = {
                    "frame": int(t),
                    "target_object": target_object,
                    "target_for_seed": target_for_seed,
                    "seed_uv": list(seed_uv) if seed_uv is not None else None,
                    "seed_reason": seed_reason,
                    "has_effective_camera_cfg": bool(effective_camera_cfg is not None),
                    "obj_pos": (_to_numpy(obj_pos).reshape(-1)[:3].tolist() if obj_pos is not None else None),
                }
                frame_debug["instance"] = _instance_debug_info(camera_state, shape_hw, seed_uv=seed_uv)
            if shape_hw is not None:
                mask = _mask_from_instance(
                    camera_state,
                    keywords,
                    shape_hw,
                    target_object=target_object,
                    seed_uv=seed_uv,
                )
                if mask is not None:
                    mask = _erase_robot_from_instance_mask(mask, camera_state)
                    source = "instance_seg"
                else:
                    mask = _mask_from_depth_guided_object(
                        env_state=env_state,
                        camera_state=camera_state,
                        target_object=target_object,
                        camera_cfg=effective_camera_cfg,
                        shape_hw=shape_hw,
                        radius=int(task_mask_radius),
                    )
                    if mask is not None:
                        source = "depth_guided"
                    else:
                        mask = _mask_from_projected_object(
                            env_state=env_state,
                            target_object=target_object,
                            camera_cfg=effective_camera_cfg,
                            shape_hw=shape_hw,
                            radius=int(task_mask_radius),
                        )
                    if mask is not None and source == "none":
                        source = "projected_center"
            # Avoid over-erasing fallback masks (can collapse to all-black in close_box).
            # Keep robot erasing only for instance segmentation branch.
            if mask is not None and source == "instance_seg":
                if effective_camera_cfg is not None:
                    mask2 = _erase_robot_projection(
                        env_state=env_state,
                        mask=mask,
                        camera_cfg=effective_camera_cfg,
                        radius=max(8, int(task_mask_radius * 0.7)),
                    )
                    # Only accept erase result if still has enough valid pixels.
                    if int((mask2 > 0).sum()) >= max(20, int(0.2 * (mask > 0).sum())):
                        mask = mask2

            # Optional safety fallback: keep disabled by default to avoid unwanted round masks.
            if allow_center_fallback and shape_hw is not None and (mask is None or int((mask > 0).sum()) < 20):
                forced = _mask_from_projected_object(
                    env_state=env_state,
                    target_object=target_for_seed or target_object,
                    camera_cfg=effective_camera_cfg,
                    shape_hw=shape_hw,
                    radius=max(int(task_mask_radius), 18),
                )
                if forced is not None:
                    mask = forced
                    source = "projected_center_forced"
                else:
                    # Last fallback: image center circle.
                    mask = _mask_from_center(shape_hw, radius=max(int(task_mask_radius), 18))
                    source = "image_center_fallback"
            if mask is None and shape_hw is not None:
                mask = np.zeros(shape_hw, dtype=np.uint8)
                if source == "none":
                    source = "no_mask"
            if mask is not None:
                task_mask_frames.append(mask.astype(np.uint8))
                task_mask_sources.append(source)
                metadata["task_mask_area"].append(int((mask > 0).sum()))
                if frame_debug is not None:
                    frame_debug["source"] = source
                    frame_debug["area"] = int((mask > 0).sum())
                    metadata["task_mask_debug"].append(frame_debug)

    # Full EE state only (no separate pos/quat/gripper fields)
    ee_states = get_ee_state_from_list(demo, robot_config, tensorize=True)  # (T, 8)
    metadata["ee_state"] = ee_states.detach().cpu().tolist()
    metadata["task_desc"] = task_desc
    metadata["task_mask_sources"] = task_mask_sources

    if rgb_frames:
        iio.mimsave(os.path.join(save_dir, "rgb.mp4"), rgb_frames, fps=30, quality=10)
    if depth_frames:
        write_16bit_depth_video(os.path.join(save_dir, "depth_uint16.mkv"), depth_frames, fps=30)
        iio.mimsave(
            os.path.join(save_dir, "depth_uint8.mp4"),
            [(d * 255).astype(np.uint8) for d in depth_frames],
            fps=30,
            quality=10,
        )
    if generate_task_mask and task_mask_frames:
        iio.mimsave(
            os.path.join(save_dir, "task_mask.mp4"),
            task_mask_frames,
            fps=30,
            quality=10,
        )
        np.save(os.path.join(save_dir, "task_mask.npy"), np.stack(task_mask_frames, axis=0))

    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
        pkl.dump(metadata, f)

    with open(os.path.join(save_dir, "status.txt"), "w") as f:
        f.write("success")

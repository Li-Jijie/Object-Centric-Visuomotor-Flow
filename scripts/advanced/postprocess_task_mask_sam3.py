#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image
from tqdm import tqdm


def iter_demo_dirs(demo_dir: str | None, demo_root: str | None) -> list[Path]:
    if demo_dir:
        p = Path(demo_dir)
        return [p] if p.is_dir() else []
    if not demo_root:
        return []
    root = Path(demo_root)
    if not root.is_dir():
        return []

    out: list[Path] = []
    for p in sorted(root.rglob("demo_*")):
        if p.is_dir() and (p / "rgb.mp4").exists() and (p / "metadata.json").exists():
            out.append(p)
    return out


def norm_prompt(s: str) -> str:
    return " ".join(s.replace("_", " ").replace("-", " ").split()).strip()


def build_prompts(meta: dict[str, Any], extra: list[str]) -> list[str]:
    prompts: list[str] = []

    target_obj = meta.get("task_mask_target_object")
    if isinstance(target_obj, str) and target_obj.strip():
        prompts.append(norm_prompt(target_obj))

    kws = meta.get("task_mask_keywords", [])
    if isinstance(kws, list):
        for k in kws:
            if isinstance(k, str) and k.strip():
                prompts.append(norm_prompt(k))

    for p in extra:
        if isinstance(p, str) and p.strip():
            prompts.append(norm_prompt(p))

    # close_box default expansion (prefer full object first)
    prompts.extend(["box", "box base", "box lid", "box cover", "box flap"])

    dedup: list[str] = []
    seen = set()
    for p in prompts:
        pl = p.lower()
        if pl and pl not in seen:
            seen.add(pl)
            dedup.append(p)
    return dedup


def read_seed_uv(meta: dict[str, Any], frame_idx: int) -> tuple[float, float] | None:
    debug = meta.get("task_mask_debug", [])
    if not isinstance(debug, list) or frame_idx >= len(debug):
        return None
    row = debug[frame_idx]
    if not isinstance(row, dict):
        return None
    uv = row.get("seed_uv")
    if isinstance(uv, list) and len(uv) == 2:
        try:
            return float(uv[0]), float(uv[1])
        except Exception:
            return None
    return None


def bin_dilate(mask_bool: np.ndarray, iterations: int = 1) -> np.ndarray:
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


def bin_erode(mask_bool: np.ndarray, iterations: int = 1) -> np.ndarray:
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


def smooth_mask(mask_bool: np.ndarray) -> np.ndarray:
    out = bin_dilate(mask_bool, iterations=1)
    out = bin_erode(out, iterations=1)
    out = bin_dilate(out, iterations=1)
    return out


def choose_mask(
    candidates: list[dict[str, Any]],
    h: int,
    w: int,
    seed_uv: tuple[float, float] | None,
    prev_mask: np.ndarray | None,
) -> np.ndarray | None:
    if not candidates:
        return None

    img_area = h * w
    valid = [c for c in candidates if 20 <= int(c["area"]) <= int(0.65 * img_area)]
    if not valid:
        valid = candidates

    # Prefer candidates containing projected seed
    if seed_uv is not None:
        u, v = int(round(seed_uv[0])), int(round(seed_uv[1]))
        if 0 <= u < w and 0 <= v < h:
            inside = [c for c in valid if c["mask"][v, u]]
            if inside:
                inside.sort(key=lambda x: (float(x["score"]), int(x["area"])), reverse=True)
                return inside[0]["mask"]

    # Temporal consistency
    if prev_mask is not None:
        prev_bool = prev_mask > 0
        for c in valid:
            m = c["mask"]
            inter = float((m & prev_bool).sum())
            union = float((m | prev_bool).sum()) + 1e-6
            c["iou_prev"] = inter / union
        valid.sort(
            key=lambda x: (float(x.get("iou_prev", 0.0)), float(x["score"]), int(x["area"])),
            reverse=True,
        )
    else:
        valid.sort(key=lambda x: (float(x["score"]), int(x["area"])), reverse=True)

    return valid[0]["mask"]


def process_demo(
    demo_dir: Path,
    model,
    processor,
    device: str,
    threshold: float,
    mask_threshold: float,
    extra_prompts: list[str],
    strict_prompts: bool,
    max_prompts: int,
    overwrite: bool,
) -> None:
    import torch

    rgb_path = demo_dir / "rgb.mp4"
    meta_path = demo_dir / "metadata.json"
    out_npy = demo_dir / "task_mask.npy"
    out_mp4 = demo_dir / "task_mask.mp4"

    if out_npy.exists() and not overwrite:
        print(f"[SKIP] exists: {out_npy}")
        return

    meta = json.loads(meta_path.read_text())
    if strict_prompts and extra_prompts:
        prompts = [norm_prompt(p) for p in extra_prompts if isinstance(p, str) and p.strip()]
    else:
        prompts = build_prompts(meta, extra_prompts)
    if max_prompts and max_prompts > 0:
        prompts = prompts[:max_prompts]

    frames = list(iio.imiter(rgb_path))
    if not frames:
        print(f"[SKIP] no frames: {rgb_path}")
        return

    print(f"[RUN ] {demo_dir} | frames={len(frames)} | prompts={prompts[:6]}")

    masks_uint8: list[np.ndarray] = []
    prev_mask: np.ndarray | None = None

    for i, frame in enumerate(tqdm(frames, desc=f"SAM3 {demo_dir.name}", leave=False)):
        h, w = frame.shape[:2]
        img = Image.fromarray(frame).convert("RGB")
        seed_uv = read_seed_uv(meta, i)

        cands: list[dict[str, Any]] = []
        for p in prompts:
            try:
                inputs = processor(images=img, text=p, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                post = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=threshold,
                    mask_threshold=mask_threshold,
                    target_sizes=inputs["original_sizes"].tolist(),
                )[0]
            except Exception:
                continue

            scores = post.get("scores")
            masks = post.get("masks")
            if scores is None or masks is None or len(scores) == 0:
                continue

            scores_np = scores.detach().cpu().numpy()
            masks_np = masks.detach().cpu().numpy().astype(bool)

            for j in range(len(scores_np)):
                m = masks_np[j]
                if m.shape != (h, w):
                    continue
                cands.append(
                    {
                        "mask": m,
                        "score": float(scores_np[j]),
                        "area": int(m.sum()),
                    }
                )

        picked = choose_mask(cands, h, w, seed_uv, prev_mask)
        if picked is None:
            if prev_mask is not None:
                picked = prev_mask > 0
            else:
                picked = np.zeros((h, w), dtype=bool)
        else:
            picked = smooth_mask(picked)

        out = picked.astype(np.uint8) * 255
        masks_uint8.append(out)
        prev_mask = out
        if i % 20 == 0:
            print(f"[PROG] {demo_dir.name} frame={i+1}/{len(frames)}", flush=True)

    masks_arr = np.stack(masks_uint8, axis=0)
    np.save(out_npy, masks_arr)
    iio.imwrite(out_mp4, masks_arr, fps=30)

    n = int(masks_arr.shape[0])
    meta["task_mask_enabled"] = True
    meta["task_mask_sources"] = ["sam3_offline"] * n
    meta["task_mask_area"] = [int((m > 0).sum()) for m in masks_arr]
    meta["task_mask_sam3_prompts"] = prompts
    meta["task_mask_sam3_threshold"] = float(threshold)
    meta["task_mask_sam3_mask_threshold"] = float(mask_threshold)

    meta_path.write_text(json.dumps(meta))
    print(f"[DONE] {demo_dir} | mean_area={np.mean(meta['task_mask_area']):.1f}")


def _chunk_round_robin(items: list[Path], n: int) -> list[list[Path]]:
    chunks: list[list[Path]] = [[] for _ in range(n)]
    for i, x in enumerate(items):
        chunks[i % n].append(x)
    return chunks


def _resolve_torch_dtype(name: str, device: str):
    import torch

    n = (name or "").lower().strip()
    if not device.startswith("cuda"):
        return torch.float32
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp32", "float32"}:
        return torch.float32
    # auto
    return torch.float16


def _worker_run(
    worker_id: int,
    device: str,
    demo_dirs: list[str],
    model_path: str,
    threshold: float,
    mask_threshold: float,
    prompts: list[str],
    strict_prompts: bool,
    max_prompts: int,
    overwrite: bool,
    torch_dtype: str,
):
    import torch

    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    print(f"[W{worker_id}] device={device} demos={len(demo_dirs)} dtype={torch_dtype}", flush=True)
    print(f"[W{worker_id}] importing transformers ...", flush=True)
    try:
        from transformers.models.sam3.modeling_sam3 import Sam3Model
        from transformers.models.sam3.processing_sam3 import Sam3Processor
    except Exception:
        from transformers import Sam3Model, Sam3Processor

    dtype = _resolve_torch_dtype(torch_dtype, device)
    print(f"[W{worker_id}] loading SAM3 from {model_path} ...", flush=True)
    model = Sam3Model.from_pretrained(model_path, torch_dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(model_path)
    print(f"[W{worker_id}] model ready", flush=True)

    # Small runtime speed knobs.
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    for d in demo_dirs:
        process_demo(
            demo_dir=Path(d),
            model=model,
            processor=processor,
            device=device,
            threshold=threshold,
            mask_threshold=mask_threshold,
            extra_prompts=prompts,
            strict_prompts=strict_prompts,
            max_prompts=max_prompts,
            overwrite=overwrite,
        )


def main():
    print("[BOOT] starting postprocess_task_mask_sam3.py", flush=True)
    parser = argparse.ArgumentParser(description="Offline SAM3 segmentation for task masks")
    parser.add_argument("--demo-dir", type=str, default=None)
    parser.add_argument("--demo-root", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=os.environ.get("SAM3_MODEL_PATH", ""))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--devices", type=str, default="", help="Comma-separated devices, e.g. cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--strict-prompts", action="store_true", help="Only use --prompt items; disable auto prompt expansion")
    parser.add_argument("--max-prompts", type=int, default=0, help=">0 时只保留前 N 个 prompts 以加速")
    parser.add_argument("--torch-dtype", type=str, default="fp16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print("[BOOT] args parsed", flush=True)
    print(f"[BOOT] demo_dir={args.demo_dir} demo_root={args.demo_root}", flush=True)

    demo_dirs = iter_demo_dirs(args.demo_dir, args.demo_root)
    if not demo_dirs:
        raise SystemExit("No demo dirs found. Provide --demo-dir or --demo-root")

    print(f"[INIT] found {len(demo_dirs)} demo dir(s)", flush=True)

    devices = [x.strip() for x in args.devices.split(",") if x.strip()] if args.devices else []
    if not devices:
        devices = [args.device]
    print(f"[INIT] using devices={devices}", flush=True)

    if len(devices) == 1 or len(demo_dirs) == 1:
        _worker_run(
            worker_id=0,
            device=devices[0],
            demo_dirs=[str(x) for x in demo_dirs],
            model_path=args.model_path,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            prompts=args.prompt,
            strict_prompts=args.strict_prompts,
            max_prompts=args.max_prompts,
            overwrite=args.overwrite,
            torch_dtype=args.torch_dtype,
        )
        return

    chunks = _chunk_round_robin(demo_dirs, len(devices))
    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for wid, (dev, chunk) in enumerate(zip(devices, chunks)):
        if not chunk:
            continue
        p = ctx.Process(
            target=_worker_run,
            kwargs=dict(
                worker_id=wid,
                device=dev,
                demo_dirs=[str(x) for x in chunk],
                model_path=args.model_path,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
                prompts=args.prompt,
                strict_prompts=args.strict_prompts,
                max_prompts=args.max_prompts,
                overwrite=args.overwrite,
                torch_dtype=args.torch_dtype,
            ),
        )
        p.start()
        procs.append(p)

    exit_codes = []
    for p in procs:
        p.join()
        exit_codes.append(p.exitcode)
    if any(code not in (0, None) for code in exit_codes):
        raise SystemExit(f"Some workers failed, exit_codes={exit_codes}")
    print(f"[DONE] all workers finished, exit_codes={exit_codes}", flush=True)


if __name__ == "__main__":
    main()

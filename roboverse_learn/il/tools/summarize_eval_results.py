#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


BOOL_TRUE = {"true", "1", "yes"}


def parse_bool(s: str) -> bool:
    return s.strip().lower() in BOOL_TRUE


def parse_float_ms(s: str) -> float:
    s = s.strip().lower().replace("ms", "")
    return float(s)


@dataclass
class DemoRecord:
    demo_index: int | None = None
    success_once: bool | None = None
    success_end: bool | None = None
    timeout: bool | None = None
    total_steps: int | None = None
    avg_infer_ms: float | None = None
    min_infer_ms: float | None = None
    max_infer_ms: float | None = None


@dataclass
class RunSummary:
    run_dir: str
    task: str
    policy: str
    robot: str
    run_name: str
    num_demos: int
    success_once: int
    success_end: int
    timeout: int
    success_rate_once: float
    success_rate_end: float
    avg_infer_ms: float | None
    std_demo_avg_infer_ms: float | None
    min_infer_ms: float | None
    max_infer_ms: float | None
    eval_tag: str | None = None
    mid_obj_enabled: bool | None = None
    mid_obj_step: int | None = None
    mid_obj_x: float | None = None
    mid_obj_y: float | None = None
    mid_obj_z: float | None = None
    mid_obj_names: str | None = None


LINE_PATTERNS = {
    "demo_index": re.compile(r"^Demo Index:\s*(\d+)\s*$"),
    "success_once": re.compile(r"^SuccessOnce:\s*(\w+)\s*$"),
    "success_end": re.compile(r"^SuccessEnd:\s*(\w+)\s*$"),
    "timeout": re.compile(r"^TimeOut:\s*(\w+)\s*$"),
    "total_steps": re.compile(r"^Total Steps:\s*(\d+)\s*$"),
    "avg_infer": re.compile(r"^Average Inference Time:\s*([0-9.]+)ms\s*$"),
    "min_infer": re.compile(r"^Min Inference Time:\s*([0-9.]+)ms\s*$"),
    "max_infer": re.compile(r"^Max Inference Time:\s*([0-9.]+)ms\s*$"),
}


def parse_demo_txt(path: Path) -> DemoRecord:
    rec = DemoRecord()
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_PATTERNS["demo_index"].match(line)
        if m:
            rec.demo_index = int(m.group(1))
            continue
        m = LINE_PATTERNS["success_once"].match(line)
        if m:
            rec.success_once = parse_bool(m.group(1))
            continue
        m = LINE_PATTERNS["success_end"].match(line)
        if m:
            rec.success_end = parse_bool(m.group(1))
            continue
        m = LINE_PATTERNS["timeout"].match(line)
        if m:
            rec.timeout = parse_bool(m.group(1))
            continue
        m = LINE_PATTERNS["total_steps"].match(line)
        if m:
            rec.total_steps = int(m.group(1))
            continue
        m = LINE_PATTERNS["avg_infer"].match(line)
        if m:
            rec.avg_infer_ms = parse_float_ms(m.group(1))
            continue
        m = LINE_PATTERNS["min_infer"].match(line)
        if m:
            rec.min_infer_ms = parse_float_ms(m.group(1))
            continue
        m = LINE_PATTERNS["max_infer"].match(line)
        if m:
            rec.max_infer_ms = parse_float_ms(m.group(1))
            continue
    return rec


def parse_run_dir_parts(eval_root: Path, run_dir: Path) -> tuple[str, str, str, str] | None:
    rel = run_dir.relative_to(eval_root)
    parts = rel.parts
    if len(parts) >= 5:
        # Current layout:
        # eval/<task>/<policy>/<robot>/<condition>/<ckpt_or_seed>/
        task, policy, robot = parts[:3]
        run_name = "/".join(parts[3:])
        return task, policy, robot, run_name
    if len(parts) == 4:
        # Older layout:
        # eval/<task>/<policy>/<robot>/<run_name>/
        task, policy, robot, run_name = parts
        return task, policy, robot, run_name
    return None


def discover_run_dirs(eval_root: Path, task: str | None, policy: str | None, robot: str | None) -> list[Path]:
    if not eval_root.exists():
        return []

    all_run_dirs: list[Path] = []
    run_dir_candidates = {p.parent for p in eval_root.rglob("final_stats.txt")}
    if not run_dir_candidates:
        run_dir_candidates = {
            p.parent
            for p in eval_root.rglob("*.txt")
            if p.name != "final_stats.txt"
        }
    for run_dir in run_dir_candidates:
        parsed = parse_run_dir_parts(eval_root, run_dir)
        if parsed is None:
            continue
        t, p, r, _ = parsed
        if task and t != task:
            continue
        if policy and p != policy:
            continue
        if robot and r != robot:
            continue
        all_run_dirs.append(run_dir)
    return sorted(all_run_dirs)


def summarize_one_run(run_dir: Path, eval_root: Path) -> RunSummary | None:
    parsed = parse_run_dir_parts(eval_root, run_dir)
    if parsed is None:
        return None
    task, policy, robot, run_name = parsed
    summary_json = run_dir / "eval_summary.json"
    if summary_json.exists():
        try:
            payload = json.loads(summary_json.read_text(encoding="utf-8"))
            results = payload.get("results", {})
            mid = payload.get("perturbations", {}).get("mid_rollout_object", {})
            return RunSummary(
                run_dir=str(run_dir),
                task=str(payload.get("task", task)),
                policy=str(payload.get("policy", policy)),
                robot=str(payload.get("robot", robot)),
                run_name=run_name,
                num_demos=int(results.get("total_completed", results.get("num_demos_evaluated", 0))),
                success_once=int(results.get("success_once", results.get("total_success", 0))),
                success_end=int(results.get("success_end", results.get("total_success", 0))),
                timeout=sum(1 for d in payload.get("demos", []) if d.get("timeout") is True),
                success_rate_once=float(results.get("success_rate", 0.0)),
                success_rate_end=float(results.get("success_rate_end", results.get("success_rate", 0.0))),
                avg_infer_ms=results.get("overall_avg_infer_ms"),
                std_demo_avg_infer_ms=results.get("demo_avg_infer_std_ms"),
                min_infer_ms=results.get("overall_min_infer_ms"),
                max_infer_ms=results.get("overall_max_infer_ms"),
                eval_tag=payload.get("eval_tag"),
                mid_obj_enabled=mid.get("enabled"),
                mid_obj_step=mid.get("step"),
                mid_obj_x=mid.get("x"),
                mid_obj_y=mid.get("y"),
                mid_obj_z=mid.get("z"),
                mid_obj_names=mid.get("obj_names"),
            )
        except Exception:
            pass

    demo_files = sorted(
        [p for p in run_dir.glob("*.txt") if p.name != "final_stats.txt"],
        key=lambda p: p.stem,
    )
    if not demo_files:
        return None

    demos = [parse_demo_txt(p) for p in demo_files]
    n = len(demos)

    success_once = sum(1 for d in demos if d.success_once is True)
    success_end = sum(1 for d in demos if d.success_end is True)
    timeout = sum(1 for d in demos if d.timeout is True)

    avg_vals = [d.avg_infer_ms for d in demos if d.avg_infer_ms is not None]
    min_vals = [d.min_infer_ms for d in demos if d.min_infer_ms is not None]
    max_vals = [d.max_infer_ms for d in demos if d.max_infer_ms is not None]

    avg_infer = mean(avg_vals) if avg_vals else None
    std_demo_avg = pstdev(avg_vals) if len(avg_vals) > 1 else (0.0 if len(avg_vals) == 1 else None)
    min_infer = min(min_vals) if min_vals else None
    max_infer = max(max_vals) if max_vals else None

    return RunSummary(
        run_dir=str(run_dir),
        task=task,
        policy=policy,
        robot=robot,
        run_name=run_name,
        num_demos=n,
        success_once=success_once,
        success_end=success_end,
        timeout=timeout,
        success_rate_once=success_once / n,
        success_rate_end=success_end / n,
        avg_infer_ms=avg_infer,
        std_demo_avg_infer_ms=std_demo_avg,
        min_infer_ms=min_infer,
        max_infer_ms=max_infer,
        eval_tag=None,
        mid_obj_enabled=None,
        mid_obj_step=None,
        mid_obj_x=None,
        mid_obj_y=None,
        mid_obj_z=None,
        mid_obj_names=None,
    )


def fmt_float(v: float | None, nd: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{nd}f}"


def print_table(rows: Iterable[RunSummary]) -> None:
    rows = list(rows)
    if not rows:
        print("No eval runs found.")
        return

    header = (
        "run_name",
        "task",
        "policy",
        "robot",
        "demos",
        "succ_once",
        "succ_end",
        "timeout",
        "sr_once",
        "sr_end",
        "avg_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "mid_obj",
        "mid_step",
        "mid_dx",
    )
    print("\t".join(header))
    for r in rows:
        print(
            "\t".join(
                [
                    r.run_name,
                    r.task,
                    r.policy,
                    r.robot,
                    str(r.num_demos),
                    str(r.success_once),
                    str(r.success_end),
                    str(r.timeout),
                    fmt_float(r.success_rate_once),
                    fmt_float(r.success_rate_end),
                    fmt_float(r.avg_infer_ms, 3),
                    fmt_float(r.std_demo_avg_infer_ms, 3),
                    fmt_float(r.min_infer_ms, 3),
                    fmt_float(r.max_infer_ms, 3),
                    str(r.mid_obj_enabled) if r.mid_obj_enabled is not None else "NA",
                    str(r.mid_obj_step) if r.mid_obj_step is not None else "NA",
                    fmt_float(r.mid_obj_x, 3),
                ]
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RoboVerse IL eval outputs")
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("il_outputs"),
        help="Root folder containing policy/task outputs (default: ./il_outputs)",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--policy", type=str, default=None)
    parser.add_argument("--robot", type=str, default=None)
    parser.add_argument("--latest-only", action="store_true", help="Only summarize the latest run")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")

    args = parser.parse_args()

    # accept both il_outputs root and .../eval root
    if args.eval_root.name == "eval":
        eval_root = args.eval_root
    else:
        eval_root = args.eval_root
        if (args.eval_root / "a2a").exists():
            # aggregate all policy/task eval dirs under il_outputs
            run_summaries: list[RunSummary] = []
            for eval_dir in args.eval_root.glob("*/*/eval"):
                if not eval_dir.is_dir():
                    continue
                run_dirs = discover_run_dirs(eval_dir, args.task, args.policy, args.robot)
                if not run_dirs:
                    continue
                if args.latest_only:
                    run_dirs = [max(run_dirs, key=lambda p: p.stat().st_mtime)]
                for rd in run_dirs:
                    summary = summarize_one_run(rd, eval_dir)
                    if summary is not None:
                        run_summaries.append(summary)
        else:
            run_summaries = []

        if not run_summaries:
            # fallback: assume passed root is directly an eval dir
            run_dirs = discover_run_dirs(args.eval_root, args.task, args.policy, args.robot)
            if args.latest_only and run_dirs:
                run_dirs = [max(run_dirs, key=lambda p: p.stat().st_mtime)]
            run_summaries = [s for s in (summarize_one_run(rd, args.eval_root) for rd in run_dirs) if s is not None]

        run_summaries = sorted(run_summaries, key=lambda x: x.run_name)

        print_table(run_summaries)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps([asdict(x) for x in run_summaries], indent=2), encoding="utf-8")
            print(f"\nSaved JSON summary to: {args.json_out}")
        return

    run_dirs = discover_run_dirs(eval_root, args.task, args.policy, args.robot)
    if args.latest_only and run_dirs:
        run_dirs = [max(run_dirs, key=lambda p: p.stat().st_mtime)]

    run_summaries = [s for s in (summarize_one_run(rd, eval_root) for rd in run_dirs) if s is not None]
    run_summaries = sorted(run_summaries, key=lambda x: x.run_name)

    print_table(run_summaries)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(x) for x in run_summaries], indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


OmegaConf.register_new_resolver("eval", eval, replace=True)


@dataclass
class ParamRow:
    name: str
    total_params: int
    trainable_params: int


def count_params(module) -> ParamRow:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return ParamRow(module.__class__.__name__, total, trainable)


def format_int(v: int) -> str:
    return f"{v:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Count policy total/trainable parameters.")
    parser.add_argument("--policy", required=True, help="policy_config name, e.g. a2a_slot_adapter_add")
    parser.add_argument("--task", default="close_box")
    parser.add_argument("--config-dir", type=Path, default=Path("roboverse_learn/il/configs"))
    parser.add_argument("--dino_slot_ckpt_path", default="")
    parser.add_argument("--resnet_weights", default="")
    parser.add_argument("--robot_dim", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    os.environ["policy_name"] = args.policy
    overrides = [f"task_name={args.task}"]
    if args.dino_slot_ckpt_path:
        overrides.append(f"policy_config.dino_slot_ckpt_path={args.dino_slot_ckpt_path}")
    if args.resnet_weights:
        overrides.append(f"policy_config.obs_encoder.rgb_model.weights={args.resnet_weights}")
    if args.robot_dim is not None:
        overrides.extend(
            [
                f"+policy_config.robot_dim={args.robot_dim}",
                f"shape_meta.obs.agent_pos.shape=[{args.robot_dim}]",
                f"shape_meta.action.shape=[{args.robot_dim}]",
            ]
        )

    config_dir = args.config_dir.resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="default_runner", overrides=overrides)
    OmegaConf.resolve(cfg)

    policy = hydra.utils.instantiate(cfg.policy_config)
    rows = [
        ParamRow(
            "policy_total",
            sum(p.numel() for p in policy.parameters()),
            sum(p.numel() for p in policy.parameters() if p.requires_grad),
        )
    ]
    for name, child in policy.named_children():
        rows.append(
            ParamRow(
                name,
                sum(p.numel() for p in child.parameters()),
                sum(p.numel() for p in child.parameters() if p.requires_grad),
            )
        )

    print("name\ttotal_params\ttrainable_params")
    for row in rows:
        print(f"{row.name}\t{format_int(row.total_params)}\t{format_int(row.trainable_params)}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
        print(f"\nSaved JSON profile to: {args.json_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

mode="reproduce"
gpu="0"
task="close_box"
demo_num="94"
epochs="100"
max_demo_eval="50"
dry_run="0"

usage() {
  cat <<'EOF'
Usage:
  bash roboverse_learn/il/experiments/a2a_experiment_suite.sh [options]

Options:
  --mode <reproduce|isaac_eval|extra>
  --gpu <id>                  GPU id passed to il_run.sh as cuda:<id>
  --task <task_name_set>      default: close_box
  --demo_num <N>              default: 94
  --epochs <N>                default: 100
  --max_demo_eval <N>         default: 50
  --dry_run <0|1>             print commands only
  -h, --help

Env:
  ISAAC_ROOT                  optional local isaacsim standalone path
  ISAAC_USE_XVFB              0/1, default 0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --task) task="$2"; shift 2 ;;
    --demo_num) demo_num="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --max_demo_eval) max_demo_eval="$2"; shift 2 ;;
    --dry_run) dry_run="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

isaac_root="${ISAAC_ROOT:-}"
isaac_use_xvfb="${ISAAC_USE_XVFB:-0}"
gpu_arg="cuda:${gpu}"

run_cmd() {
  echo
  echo "[RUN] $*"
  if [[ "${dry_run}" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_il() {
  local policy_name="$1"
  local sim_set="$2"
  local train_enable="$3"
  local eval_enable="$4"
  local dr_level_eval="$5"

  local cmd=(bash roboverse_learn/il/il_run.sh
    --task_name_set "${task}"
    --policy_name "${policy_name}"
    --sim_set "${sim_set}"
    --demo_num "${demo_num}"
    --num_epochs "${epochs}"
    --train_enable "${train_enable}"
    --eval_enable "${eval_enable}"
    --gpu "${gpu_arg}"
    --dr_level_eval "${dr_level_eval}"
    --max_demo_eval "${max_demo_eval}"
    --isaac_use_xvfb "${isaac_use_xvfb}"
  )

  if [[ -n "${isaac_root}" ]]; then
    cmd+=(--isaac_root "${isaac_root}")
  fi

  run_cmd "${cmd[@]}"
}

run_reproduce() {
  # 1) Baseline A2A full pipeline on MuJoCo
  run_il "a2a" "mujoco" "True" "True" "0"
  # 2) A2A slot baseline on MuJoCo
  run_il "a2a_slot" "mujoco" "True" "True" "0"
  # 3) A2A attention-only baseline on MuJoCo
  run_il "a2a_attn_only" "mujoco" "True" "True" "0"
}

run_isaac_eval() {
  # Evaluation-only, requires trained checkpoints in il_outputs/<policy>/<task>/checkpoints/<epochs>.ckpt
  run_il "a2a" "isaacsim" "False" "True" "1"
  run_il "a2a_slot" "isaacsim" "False" "True" "1"
  run_il "a2a_attn_only" "isaacsim" "False" "True" "1"
}

run_extra() {
  # Example additional experiment grid: 2 policies x 2 DR levels
  for policy in a2a a2a_attn_only; do
    for dr in 0 1; do
      run_il "${policy}" "mujoco" "False" "True" "${dr}"
    done
  done
}

case "${mode}" in
  reproduce) run_reproduce ;;
  isaac_eval) run_isaac_eval ;;
  extra) run_extra ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    usage
    exit 1
    ;;
esac


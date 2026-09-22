#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) collect demos without online task mask (avoid IsaacSim seg/renderer instability)
# 2) run offline SAM3 to generate task_mask.npy/mp4 + metadata fields
#
# Usage:
#   bash scripts/advanced/run_collect_then_sam3.sh
#   bash scripts/advanced/run_collect_then_sam3.sh --task close_box --cust_name close_box_l0_sam3 --num_demo_success 5
#   SAM3_ENV=sam3 SAM3_MODEL_PATH=/path/to/sam3 bash scripts/advanced/run_collect_then_sam3.sh ...

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Point this to a local Isaac Sim installation, e.g.
# export ISAAC_SIM_ROOT=/path/to/isaac-sim
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-}"
ISAAC_PYTHON="${ISAAC_SIM_ROOT:+${ISAAC_SIM_ROOT}/python.sh}"
if [[ -z "${ISAAC_SIM_ROOT}" || ! -x "${ISAAC_PYTHON}" ]]; then
  echo "[ERROR] set ISAAC_SIM_ROOT to an Isaac Sim installation containing python.sh" >&2
  echo "        Example: export ISAAC_SIM_ROOT=/path/to/isaac-sim" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found"
  exit 1
fi

# Pick GPU with maximum free VRAM.
BEST_GPU="$(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2 -nr \
  | head -n1 \
  | cut -d',' -f1 \
  | tr -d ' '
)"
if [[ -z "${BEST_GPU}" ]]; then
  echo "[ERROR] failed to choose GPU"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${BEST_GPU}"
export METASIM_ISAACSIM_DEVICE="cuda:0"
export METASIM_ISAACSIM_ENABLE_SLOW_SEG=0

TASK="close_box"
CUST_NAME="close_box_l0_sam3"
ROBOT="franka"
SIM="isaacsim"
CUSTOM_SAVE_DIR=""

ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --task) TASK="${ARGS[$((i+1))]:-$TASK}" ;;
    --cust_name) CUST_NAME="${ARGS[$((i+1))]:-$CUST_NAME}" ;;
    --robot) ROBOT="${ARGS[$((i+1))]:-$ROBOT}" ;;
    --sim) SIM="${ARGS[$((i+1))]:-$SIM}" ;;
    --custom_save_dir) CUSTOM_SAVE_DIR="${ARGS[$((i+1))]:-}" ;;
  esac
done

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] collecting demos on sim=${SIM}, task=${TASK}, robot=${ROBOT}, cust_name=${CUST_NAME}"

"${ISAAC_PYTHON}" ./scripts/advanced/collect_demo.py \
  --sim "${SIM}" \
  --task "${TASK}" \
  --robot "${ROBOT}" \
  --headless \
  --run_all \
  --num_envs 1 \
  --level 0 \
  --cust_name "${CUST_NAME}" \
  --no-auto-enable-task-mask-for-isaac \
  --no-task-mask-debug \
  "${ARGS[@]}"

if [[ -n "${CUSTOM_SAVE_DIR}" ]]; then
  if [[ "${CUSTOM_SAVE_DIR}" = /* ]]; then
    DEMO_ROOT="${CUSTOM_SAVE_DIR}/success"
  else
    DEMO_ROOT="${ROOT_DIR}/${CUSTOM_SAVE_DIR}/success"
  fi
else
  DEMO_ROOT="${ROOT_DIR}/roboverse_demo/demo_${SIM}/${TASK}-${CUST_NAME}/robot-${ROBOT}/success"
fi

if [[ ! -d "${DEMO_ROOT}" ]]; then
  echo "[ERROR] demo root not found: ${DEMO_ROOT}"
  exit 1
fi

SAM3_ENV="${SAM3_ENV:-sam3}"
SAM3_MODEL_PATH="${SAM3_MODEL_PATH:-}"
if [[ -z "${SAM3_MODEL_PATH}" ]]; then
  echo "[ERROR] set SAM3_MODEL_PATH to the external SAM3 directory" >&2
  exit 1
fi

PY_BIN="$(command -v python || true)"
if [[ -z "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python3 || true)"
fi
if [[ -z "${PY_BIN}" ]]; then
  echo "[ERROR] neither python nor python3 found in current shell"
  exit 1
fi

SAM3_CMD=("${PY_BIN}" ./scripts/advanced/postprocess_task_mask_sam3.py)
if command -v conda >/dev/null 2>&1; then
  CUR_ENV="${CONDA_DEFAULT_ENV:-}"
  if [[ "${CUR_ENV}" != "${SAM3_ENV}" ]] && conda env list | awk '{print $1}' | grep -qx "${SAM3_ENV}"; then
    SAM3_CMD=(conda run --no-capture-output -n "${SAM3_ENV}" python ./scripts/advanced/postprocess_task_mask_sam3.py)
  fi
fi

echo "[INFO] offline SAM3 on ${DEMO_ROOT}"
"${SAM3_CMD[@]}" \
  --demo-root "${DEMO_ROOT}" \
  --model-path "${SAM3_MODEL_PATH}" \
  --device cuda:0 \
  --threshold 0.35 \
  --mask-threshold 0.5 \
  --prompt "box lid" \
  --prompt "box base" \
  --prompt "box cover" \
  --overwrite

echo "[DONE] collect + offline SAM3 mask pipeline finished"

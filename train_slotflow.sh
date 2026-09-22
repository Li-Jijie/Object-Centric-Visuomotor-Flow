#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
SLOT_CKPT="${2:-}"

if [[ -z "$TASK" || -z "$SLOT_CKPT" ]]; then
  echo "Usage: $0 {close_box|pick_cube|push_cube} /path/to/slot_checkpoint.pt" >&2
  exit 2
fi
case "$TASK" in
  close_box|pick_cube|push_cube) ;;
  *) echo "Unsupported task: $TASK" >&2; exit 2 ;;
esac
if [[ ! -f "$SLOT_CKPT" ]]; then
  echo "Slot checkpoint not found: $SLOT_CKPT" >&2
  exit 1
fi

GPU="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DINO_MODEL_PATH="${DINO_MODEL_PATH:-}"
RESNET_WEIGHTS="${TORCHVISION_RESNET18_WEIGHTS:-IMAGENET1K_V1}"

CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$PYTHON_BIN" \
bash roboverse_learn/il/il_run.sh \
  --task_name_set "$TASK" \
  --policy_name a2a_slot_adapter_add \
  --sim_set isaacsim \
  --demo_num 100 \
  --num_epochs 100 \
  --gpu cuda:0 \
  --train_enable True \
  --eval_enable False \
  --dino_model_path "$DINO_MODEL_PATH" \
  --dino_slot_ckpt_path "$SLOT_CKPT" \
  --resnet_weights "$RESNET_WEIGHTS"

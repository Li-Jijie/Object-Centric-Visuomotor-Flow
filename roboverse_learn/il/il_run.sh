#!/bin/bash
# Usage: bash roboverse_learn/il/il_run.sh --task_name_set close_box --policy_name ddpm_dit --dr_level_eval 2 --train_enable False
export PYTHONPATH=$(pwd):$PYTHONPATH

task_name_set="close_box" # Tasks, e.g., close_box, stack_cube, pick_cube
policy_name="a2a"    # IL policy, opts: ddpm_unet, ddpm_dit, ddim_unet, fm_unet, fm_dit, vita, a2a, a2a_slot_adapter_add, score
sim_set="isaacsim"          # Simulator, e.g., mujoco, isaacsim
demo_num=100              # Number of demonstrations to collect, train, and eval

# Training/eval control
train_enable=True   # True for training, False for evaluation
eval_enable=True 

# Training parameters
num_epochs=200
seed=42
gpu=0
obs_space=joint_pos
act_space=joint_pos
delta_ee=0
eval_num_envs=1
eval_max_step=300
max_demo_eval=50
eval_render_mode=raytracing
wall_timeout_sec=0
timeout_kill_after_sec=30
probe_enable=False
probe_sigma=0.0
probe_transform=noise
probe_mode=continuous
probe_step=30
probe_seed=null
action_shock_enable=False
action_shock_sigma=0.0
action_shock_mode=one_shot
action_shock_step=30
action_shock_seed=null
action_shock_clip=True
# Position generalization eval (object position offset, independent from visual DR)
pos_offset_enable=False
pos_offset_x_min=-0.05
pos_offset_x_max=0.05
pos_offset_y_min=-0.05
pos_offset_y_max=0.05
pos_offset_z_min=0.0
pos_offset_z_max=0.0
pos_offset_obj_names=""
pos_offset_per_demo=True
pos_offset_seed=42
mid_obj_perturb_enable=False
mid_obj_perturb_step=40
mid_obj_perturb_x=0.10
mid_obj_perturb_y=0.0
mid_obj_perturb_z=0.0
mid_obj_perturb_obj_names=""
mid_obj_perturb_zero_vel=True
# Robot initial state perturbation (joint noise, independent from visual DR and pos offset)
robot_init_noise_enable=False
robot_init_noise_sigma=0.0
robot_init_noise_per_demo=True
robot_init_noise_seed=42
robot_init_noise_clip=null
cond_obj_weight=1.0
flow_start_mode=""
flow_start_noise_std=""
flow_start_scale=""
flow_sampling_steps=""
slot_semantic_enable=""
slot_geometry_enable=""
object_condition_source=""
dino_two_scale_enable=""
cond_centroid_enable=""
saliency_reg_enable=False
saliency_reg_weight=0.0
saliency_noise_std=0.05
saliency_apply_prob=1.0
saliency_num_steps=1
slot_mask_supervise_enable=False
slot_mask_supervise_weight=0.0
obj_cond_enable=False
obj_cond_weight=0.0
obj_start_enable=False
obj_start_weight=0.0
obj_start_reg_weight=0.0
obj_mask_supervise_enable=False
obj_mask_supervise_weight=0.0
slot_distill_ckpt=""
dino_model_path=""
dino_slot_ckpt_path=""
slot_start_freeze_flow="false"
slot_start_init_from=""
correction_substage="ab"
max_train_episodes=""
val_every=5
slot_distill_freeze_backbone=True
slot_distill_freeze_slot=True
train_batch_size=32
grad_accumulate=1
resnet_weights="IMAGENET1K_V1"
isaac_root="${ISAACSIM_ROOT:-}"
isaac_force_exit_on_close=0
isaac_close_timeout_sec=8
isaac_skip_first_reset=auto
isaac_soft_reset=auto
isaac_wait_use_step=auto
isaac_use_xvfb=auto
xvfb_screen="1280x720x24"
python_bin="${PYTHON_BIN:-python}"
output_suffix=""

# Domain Randomization Level
dr_level_collect=0
dr_level_eval=0
dr_seed=42
robot_name="Franka"

# Parse parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task_name_set)
            task_name_set="$2"
            shift 2
            ;;
        --policy_name)
            policy_name="$2"
            shift 2
            ;;
        --sim_set)
            sim_set="$2"
            shift 2
            ;;
        --demo_num)
            demo_num="$2"
            shift 2
            ;;
        --train_enable)
            train_enable="$2"
            shift 2
            ;;
        --eval_enable)
            eval_enable="$2"
            shift 2
            ;;
        --dr_level_collect)
            dr_level_collect="$2"
            shift 2
            ;;
        --dr_level_eval)
            dr_level_eval="$2"
            shift 2
            ;;
        --randomization_seed)
            dr_seed="$2"
            shift 2
            ;;
        --robot_name)
            robot_name="$2"
            shift 2
            ;;
        --num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --gpu)
            gpu="$2"
            shift 2
            ;;
        --output_suffix)
            output_suffix="$2"
            shift 2
            ;;
        --obs_space)
            obs_space="$2"
            shift 2
            ;;
        --act_space)
            act_space="$2"
            shift 2
            ;;
        --delta_ee)
            delta_ee="$2"
            shift 2
            ;;
        --probe_enable)
            probe_enable="$2"
            shift 2
            ;;
        --probe_sigma)
            probe_sigma="$2"
            shift 2
            ;;
        --probe_transform)
            probe_transform="$2"
            shift 2
            ;;
        --probe_mode)
            probe_mode="$2"
            shift 2
            ;;
        --probe_step)
            probe_step="$2"
            shift 2
            ;;
        --probe_seed)
            probe_seed="$2"
            shift 2
            ;;
        --action_shock_enable)
            action_shock_enable="$2"
            shift 2
            ;;
        --action_shock_sigma)
            action_shock_sigma="$2"
            shift 2
            ;;
        --action_shock_mode)
            action_shock_mode="$2"
            shift 2
            ;;
        --action_shock_step)
            action_shock_step="$2"
            shift 2
            ;;
        --action_shock_seed)
            action_shock_seed="$2"
            shift 2
            ;;
        --action_shock_clip)
            action_shock_clip="$2"
            shift 2
            ;;
        --pos_offset_enable)
            pos_offset_enable="$2"
            shift 2
            ;;
        --pos_offset_x_min)
            pos_offset_x_min="$2"
            shift 2
            ;;
        --pos_offset_x_max)
            pos_offset_x_max="$2"
            shift 2
            ;;
        --pos_offset_y_min)
            pos_offset_y_min="$2"
            shift 2
            ;;
        --pos_offset_y_max)
            pos_offset_y_max="$2"
            shift 2
            ;;
        --pos_offset_z_min)
            pos_offset_z_min="$2"
            shift 2
            ;;
        --pos_offset_z_max)
            pos_offset_z_max="$2"
            shift 2
            ;;
        --pos_offset_obj_names)
            pos_offset_obj_names="$2"
            shift 2
            ;;
        --pos_offset_per_demo)
            pos_offset_per_demo="$2"
            shift 2
            ;;
        --pos_offset_seed)
            pos_offset_seed="$2"
            shift 2
            ;;
        --mid_obj_perturb_enable)
            mid_obj_perturb_enable="$2"
            shift 2
            ;;
        --mid_obj_perturb_step)
            mid_obj_perturb_step="$2"
            shift 2
            ;;
        --mid_obj_perturb_x)
            mid_obj_perturb_x="$2"
            shift 2
            ;;
        --mid_obj_perturb_y)
            mid_obj_perturb_y="$2"
            shift 2
            ;;
        --mid_obj_perturb_z)
            mid_obj_perturb_z="$2"
            shift 2
            ;;
        --mid_obj_perturb_obj_names)
            mid_obj_perturb_obj_names="$2"
            shift 2
            ;;
        --mid_obj_perturb_zero_vel)
            mid_obj_perturb_zero_vel="$2"
            shift 2
            ;;
        --robot_init_noise_enable)
            robot_init_noise_enable="$2"
            shift 2
            ;;
        --robot_init_noise_sigma)
            robot_init_noise_sigma="$2"
            shift 2
            ;;
        --robot_init_noise_per_demo)
            robot_init_noise_per_demo="$2"
            shift 2
            ;;
        --robot_init_noise_seed)
            robot_init_noise_seed="$2"
            shift 2
            ;;
        --robot_init_noise_clip)
            robot_init_noise_clip="$2"
            shift 2
            ;;
        --eval_render_mode)
            eval_render_mode="$2"
            shift 2
            ;;
        --wall_timeout_sec)
            wall_timeout_sec="$2"
            shift 2
            ;;
        --timeout_kill_after_sec)
            timeout_kill_after_sec="$2"
            shift 2
            ;;
        --max_demo_eval)
            max_demo_eval="$2"
            shift 2
            ;;
        --cond_obj_weight)
            cond_obj_weight="$2"
            shift 2
            ;;
        --flow_start_mode)
            flow_start_mode="$2"
            shift 2
            ;;
        --flow_start_noise_std)
            flow_start_noise_std="$2"
            shift 2
            ;;
        --flow_start_scale)
            flow_start_scale="$2"
            shift 2
            ;;
        --flow_sampling_steps)
            flow_sampling_steps="$2"
            shift 2
            ;;
        --slot_semantic_enable)
            slot_semantic_enable="$2"
            shift 2
            ;;
        --slot_geometry_enable)
            slot_geometry_enable="$2"
            shift 2
            ;;
        --object_condition_source)
            object_condition_source="$2"
            shift 2
            ;;
        --dino_two_scale_enable)
            dino_two_scale_enable="$2"
            shift 2
            ;;
        --cond_centroid_enable)
            cond_centroid_enable="$2"
            shift 2
            ;;
        --saliency_reg_enable)
            saliency_reg_enable="$2"
            shift 2
            ;;
        --saliency_reg_weight)
            saliency_reg_weight="$2"
            shift 2
            ;;
        --saliency_noise_std)
            saliency_noise_std="$2"
            shift 2
            ;;
        --saliency_apply_prob)
            saliency_apply_prob="$2"
            shift 2
            ;;
        --saliency_num_steps)
            saliency_num_steps="$2"
            shift 2
            ;;
        --slot_mask_supervise_enable)
            slot_mask_supervise_enable="$2"
            shift 2
            ;;
        --slot_mask_supervise_weight)
            slot_mask_supervise_weight="$2"
            shift 2
            ;;
        --obj_cond_enable)
            obj_cond_enable="$2"
            shift 2
            ;;
        --obj_cond_weight)
            obj_cond_weight="$2"
            shift 2
            ;;
        --obj_start_enable)
            obj_start_enable="$2"
            shift 2
            ;;
        --obj_start_weight)
            obj_start_weight="$2"
            shift 2
            ;;
        --obj_start_reg_weight)
            obj_start_reg_weight="$2"
            shift 2
            ;;
        --obj_mask_supervise_enable)
            obj_mask_supervise_enable="$2"
            shift 2
            ;;
        --obj_mask_supervise_weight)
            obj_mask_supervise_weight="$2"
            shift 2
            ;;
        --slot_distill_freeze_backbone)
            slot_distill_freeze_backbone="$2"
            shift 2
            ;;
        --slot_distill_ckpt)
            slot_distill_ckpt="$2"
            shift 2
            ;;
        --dino_model_path)
            dino_model_path="$2"
            shift 2
            ;;
        --dino_slot_ckpt_path)
            dino_slot_ckpt_path="$2"
            shift 2
            ;;
        --slot_start_freeze_flow)
            slot_start_freeze_flow="$2"
            shift 2
            ;;
        --slot_start_init_from)
            slot_start_init_from="$2"
            shift 2
            ;;
        --correction_substage)
            correction_substage="$2"
            shift 2
            ;;
        --max_train_episodes)
            max_train_episodes="$2"
            shift 2
            ;;
        --val_every)
            val_every="$2"
            shift 2
            ;;
        --slot_distill_freeze_slot)
            slot_distill_freeze_slot="$2"
            shift 2
            ;;
        --train_batch_size)
            train_batch_size="$2"
            shift 2
            ;;
        --grad_accumulate)
            grad_accumulate="$2"
            shift 2
            ;;
        --resnet_weights)
            resnet_weights="$2"
            shift 2
            ;;
        --isaac_root)
            isaac_root="$2"
            shift 2
            ;;
        --isaac_force_exit_on_close)
            isaac_force_exit_on_close="$2"
            shift 2
            ;;
        --isaac_close_timeout_sec)
            isaac_close_timeout_sec="$2"
            shift 2
            ;;
        --isaac_skip_first_reset)
            isaac_skip_first_reset="$2"
            shift 2
            ;;
        --isaac_soft_reset)
            isaac_soft_reset="$2"
            shift 2
            ;;
        --isaac_wait_use_step)
            isaac_wait_use_step="$2"
            shift 2
            ;;
        --isaac_use_xvfb)
            isaac_use_xvfb="$2"
            shift 2
            ;;
        --xvfb_screen)
            xvfb_screen="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            echo "Optional parameters: --task_name_set --policy_name --sim_set --demo_num --train_enable --eval_enable --num_epochs --gpu --output_suffix --obs_space --act_space --delta_ee --probe_enable --probe_sigma --probe_transform --probe_mode --probe_step --probe_seed --action_shock_enable --action_shock_sigma --action_shock_mode --action_shock_step --action_shock_seed --action_shock_clip --pos_offset_enable --pos_offset_x_min --pos_offset_x_max --pos_offset_y_min --pos_offset_y_max --pos_offset_z_min --pos_offset_z_max --pos_offset_obj_names --pos_offset_per_demo --pos_offset_seed --mid_obj_perturb_enable --mid_obj_perturb_step --mid_obj_perturb_x --mid_obj_perturb_y --mid_obj_perturb_z --mid_obj_perturb_obj_names --mid_obj_perturb_zero_vel --eval_render_mode --wall_timeout_sec --timeout_kill_after_sec --max_demo_eval --cond_obj_weight --flow_start_mode --flow_start_noise_std --flow_start_scale --flow_sampling_steps --slot_semantic_enable --slot_geometry_enable --object_condition_source --dino_two_scale_enable --cond_centroid_enable --saliency_reg_enable --saliency_reg_weight --saliency_noise_std --saliency_apply_prob --saliency_num_steps --slot_mask_supervise_enable --slot_mask_supervise_weight --obj_cond_enable --obj_cond_weight --obj_start_enable --obj_start_weight --obj_start_reg_weight --obj_mask_supervise_enable --obj_mask_supervise_weight --slot_distill_freeze_backbone --slot_distill_freeze_slot --dino_model_path --dino_slot_ckpt_path --train_batch_size --grad_accumulate --resnet_weights --isaac_root --isaac_force_exit_on_close --isaac_close_timeout_sec --isaac_skip_first_reset --isaac_soft_reset --isaac_wait_use_step --isaac_use_xvfb --xvfb_screen"
            exit 1
            ;;
    esac
done

# Keep perturbation/noise knobs controlled by CLI args (no hard override here).

# Prefer Isaac Sim bundled Python when available (Python 3.11 + isaacsim packages).
if [ "${python_bin}" = "python" ] && [ -x "/isaac-sim/python.sh" ]; then
  python_bin="/isaac-sim/python.sh"
fi

# # Collect demo
# echo "=== Running collect_demo.sh ==="
# sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^num_demo_success=.*/num_demo_success=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/collect_demo.sh
# sed -i "s/^random_level=.*/random_level=$dr_level_collect/" ./roboverse_learn/il/collect_demo.sh
# bash ./roboverse_learn/il/collect_demo.sh

# Map policy_name to model config
config_name="default_runner"
main_script="./roboverse_learn/il/train.py"

# Policy-specific defaults for slot distillation freezing.
# Keep old behavior for existing policies while making freeze/unfreeze variants explicit.
if [ "${policy_name}" = "a2a_slot_attn_freeze" ]; then
  slot_distill_freeze_backbone=True
  slot_distill_freeze_slot=True
fi
if [ "${policy_name}" = "a2a_slot_attn_unfreeze" ]; then
  slot_distill_freeze_backbone=False
  slot_distill_freeze_slot=False
fi

# if policy_name is ACT
if [ "${policy_name}" = "act" ]; then
    echo "=== Running ACT training and evaluation==="
    sed -i "s/^task_name_set=.*/task_name_set=$task_name_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^sim_set=.*/sim_set=$sim_set/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^expert_data_num=.*/expert_data_num=$demo_num/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^train_enable=.*/train_enable=$train_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_enable=.*/eval_enable=$eval_enable/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^collect_level=.*/collect_level=$dr_level_collect/" ./roboverse_learn/il/policies/act/act_run.sh
    sed -i "s/^eval_level=.*/eval_level=$dr_level_eval/" ./roboverse_learn/il/policies/act/act_run.sh
    bash ./roboverse_learn/il/policies/act/act_run.sh
    echo "=== Completed all data collection, training, and evaluation ==="
    exit 0
fi

# Run training/evaluation for DP/FM/VITA policies
echo "=== Running ${policy_name} ==="

eval_ckpt_name=$num_epochs

extra_base="obs:${obs_space}_act:${act_space}"
dataset_prefixes=()
if [ "${delta_ee}" = 1 ]; then
  # Backward-compatible naming for EE-delta datasets.
  # Prefer *_delta, fallback to older *_delta1.
  dataset_prefixes+=("./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}_delta")
  dataset_prefixes+=("./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}_delta1")
  # Legacy fallback: some datasets were exported with delta_ee=1 but plain ee naming.
  dataset_prefixes+=("./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}")
else
  dataset_prefixes+=("./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}")
fi

is_valid_dataset_dir() {
  local d="$1"
  [ -d "${d}" ] && [ -f "${d}/.zgroup" ] && [ -d "${d}/data" ] && [ -d "${d}/meta" ]
}

# Prefer exact demo_num match first.
dataset_path=""
for prefix in "${dataset_prefixes[@]}"; do
  candidate="${prefix}_${demo_num}.zarr"
  if is_valid_dataset_dir "${candidate}"; then
    dataset_path="${candidate}"
    if [ "${delta_ee}" = 1 ] && [ "${prefix}" = "./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}" ]; then
      echo "Warning: using plain ee naming with delta_ee=1: ${dataset_path}"
    fi
    break
  elif [ -d "${candidate}" ]; then
    echo "Warning: found invalid dataset directory, skipping: ${candidate}"
  fi
done

# Derive eval GPU id from --gpu input.
# Supports formats like "cuda:4" and "4".
eval_gpu_id="${gpu}"
if [[ "${gpu}" == cuda:* ]]; then
    eval_gpu_id="${gpu#cuda:}"
fi
if ! [[ "${eval_gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "Warning: unable to parse eval GPU id from --gpu=${gpu}, fallback to 0 for eval."
    eval_gpu_id=0
fi

if [ -z "${dataset_path}" ] || [ ! -d "${dataset_path}" ]; then
    fallback_found=0
    for prefix in "${dataset_prefixes[@]}"; do
        shopt -s nullglob
        matches=("${prefix}"_*.zarr)
        shopt -u nullglob
        valid_matches=()
        for m in "${matches[@]}"; do
            if is_valid_dataset_dir "${m}"; then
                valid_matches+=("${m}")
            else
                echo "Warning: skipping invalid dataset directory: ${m}"
            fi
        done
        if [ ${#valid_matches[@]} -gt 0 ]; then
            dataset_path=$(printf "%s\n" "${valid_matches[@]}" | sort -V | tail -n 1)
            fallback_found=1
            echo "Warning: requested dataset ${prefix}_${demo_num}.zarr not found."
            echo "Using fallback dataset: ${dataset_path}"
            if [ "${delta_ee}" = 1 ] && [ "${prefix}" = "./data_policy/${task_name_set}${robot_name}L${dr_level_collect}_${extra_base}" ]; then
                echo "Warning: fallback is plain ee naming with delta_ee=1. Verify dataset metadata delta_ee=1."
            fi
            break
        fi
    done

    if [ ${fallback_found} -eq 0 ]; then
        echo "Error: dataset not found for task=${task_name_set}, robot=${robot_name}, dr=${dr_level_collect}, obs=${obs_space}, act=${act_space}, delta_ee=${delta_ee}."
        echo "Searched prefixes:"
        for prefix in "${dataset_prefixes[@]}"; do
            echo "  - ${prefix}_*.zarr"
        done
        exit 1
    fi
fi

export policy_name="${policy_name}"
policy_overrides=()
if [ "${policy_name}" = "a2a_slot_adapter_add" ]; then
if ! echo "${policy_name}" | grep -q "dino"; then
  if [ "${resnet_weights}" = "null" ] || [ "${resnet_weights}" = "none" ] || [ -z "${resnet_weights}" ]; then
    policy_overrides+=("policy_config.obs_encoder.rgb_model.weights=null")
  else
    policy_overrides+=("policy_config.obs_encoder.rgb_model.weights=${resnet_weights}")
  fi

fi
  policy_overrides+=("policy_config.cond_obj_weight=${cond_obj_weight}")
  policy_overrides+=("policy_config.saliency_reg_enable=${saliency_reg_enable}")
  policy_overrides+=("policy_config.saliency_reg_weight=${saliency_reg_weight}")
  policy_overrides+=("policy_config.saliency_noise_std=${saliency_noise_std}")
  policy_overrides+=("policy_config.saliency_apply_prob=${saliency_apply_prob}")
  policy_overrides+=("policy_config.saliency_num_steps=${saliency_num_steps}")
fi
if [ "${policy_name}" = "a2a_objcond" ] || [ "${policy_name}" = "a2a_objcond_start" ]; then
  policy_overrides+=("policy_config.obj_cond_enable=${obj_cond_enable}")
  policy_overrides+=("policy_config.obj_cond_weight=${obj_cond_weight}")
  policy_overrides+=("policy_config.obj_start_enable=${obj_start_enable}")
  policy_overrides+=("policy_config.obj_start_weight=${obj_start_weight}")
  policy_overrides+=("policy_config.obj_start_reg_weight=${obj_start_reg_weight}")
  policy_overrides+=("policy_config.obj_mask_supervise_enable=${obj_mask_supervise_enable}")
  policy_overrides+=("policy_config.obj_mask_supervise_weight=${obj_mask_supervise_weight}")
fi
if [ "${policy_name}" = "a2a" ]; then
  if [ -n "${flow_start_mode}" ]; then
    policy_overrides+=("policy_config.flow_start_mode=${flow_start_mode}")
  fi
  if [ -n "${flow_start_noise_std}" ]; then
    policy_overrides+=("policy_config.flow_start_noise_std=${flow_start_noise_std}")
  fi
  if [ -n "${flow_start_scale}" ]; then
    policy_overrides+=("policy_config.flow_start_scale=${flow_start_scale}")
  fi
  if [ -n "${flow_sampling_steps}" ]; then
    policy_overrides+=("policy_config.flow_matcher.num_sampling_steps=${flow_sampling_steps}")
  fi
fi
if [ "${policy_name}" = "a2a_slot_adapter_add" ]; then
  policy_overrides+=("policy_config.slot_mask_supervise_enable=${slot_mask_supervise_enable}")
  policy_overrides+=("policy_config.slot_mask_supervise_weight=${slot_mask_supervise_weight}")
  policy_overrides+=("policy_config.slot_distill_freeze_backbone=${slot_distill_freeze_backbone}")
  policy_overrides+=("policy_config.slot_distill_freeze_slot=${slot_distill_freeze_slot}")
  if [ -n "${dino_model_path}" ]; then
    policy_overrides+=("policy_config.dino_model_path=${dino_model_path}")
  fi
  if [ -n "${flow_start_mode}" ]; then
    policy_overrides+=("policy_config.flow_start_mode=${flow_start_mode}")
  fi
  if [ -n "${flow_start_noise_std}" ]; then
    policy_overrides+=("policy_config.flow_start_noise_std=${flow_start_noise_std}")
  fi
  if [ -n "${flow_start_scale}" ]; then
    policy_overrides+=("policy_config.flow_start_scale=${flow_start_scale}")
  fi
  if [ -n "${flow_sampling_steps}" ]; then
    policy_overrides+=("policy_config.flow_matcher.num_sampling_steps=${flow_sampling_steps}")
  fi
  if [ -n "${slot_semantic_enable}" ]; then
    policy_overrides+=("policy_config.slot_semantic_enable=${slot_semantic_enable}")
  fi
  if [ -n "${slot_geometry_enable}" ]; then
    policy_overrides+=("policy_config.slot_geometry_enable=${slot_geometry_enable}")
  fi
  if [ -n "${object_condition_source}" ]; then
    policy_overrides+=("policy_config.object_condition_source=${object_condition_source}")
  fi
  if [ -n "${dino_two_scale_enable}" ]; then
    policy_overrides+=("policy_config.dino_two_scale_enable=${dino_two_scale_enable}")
  fi
  if [ -n "${cond_centroid_enable}" ]; then
    policy_overrides+=("policy_config.cond_centroid_enable=${cond_centroid_enable}")
  fi
  if [ "${obs_space}" = "ee" ] && [ "${act_space}" = "ee" ] && [ "${delta_ee}" = "1" ]; then
    policy_overrides+=("+policy_config.delta_action_history_enable=True")
  fi
  if [ -n "${slot_distill_ckpt}" ]; then
    policy_overrides+=("policy_config.slot_distill_ckpt=${slot_distill_ckpt}")
  fi
  if [ -n "${dino_slot_ckpt_path}" ]; then
    policy_overrides+=("policy_config.dino_slot_ckpt_path=${dino_slot_ckpt_path}")
  fi
  if [ "${policy_name}" = "a2a_slot_adapter_add_wo_pos" ] || [ "${policy_name}" = "a2a_slot_adapter_add_dino_wo_pos" ]; then
    policy_overrides+=("policy_config.cond_centroid_enable=False")
  fi
  if [ "${slot_start_freeze_flow}" = "true" ] || [ "${slot_start_freeze_flow}" = "True" ]; then
    policy_overrides+=("policy_config.slot_start_freeze_flow=${slot_start_freeze_flow}")
  fi
  if [ -n "${slot_start_init_from}" ]; then
    policy_overrides+=("policy_config.slot_start_init_from=${slot_start_init_from}")
  fi
  if [ "${correction_substage}" != "ab" ]; then
    policy_overrides+=("policy_config.correction_substage=${correction_substage}")
  fi
  if [ -n "${max_train_episodes}" ]; then
    policy_overrides+=("dataset_config.max_train_episodes=${max_train_episodes}")
  fi
  policy_overrides+=("train_config.training_params.val_every=${val_every}")

fi

  # UR3 robot: 7-D state+action override (outside policy_name checks)
  if [ "${robot_name}" = "UR3" ]; then
    policy_name="${policy_name}_ur3"
    export ROBOT_DIM=7
    policy_overrides+=("+policy_config.robot_dim=7")
    # Keep Hydra shape_meta consistent with 7-DoF UR3 data to avoid 9->7 runtime shape fallback.
    policy_overrides+=("shape_meta.obs.agent_pos.shape=[7]")
    policy_overrides+=("shape_meta.action.shape=[7]")
  fi

  # Ensure downstream Hydra uses the final (possibly UR3-suffixed) policy name.
  export policy_name="${policy_name}"

  # Build output dir after policy_name finalization (important for UR3 suffix).
  run_name="${policy_name}"
  if [ -n "${output_suffix}" ]; then
    run_name="${policy_name}${output_suffix}"
  fi
  output_dir="./il_outputs/${run_name}"
  policy_overrides+=("multi_run.run_dir=${output_dir}/${task_name_set}")
  policy_overrides+=("checkpoint.save_root_dir=${output_dir}/${task_name_set}")
  policy_overrides+=("hydra.run.dir=${output_dir}/${task_name_set}")
  policy_overrides+=("hydra.sweep.dir=${output_dir}/${task_name_set}")
  eval_path="${output_dir}/${task_name_set}/checkpoints/${eval_ckpt_name}.ckpt"
  echo "Output dir: ${output_dir}/${task_name_set}"
  echo "Checkpoint path: $eval_path"

# Reduce allocator fragmentation on long training runs (no-op if already set).
if [ -z "${PYTORCH_CUDA_ALLOC_CONF}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

if { [ "${sim_set}" = "isaacsim" ] || [ "${dr_level_eval}" -gt 0 ] || [ "${dr_level_collect}" -gt 0 ]; }; then
  # Isaac Sim 4.x Python bindings are typically built for CPython 3.10.
  if [ "${python_bin}" = "/isaac-sim/python.sh" ]; then
    py_ver="3.11"
  else
    py_ver="$("${python_bin}" -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")' 2>/dev/null || echo unknown)"
  fi
  if [ "${py_ver}" != "3.10" ]; then
    echo "Warning: current python is ${py_ver}. Isaac Sim 4.x usually requires Python 3.10 in this workflow."
    echo "Recommendation: conda activate a2a (or another py3.10 env) before running."
  fi

  export METASIM_FORCE_EXIT_ON_CLOSE="${isaac_force_exit_on_close}"
  export METASIM_CLOSE_TIMEOUT_SEC="${isaac_close_timeout_sec}"
  if [ "${sim_set}" = "isaacsim" ]; then
    export METASIM_ISAACSIM_DEVICE="cuda:${eval_gpu_id}"
    if [ "${isaac_skip_first_reset}" = "auto" ]; then
      if [ "${obs_space}" = "joint_pos" ]; then
        export METASIM_ISAACSIM_SKIP_FIRST_RESET=0
      else
        export METASIM_ISAACSIM_SKIP_FIRST_RESET=0
      fi
    else
      export METASIM_ISAACSIM_SKIP_FIRST_RESET="${isaac_skip_first_reset}"
    fi
    if [ "${isaac_soft_reset}" = "auto" ]; then
      if [ "${obs_space}" = "joint_pos" ]; then
        export METASIM_ISAACSIM_SOFT_RESET=0
      else
        export METASIM_ISAACSIM_SOFT_RESET=0
      fi
    else
      export METASIM_ISAACSIM_SOFT_RESET="${isaac_soft_reset}"
    fi
    if [ "${isaac_wait_use_step}" = "auto" ]; then
      export METASIM_ISAACSIM_WAIT_USE_STEP=1
    else
      export METASIM_ISAACSIM_WAIT_USE_STEP="${isaac_wait_use_step}"
    fi
    echo "IsaacSim skip first reset: ${METASIM_ISAACSIM_SKIP_FIRST_RESET}"
    echo "IsaacSim soft reset: ${METASIM_ISAACSIM_SOFT_RESET}"
    echo "IsaacSim wait use step: ${METASIM_ISAACSIM_WAIT_USE_STEP}"
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
      echo "Info: keeping CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (user-specified)."
      echo "Info: --gpu uses index within visible devices (e.g., cuda:0 -> first visible GPU)."
    fi
  fi
  if [ -n "${isaac_root}" ] && [ -f "${isaac_root}/setup_conda_env.sh" ]; then
    if [ "${python_bin}" = "/isaac-sim/python.sh" ]; then
      echo "Skip sourcing setup_conda_env.sh because PYTHON_BIN is IsaacSim python.sh."
    else
      echo "Sourcing IsaacSim env: ${isaac_root}"
      # shellcheck source=/dev/null
      source "${isaac_root}/setup_conda_env.sh"
      # Avoid loading IsaacSim's bundled pip_prebundle modules, which can override
      # conda env packages (e.g., typing_extensions/torchvision) and cause ABI/version conflicts.
      export PYTHONPATH="$(echo "${PYTHONPATH}" | tr ':' '\n' \
        | grep -v '/pip_prebundle' \
        | grep -v '/IsaacLab-1.3.0' \
        | grep -v '/IsaacLab-main' \
        | paste -sd ':' -)"
    fi
  else
    echo "Warning: IsaacSim env is not sourced. Set --isaac_root or ISAACSIM_ROOT if you need IsaacSim-backed DR."
  fi
fi

timeout_prefix=()
if [ "${wall_timeout_sec}" -gt 0 ] 2>/dev/null; then
  timeout_prefix=(timeout --signal=TERM --kill-after="${timeout_kill_after_sec}s" "${wall_timeout_sec}s")
  echo "Timeout guard enabled: ${wall_timeout_sec}s (kill-after ${timeout_kill_after_sec}s)"
fi

xvfb_prefix=()
if [ "${sim_set}" = "isaacsim" ]; then
  if [ "${isaac_use_xvfb}" = "auto" ]; then
    isaac_use_xvfb=1
  fi
fi
if [ "${isaac_use_xvfb}" = "1" ]; then
  if command -v xvfb-run >/dev/null 2>&1; then
    xvfb_prefix=(xvfb-run -a -s "-screen 0 ${xvfb_screen}")
    echo "Using xvfb-run with screen ${xvfb_screen}"
  else
    echo "Warning: xvfb-run not found, continue without virtual display."
  fi
fi

run_prefix=()
if [ ${#timeout_prefix[@]} -gt 0 ]; then
  run_prefix+=("${timeout_prefix[@]}")
fi
if [ ${#xvfb_prefix[@]} -gt 0 ]; then
  run_prefix+=("${xvfb_prefix[@]}")
fi

"${run_prefix[@]}" "${python_bin}" ${main_script} --config-name=${config_name}.yaml \
task_name=${task_name_set} \
"dataset_config.zarr_path=${dataset_path}" \
dataset_config.batch_size=${train_batch_size} \
train_config.training_params.seed=${seed} \
train_config.training_params.num_epochs=${num_epochs} \
train_config.training_params.device=${gpu} \
train_config.training_params.gradient_accumulate_every=${grad_accumulate} \
train_config.dataloader.batch_size=${train_batch_size} \
train_config.val_dataloader.batch_size=${train_batch_size} \
eval_config.policy_runner.obs.obs_type=${obs_space} \
eval_config.policy_runner.action.action_type=${act_space} \
eval_config.policy_runner.action.delta=${delta_ee} \
eval_config.eval_args.task=${task_name_set} \
eval_config.eval_args.max_step=${eval_max_step} \
eval_config.eval_args.num_envs=${eval_num_envs} \
eval_config.eval_args.sim=${sim_set} \
eval_config.eval_args.level=${dr_level_eval} \
+eval_config.eval_args.randomization_seed=${dr_seed} \
eval_config.eval_args.render_mode=${eval_render_mode} \
eval_config.eval_args.gpu_id=${eval_gpu_id} \
eval_config.eval_args.probe_enable=${probe_enable} \
eval_config.eval_args.probe_sigma=${probe_sigma} \
eval_config.eval_args.probe_transform=${probe_transform} \
eval_config.eval_args.probe_mode=${probe_mode} \
eval_config.eval_args.probe_step=${probe_step} \
eval_config.eval_args.probe_seed=${probe_seed} \
eval_config.eval_args.action_shock_enable=${action_shock_enable} \
eval_config.eval_args.action_shock_sigma=${action_shock_sigma} \
eval_config.eval_args.action_shock_mode=${action_shock_mode} \
eval_config.eval_args.action_shock_step=${action_shock_step} \
eval_config.eval_args.action_shock_seed=${action_shock_seed} \
eval_config.eval_args.action_shock_clip=${action_shock_clip} \
eval_config.eval_args.pos_offset_enable=${pos_offset_enable} \
eval_config.eval_args.pos_offset_x_min=${pos_offset_x_min} \
eval_config.eval_args.pos_offset_x_max=${pos_offset_x_max} \
eval_config.eval_args.pos_offset_y_min=${pos_offset_y_min} \
eval_config.eval_args.pos_offset_y_max=${pos_offset_y_max} \
eval_config.eval_args.pos_offset_z_min=${pos_offset_z_min} \
eval_config.eval_args.pos_offset_z_max=${pos_offset_z_max} \
eval_config.eval_args.pos_offset_obj_names="${pos_offset_obj_names}" \
eval_config.eval_args.pos_offset_per_demo=${pos_offset_per_demo} \
eval_config.eval_args.pos_offset_seed=${pos_offset_seed} \
eval_config.eval_args.mid_obj_perturb_enable=${mid_obj_perturb_enable} \
eval_config.eval_args.mid_obj_perturb_step=${mid_obj_perturb_step} \
eval_config.eval_args.mid_obj_perturb_x=${mid_obj_perturb_x} \
eval_config.eval_args.mid_obj_perturb_y=${mid_obj_perturb_y} \
eval_config.eval_args.mid_obj_perturb_z=${mid_obj_perturb_z} \
eval_config.eval_args.mid_obj_perturb_obj_names="${mid_obj_perturb_obj_names}" \
eval_config.eval_args.mid_obj_perturb_zero_vel=${mid_obj_perturb_zero_vel} \
eval_config.eval_args.robot_init_noise_enable=${robot_init_noise_enable} \
eval_config.eval_args.robot_init_noise_sigma=${robot_init_noise_sigma} \
eval_config.eval_args.robot_init_noise_per_demo=${robot_init_noise_per_demo} \
eval_config.eval_args.robot_init_noise_seed=${robot_init_noise_seed} \
eval_config.eval_args.robot_init_noise_clip=${robot_init_noise_clip} \
${policy_overrides[@]} \
+eval_config.eval_args.max_demo=${max_demo_eval} \
train_enable=${train_enable} \
eval_enable=${eval_enable} \
eval_path=${eval_path}
py_exit_code=$?

if [ ${py_exit_code} -ne 0 ]; then
  echo "=== Pipeline failed (python exit code: ${py_exit_code}) ==="
  exit ${py_exit_code}
fi

echo "=== Completed all data collection, training, and evaluation ==="

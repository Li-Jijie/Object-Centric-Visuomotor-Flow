# [CoRL 2026] Object-Centric Conditioning for Visuomotor Flow Matching

<p align="center">
  <img src="assets/teaser.png" width="95%" alt="SlotFlow overview">
</p>

Official repository for the paper **Object-Centric Conditioning for Visuomotor Flow Matching**.

[Jijie Li](https://li-jijie.github.io)<sup>1,2</sup>, <strong>Xu Yang</strong><sup>1,2</sup>, <strong>Junhong Zou</strong><sup>1,2</sup>, <strong>Chunhai Zhao</strong><sup>3</sup>, <strong>Chaoyang Zhao</strong><sup>1,3,&#42;</sup>, [Zhen Lei](https://www.cbsr.ia.ac.cn/users/zlei/)<sup>1,2,4,5</sup>, and [Xiangyu Zhu](https://xiangyuzhu-open.github.io/homepage/)<sup>1,2,&#42;</sup>

<sup>1</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br>
<sup>2</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems, Institute of Automation, Chinese Academy of Sciences<br>
<sup>3</sup>Foundation Model Research Center, Institute of Automation, Chinese Academy of Sciences<br>
<sup>4</sup>Centre for Artificial Intelligence and Robotics, Hong Kong Institute of Science & Innovation, Chinese Academy of Sciences<br>
<sup>5</sup>School of Computer Science and Engineering, Faculty of Innovation Engineering, Macau University of Science and Technology<br>
<sup>*</sup>Corresponding authors

**[Paper](PAPER_LINK)** · **[Project Page](PROJECT_PAGE_LINK)**

> **Release status:** The repository is being prepared for public release. Code, checkpoints, and reproduction instructions will be added progressively.

## Pipeline

SlotFlow combines object-centric conditioning with history-conditioned flow matching for robust and efficient visuomotor manipulation.

<p align="center">
  <img src="assets/pipeline.png" width="90%" alt="SlotFlow pipeline">
</p>

## Environment

SlotFlow is an extension of the Action-to-Action (A2A) flow-matching policy and uses a RoboVerse-based environment. Training and SAM3 mask generation require Linux with an NVIDIA GPU. Isaac Sim is required only for collecting demonstrations; training from an existing Zarr dataset does not require Isaac Sim.

```bash
git clone https://github.com/Li-Jijie/Object-Centric-Visuomotor-Flow.git
cd Object-Centric-Visuomotor-Flow
pip install -e .
pip install -r roboverse_learn/il/policies/a2a/requirements.txt
```

Install Isaac Sim separately if you need to collect demonstrations, and provide its local path when doing so:

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim
```

The collection script expects `$ISAAC_SIM_ROOT/python.sh` to be the Isaac Sim Python launcher.

For the shared simulator environment, Docker configuration, and baseline setup, refer to the upstream A2A project:

[A2A Flow Matching](https://github.com/JIAjindou/A2A_Flow_Matching)

> **Acknowledgment.** Our implementation is built upon and substantially adapts the [A2A Flow Matching](https://github.com/JIAjindou/A2A_Flow_Matching) codebase. We retain its RoboVerse-based training infrastructure and extend the policy with SlotFlow's frozen-DINO object-centric conditioning and slot-distillation components.

## External Models

Keep external model files outside this repository:

```bash
export MODEL_ROOT=/path/to/slotflow_models
export DINO_MODEL_PATH="$MODEL_ROOT/dinov3-s"
export SAM3_MODEL_PATH="$MODEL_ROOT/sam3"
export TORCHVISION_RESNET18_WEIGHTS="$MODEL_ROOT/resnet18-f37072fd.pth"
```

Download DINOv3 ViT-S/16 after accepting the upstream terms:

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir "$DINO_MODEL_PATH"
```

SAM3 is used only to generate offline task masks:

```bash
hf auth login
hf download facebook/sam3 --local-dir "$SAM3_MODEL_PATH"
```

## Data Preparation

The released tasks are `close_box`, `pick_cube`, and `push_cube`. To collect demonstrations and generate masks offline with SAM3:

```bash
export ISAAC_SIM_ROOT=/path/to/isaac-sim
SAM3_MODEL_PATH="$SAM3_MODEL_PATH" \
bash scripts/advanced/run_collect_then_sam3.sh \
  --task close_box \
  --cust_name close_box_l0_sam3 \
  --robot franka \
  --sim isaacsim \
  --num_demo_success 100
```

For existing demonstrations, run the postprocessor directly:

```bash
python scripts/advanced/postprocess_task_mask_sam3.py \
  --demo-root /path/to/success \
  --model-path "$SAM3_MODEL_PATH" \
  --device cuda:0 \
  --threshold 0.35 \
  --mask-threshold 0.5 \
  --overwrite
```

Convert a successful-demo directory to Zarr:

```bash
python roboverse_learn/il/data2zarr_dp.py \
  --task_name close_boxFrankaL0_obs:joint_pos_act:joint_pos \
  --expert_data_num 100 \
  --metadata_dir /path/to/close_box-success \
  --observation_space joint_pos \
  --action_space joint_pos \
  --require_task_mask
```

## Training

Train a frozen-DINO slot head for each task. The camera-ready configuration uses 448-pixel inputs, DINO layer 9, two slots, and a two-stage coarse-to-fine head:

```bash
python slot_distill/train_slot_on_frozen_dino.py \
  --preset close_box_recon_s_bgaug \
  --demo-root /path/to/close_box-success \
  --dino-model-path "$DINO_MODEL_PATH" \
  --device cuda:0 \
  --output "$MODEL_ROOT/slot_close_box.best_mass.pt" \
  --log-dir "$MODEL_ROOT/slot_logs_close_box" \
  --img-size 448 \
  --dino-layer 9 \
  --two-scale-enable
```

Then train SlotFlow:

```bash
export CUDA_VISIBLE_DEVICES=0
bash train_slotflow.sh close_box "$MODEL_ROOT/slot_close_box.best_mass.pt"
```

The final configuration is `roboverse_learn/il/configs/policy_config/a2a_slot_adapter_add.yaml`. Outputs, datasets, masks, and checkpoints are intentionally excluded from version control.

## Release Scope

This repository contains the SlotFlow training implementation, data-conversion utilities, and offline SAM3 mask pipeline. It does not include pretrained weights, demonstration datasets, checkpoints, evaluation outputs, Docker images, or real-robot deployment code.

## Reproducing the Paper

The experiments reported in the paper were conducted in an internal Docker-based environment on Linux GPU servers. This public release provides the SlotFlow training implementation, task-specific configuration files, and data-processing utilities used in our workflow. The original environment also contains internal infrastructure and data assets that cannot be redistributed. Consequently, we have not yet performed a complete end-to-end reproduction from this public release in a fresh environment. Users may need to adapt simulator installation, external model locations, and dataset paths to their local Linux setup.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{li2026objectcentric,
  title     = {Object-Centric Conditioning for Visuomotor Flow Matching},
  author    = {Li, Jijie and Yang, Xu and Zou, Junhong and Zhao, Chunhai and Zhao, Chaoyang and Lei, Zhen and Zhu, Xiangyu},
  booktitle = {Proceedings of the Conference on Robot Learning},
  year      = {2026}
}
```

## License

The original SlotFlow code in this repository is released under the [MIT License](LICENSE). Portions derived from third-party projects retain their original notices and licenses; the bundled Apache-2.0 license text is available in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). DINOv3 and SAM3 weights are not redistributed and remain subject to their upstream terms.

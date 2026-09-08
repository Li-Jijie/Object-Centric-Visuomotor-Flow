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

SlotFlow is built on the Action-to-Action (A2A) flow-matching framework and uses the same RoboVerse-based environment. For environment installation, simulator setup, Docker configuration, and baseline usage, please follow the official A2A repository:

[A2A Flow Matching](https://github.com/JIAjindou/A2A_Flow_Matching)

After the A2A environment is configured, the SlotFlow-specific code and commands will be provided in this repository.

## Data and Checkpoints

> **TODO:** Add dataset preparation instructions, download links, license information, and checkpoint URLs.

Expected resource layout:

```text
resources/
├── datasets/
├── checkpoints/
└── assets/
```

Please do not commit large checkpoints or private datasets directly to the repository. Link them from a release page or an external storage service instead.

## Quick Start

> **TODO:** Replace the commands below with the tested commands from the released implementation.

```bash
# Train SlotFlow
python scripts/train.py --config configs/slotflow.yaml

# Evaluate on a benchmark task
python scripts/evaluate.py --config configs/eval_close_box.yaml \
    --checkpoint /path/to/checkpoint

# Generate qualitative visualizations
python scripts/visualize.py --config configs/eval_close_box.yaml \
    --checkpoint /path/to/checkpoint
```

## Reproducing the Paper

The release will include configuration files and scripts for Close Box, Pick Cube, and Push Cube evaluation; visual, kinematic, and spatial perturbation protocols; core component ablations; reviewer-motivated controls; mid-rollout object displacement experiments; and real-world UR3 evaluation, subject to hardware and dataset availability.

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

> **TODO:** Add the final open-source license and any third-party software notices before publication.

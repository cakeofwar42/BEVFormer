<div align="center">

# BEVFormer: a Cutting-edge Baseline for Camera-based Detection

</div>

https://user-images.githubusercontent.com/27915819/161392594-fc0082f7-5c37-4919-830a-2dd423c1d025.mp4

> **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers**, ECCV 2022  
> - [Paper in arXiv](http://arxiv.org/abs/2203.17270) | [Paper in Chinese](https://drive.google.com/file/d/1dKnD6gUHhBXZ8gT733cIU_A7dHEEzNTP/view?usp=sharing) | [OpenDriveLab](https://opendrivelab.com/)  
> - [Slides in English](https://docs.google.com/presentation/d/1fTjuSKpj_-KRjUACr8o5TbKetAXlWrmpaorVvfCTZUA/edit?usp=sharing) | [Occupancy and BEV Perception Talk Slides](https://docs.google.com/presentation/d/1U7wVi2_zJxM-EMqLVqC4zJ12ItUgS7ZcsXp7zQ1fkvc/edit?usp=sharing)  
> - [Blog in Chinese](https://www.zhihu.com/question/521842610/answer/2431585901) | [Video Talk](https://www.bilibili.com/video/BV12t4y1t7Lq?share_source=copy_web) and [Slides](https://docs.google.com/presentation/d/1NNeikhDPkgT14G1D_Ih7K3wbSN0DkvhO9wlAMx3CIcM/edit?usp=sharing) (in Chinese)  
> - [BEV Perception Survey](https://arxiv.org/abs/2209.05324) (Accepted by PAMI) | [Github repo](https://github.com/OpenDriveLab/BEVPerception-Survey-Recipe)

---

## Table of contents
1. [News](#news)
2. [Overview](#overview)
3. [Repository structure](#repository-structure)
4. [Core components and modules](#core-components-and-modules)
5. [Supported workflows](#supported-workflows)
6. [Environment and training requirements](#environment-and-training-requirements)
7. [Data preparation](#data-preparation)
8. [Python usage examples](#python-usage-examples)
9. [Command-line recipes](#command-line-recipes)
10. [Model zoo](#model-zoo)
11. [Citation](#bibtex)
12. [Acknowledgement](#acknowledgement)

---

## News
- **2022/6/16** – Added memory-friendly BEVFormer configurations. Please pull the latest code.  
- **2022/6/13** – Initial release of BEVFormer with **51.7%** NDS on nuScenes.  
- **2022/5/23** – **BEVFormer++** reached **1st** on the [Waymo Open Dataset 3D Camera-Only Detection Challenge](https://waymo.com/open/challenges/2022/3d-camera-only-detection/).  
- **2022/3/10** – Camera-only BEVFormer achieved **56.9% NDS** on the [nuScenes Detection Task](https://nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera).

## Overview
BEVFormer learns a unified bird's-eye-view (BEV) representation from multi-camera images with a spatiotemporal transformer. Spatial cross-attention aggregates features across all cameras, while temporal self-attention fuses the BEV history, enabling state-of-the-art camera-only 3D detection accuracy.

![method](figs/arch.png "model arch")

## Repository structure
The repository combines OpenMMLab components with custom BEVFormer plugins. The high-level layout is:

```
BEVFormer/
├── docs/                  # Installation, dataset preparation, getting-started guides
├── figs/                  # Method figures used in documentation
├── projects/              # Custom BEVFormer plugin package
│   ├── configs/           # Training & evaluation configuration files
│   └── mmdet3d_plugin/    # Plugins extending MMDetection3D
├── tools/                 # Training, testing, and dataset conversion scripts
└── README.md              # (This file)
```

Inside the plugin package you will find the model, data, and runtime extensions that make BEVFormer work end-to-end:

```
projects/mmdet3d_plugin/
├── bevformer/             # BEVFormer-specific detectors, heads, hooks, APIs
├── core/                  # Custom evaluation hooks and geometry utilities
├── datasets/              # Dataset registry, samplers, and pipelines
├── dd3d/                  # Adapted DD3D components for data augmentation
└── models/                # Shared backbones, optimizers, and utility layers
```

## Core components and modules
- **Detectors and heads** – `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py` implements the spatiotemporal BEVFormer detector. It wraps an `MVXTwoStageDetector`, adds grid masking, caches temporal BEV features, and exposes `forward_train`, `forward_test`, and `simple_test` for detection outputs.【F:projects/mmdet3d_plugin/bevformer/detectors/bevformer.py†L7-L292】  Additional detector variants (e.g., fp16, V2) live in the same folder.
- **Training APIs** – `projects/mmdet3d_plugin/bevformer/apis/train.py` exposes `custom_train_model`, delegating to `custom_train_detector` so BEVFormer can register bespoke evaluation hooks during training.【F:projects/mmdet3d_plugin/bevformer/apis/train.py†L7-L67】  The lower-level routine in `apis/mmdet_train.py` builds dataloaders, wraps the model for distributed execution, registers optimizer and evaluation hooks, and drives the runner.【F:projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py†L6-L199】
- **Dataset build utilities** – `projects/mmdet3d_plugin/datasets/builder.py` defines `build_dataloader` with custom distributed samplers and worker seeding, plus `custom_build_dataset` for recursively composing dataset wrappers used in configs.【F:projects/mmdet3d_plugin/datasets/builder.py†L1-L111】【F:projects/mmdet3d_plugin/datasets/builder.py†L118-L166】  Dataset pipelines (loading, augmentations, formatting) are implemented under `projects/mmdet3d_plugin/datasets/pipelines/`.
- **Evaluation hooks** – `projects/mmdet3d_plugin/bevformer/apis/test.py` adds `custom_multi_gpu_test`, augmenting MMDetection3D evaluation so results can be gathered deterministically across devices and optionally return mask predictions.【F:projects/mmdet3d_plugin/bevformer/apis/test.py†L25-L161】
- **Configuration library** – `projects/configs/` collects ready-to-use training recipes covering different model scales (tiny, small, base) and BEVFormerV2 variants.
- **Command-line tools** – `tools/train.py` (single/multi GPU training), `tools/test.py` (evaluation), `tools/dist_train.sh` / `tools/dist_test.sh` (distributed launchers), and `tools/create_data.py` (dataset conversion) provide reproducible entry points.【F:tools/train.py†L33-L239】【F:projects/mmdet3d_plugin/bevformer/apis/test.py†L45-L114】【F:tools/create_data.py†L1-L196】

## Supported workflows
BEVFormer relies on MMDetection3D APIs but ships specialized wrappers to streamline common tasks:

1. **Model construction** – `tools/train.py` loads a configuration, imports any plugins, and builds the detector via `build_model(cfg.model, ...)` before initializing weights.【F:tools/train.py†L105-L224】  The same path can be reused in notebooks.
2. **Dataset creation** – Config-defined datasets are instantiated with `build_dataset(cfg.data.train)` and optional validation splits are deep-copied to share pipelines.【F:tools/train.py†L225-L238】  For standalone usage, prefer `custom_build_dataset` from the plugin builder.【F:projects/mmdet3d_plugin/datasets/builder.py†L118-L166】
3. **Training loop** – `custom_train_model` routes detectors to `custom_train_detector`, which wires data loaders, distributed wrappers, optimizer hooks (including fp16), evaluation hooks, and user-defined hooks before invoking `runner.run(...)`.【F:projects/mmdet3d_plugin/bevformer/apis/train.py†L11-L35】【F:projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py†L25-L199】
4. **Evaluation & testing** – `tools/test.py` mirrors training setup, building datasets/dataloaders and delegating to `custom_multi_gpu_test` for deterministic result aggregation.【F:projects/mmdet3d_plugin/bevformer/apis/test.py†L45-L114】  Each forward pass returns BEV embeddings plus 3D bounding box predictions packaged under the `pts_bbox` key.【F:projects/mmdet3d_plugin/bevformer/detectors/bevformer.py†L263-L292】
5. **Data conversion** – `tools/create_data.py` implements converters for nuScenes, Lyft, Waymo, KITTI, and indoor datasets. For nuScenes, `nuscenes_data_prep` generates temporal info files and optional 2D annotations used by BEVFormer training.【F:tools/create_data.py†L52-L113】

## Environment and training requirements
Follow `docs/install.md` to create a reproducible environment: Python 3.8, PyTorch ≥1.9.1 with CUDA 11.1, MMCV 1.4.0, MMDetection 2.14.0, MMDetection3D v0.17.1 (installed from source), MMSegmentation 0.14.1, plus auxiliary libraries such as Detectron2, timm, einops, fvcore, and scientific Python packages.【F:docs/install.md†L1-L65】  Pretrained checkpoints (e.g., `r101_dcn_fcos3d_pretrain.pth`) should be placed under `ckpts/` before launching training.【F:docs/install.md†L52-L63】

## Data preparation
nuScenes is the primary dataset supported out-of-the-box. Download the official v1.0 release plus CAN bus expansion, unzip the CAN bus data into `data/can_bus`, and run:

```bash
python tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes \
    --version v1.0 \
    --canbus ./data
```
This produces `nuscenes_infos_temporal_{train,val}.pkl`, which are referenced by the provided configs. A typical workspace layout is illustrated in `docs/prepare_dataset.md` and replicated below for convenience.【F:docs/prepare_dataset.md†L3-L41】

```
BEVFormer/
├── ckpts/                          # pretrained weights
├── data/
│   ├── can_bus/
│   └── nuscenes/
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       ├── v1.0-test/
│       ├── v1.0-trainval/
│       ├── nuscenes_infos_temporal_train.pkl
│       └── nuscenes_infos_temporal_val.pkl
```

## Python usage examples
All examples assume the BEVFormer repository is on `PYTHONPATH`, dependencies are installed, and data/checkpoints follow the structure above.

### 1. Load a pretrained model and run inference
```python
from mmcv import Config
from mmdet3d.apis import init_model, inference_detector

config_path = 'projects/configs/bevformer/bevformer_base.py'
checkpoint_path = 'ckpts/bevformer_r101_dcn_24ep.pth'
cfg = Config.fromfile(config_path)
cfg.model.pretrained = None  # avoid re-loading backbone weights
cfg.data.test.test_mode = True

model = init_model(cfg, checkpoint_path, device='cuda:0')
results = inference_detector(model, 'data/nuscenes/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402925702460.jpg')

# Each image returns a dict with BEV predictions under 'pts_bbox'.
pts_bbox = results[0]['pts_bbox']
boxes_3d = pts_bbox['boxes_3d']        # 3D boxes in LiDAR coordinates
scores_3d = pts_bbox['scores_3d']      # detection confidences
labels_3d = pts_bbox['labels_3d']      # semantic classes
```
`simple_test` assembles these structures via `bbox3d2result`, so `boxes_3d` can be directly projected to BEV or camera views for visualization.【F:projects/mmdet3d_plugin/bevformer/detectors/bevformer.py†L271-L292】  For quick visualization, convert the box tensor to numpy and plot it on the BEV grid or overlay the 2D projections on the source images.

### 2. Visualize detections in BEV
```python
import numpy as np
import matplotlib.pyplot as plt

bev_boxes = boxes_3d.corners[:, [0, 1]]  # X-Y footprint for each box
for footprint in bev_boxes:
    polygon = np.vstack([footprint, footprint[0]])  # close the loop
    plt.plot(polygon[:, 0], polygon[:, 1], 'r-')
plt.gca().set_aspect('equal')
plt.title('BEVFormer detections (top-down)')
plt.show()
```

### 3. Build a model programmatically
```python
from mmcv import Config
from mmdet3d.models import build_model

cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()
```
`build_model` mirrors the logic used in `tools/train.py`, letting you introspect or modify submodules before training.【F:tools/train.py†L219-L224】

### 4. Construct datasets for experimentation
```python
from mmcv import Config
from projects.mmdet3d_plugin.datasets import custom_build_dataset

cfg = Config.fromfile('projects/configs/bevformer/bevformer_tiny.py')
train_dataset = custom_build_dataset(cfg.data.train)
val_dataset = custom_build_dataset(cfg.data.val)
```
`custom_build_dataset` recursively resolves dataset wrappers like `RepeatDataset` and `CBGSDataset`, matching the behavior used during validation in the custom trainer.【F:projects/mmdet3d_plugin/datasets/builder.py†L118-L166】【F:projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py†L155-L179】

### 5. Train inside Python (instead of CLI)
```python
from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.bevformer.apis import custom_train_model

cfg = Config.fromfile('projects/configs/bevformer/bevformer_small.py')
model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.init_weights()
train_dataset = build_dataset(cfg.data.train)

custom_train_model(
    model,
    train_dataset,
    cfg,
    distributed=False,
    validate=True,
)
```
`custom_train_model` adds BEVFormer-specific evaluation behavior on top of the standard MMDetection3D training loop.【F:projects/mmdet3d_plugin/bevformer/apis/train.py†L11-L35】  The internal `custom_train_detector` wires up dataloaders, optimization hooks, and validation just like the CLI script.【F:projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py†L25-L199】

## Command-line recipes
- **Single/Distributed training** – `python tools/train.py <config>` or `bash tools/dist_train.sh <config> <num_gpus>` for multi-GPU jobs.【F:tools/train.py†L33-L239】
- **Evaluation / prediction** – `python tools/test.py <config> <checkpoint> --eval bbox` or the distributed variant `bash tools/dist_test.sh <config> <checkpoint> <num_gpus>`; both call into `custom_multi_gpu_test` for synchronized metric computation.【F:projects/mmdet3d_plugin/bevformer/apis/test.py†L45-L114】
- **Dataset conversion** – `python tools/create_data.py nuscenes --root-path ...` (see [Data preparation](#data-preparation) for full arguments).【F:tools/create_data.py†L52-L113】

## Model zoo

| Backbone | Method | Lr Schd | NDS| mAP|memroy | Config | Download |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| R50 | BEVFormer-tiny_fp16 | 24ep | 35.9|25.7 | - |[config](projects/configs/bevformer_fp16/bevformer_tiny_fp16.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.log) |
| R50 | BEVFormer-tiny | 24ep | 35.4|25.2 | 6500M |[config](projects/configs/bevformer/bevformer_tiny.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-small | 24ep | 47.9|37.0 | 10500M |[config](projects/configs/bevformer/bevformer_small.py) |[model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.log) |
| [R101-DCN](https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth)  | BEVFormer-base | 24ep | 51.7|41.6 |28500M |[config](projects/configs/bevformer/bevformer_base.py) | [model](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth)/[log](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.log) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1-base | 24ep | 42.6 | 35.1 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py) | [model/log](https://drive.google.com/drive/folders/1nts_1XxAagCEN_Ub7W2f-507SiDdVS_u?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1-base | 48ep | 43.9 | 35.9 | 23952M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-base-48ep.py) | [model/log](https://drive.google.com/drive/folders/1nts_1XxAagCEN_Ub7W2f-507SiDdVS_u?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1 | 24ep | 45.3 | 38.1 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-24ep.py) | [model/log](https://drive.google.com/drive/folders/1uVzQCJq6gYbRLhBde09yzEBeU5l1hAxk?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t1 | 48ep | 46.5 | 39.5 | 37579M |[config](projects/configs/bevformerv2/bevformerv2-r50-t1-48ep.py) | [model/log](https://drive.google.com/drive/folders/1uVzQCJq6gYbRLhBde09yzEBeU5l1hAxk?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t2 | 24ep | 51.8 | 42.0 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-24ep.py) | [model/log](https://drive.google.com/drive/folders/1bSyuFWxfJSIidGV7bC8jx2NR7idRN9-s?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t2 | 48ep | 52.6 | 43.1 | 38954M |[config](projects/configs/bevformerv2/bevformerv2-r50-t2-48ep.py) | [model/log](https://drive.google.com/drive/folders/1bSyuFWxfJSIidGV7bC8jx2NR7idRN9-s?usp=sharing) |
| [R50](https://drive.google.com/file/d/1JTVcrFcOFdPp7rtZ6K__SfF0Np15vXL7/view?usp=sharing)  | BEVformerV2-t8 | 24ep | 55.3 | 46.0 | 40392M |[config](projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py) | [model/log](https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto?usp=sharing) |

Baidu Driver also hosts the BEVFormerV2 checkpoints and logs: [link](https://pan.baidu.com/s/1ynzlAt1DQbH8NkqmisatTw?pwd=fdcv).

## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@article{li2022bevformer,
  title={BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author={Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2203.17270},
  year={2022},
}
@article{Yang2022BEVFormerVA,
  title={BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision},
  author={Chenyu Yang and Yuntao Chen and Haofei Tian and Chenxin Tao and Xizhou Zhu and Zhaoxiang Zhang and Gao Huang and Hongyang Li and Y. Qiao and Lewei Lu and Jie Zhou and Jifeng Dai},
  journal={ArXiv},
  year={2022},
}
```

## Acknowledgement
Many thanks to these excellent open source projects:
- [dd3d](https://github.com/TRI-ML/dd3d)
- [detr3d](https://github.com/WangYueFt/detr3d)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)

### ↳ Stargazers
[![Stargazers repo roster for @nastyox/Repo-Roster](https://reporoster.com/stars/fundamentalvision/BEVFormer)](https://github.com/fundamentalvision/BEVFormer/stargazers)

### ↳ Forkers
[![Forkers repo roster for @nastyox/Repo-Roster](https://reporoster.com/forks/fundamentalvision/BEVFormer)](https://github.com/fundamentalvision/BEVFormer/network/members)

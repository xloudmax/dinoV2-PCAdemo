# DINOv2 PCA Demo

欢迎来到 **DINOv2 PCA Demo** 项目！本项目旨在展示如何使用 DINOv2 框架中的 Vision Transformer (ViT-G/14) 模型进行特征提取、基于主成分分析（PCA）的降维以及可视化。通过本项目，您可以轻松复现实验，并在您的研究中进一步探索和扩展我们的工作。

## 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
  - [克隆仓库](#克隆仓库)
  - [安装依赖](#安装依赖)
- [预训练模型](#预训练模型)
- [数据准备](#数据准备)
  - [ImageNet-1k](#imagenet-1k)
  - [ImageNet-22k](#imagenet-22k)
- [训练](#训练)
  - [快速设置：在 ImageNet-1k 上训练 DINOv2 ViT-L/16](#快速设置在-imagenet-1k-上训练-dinov2-vit-l16)
  - [长时间设置：在 ImageNet-22k 上训练 DINOv2 ViT-L/14](#长时间设置在-imagenet-22k-上训练-dinov2-vit-l14)
- [评估](#评估)
  - [k-NN 分类](#knn-分类)
  - [逻辑回归分类](#逻辑回归分类)
  - [线性分类](#线性分类)
- [使用预训练模型](#使用预训练模型)
- [笔记本](#笔记本)
  - [深度估计](#深度估计)
  - [语义分割](#语义分割)
- [许可证](#许可证)
- [贡献](#贡献)
- [引用](#引用)

## 项目简介

**DINOv2 PCA Demo** 是一个基于 DINOv2 框架的示例项目，展示了如何使用预训练的 Vision Transformer (ViT-G/14) 模型进行图像特征提取，并通过 PCA 进行降维和可视化。此项目旨在确保实验的可复现性，并为进一步的研究提供便利。

## 快速开始

### 克隆仓库

首先，克隆本仓库到本地：

```bash
git clone https://github.com/xloudmax/dinoV2-PCAdemo.git
cd dinov2-feature-extraction
```

### 安装依赖

建议使用 **conda** 环境进行安装，具体步骤如下：

#### 使用 Conda（推荐）

1. 克隆仓库并进入目录：

    ```bash
    git clone https://github.com/xloudmax/dinoV2-PCAdemo.git
    cd dinov2-feature-extraction
    ```

2. 创建并激活 conda 环境：

    ```bash
    conda env create -f conda.yaml
    conda activate dinov2
    ```

#### 使用 Pip

如果不使用 conda，可以通过 pip 安装依赖：

```bash
pip install -r requirements.txt
```

对于密集任务（如深度估计和语义分割），还需要安装额外的依赖：

##### 使用 Conda（推荐）

```bash
conda env create -f conda-extras.yaml
conda activate dinov2-extras
```

##### 使用 Pip

```bash
pip install -r requirements.txt -r requirements-extras.txt
```

## 预训练模型

本项目通过 PyTorch Hub 提供了多个预训练的 DINOv2 模型。请按照以下说明加载所需的模型：

```python
import torch

# DINOv2
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# DINOv2 with registers
dinov2_vits14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
dinov2_vitl14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
dinov2_vitg14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
```

有关如何安装 PyTorch 的详细说明，请访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)。强烈建议安装带有 CUDA 支持的 PyTorch 版本以提高性能。

## 数据准备

### ImageNet-1k

确保数据集的根目录包含以下内容：

```
<ROOT>/test/ILSVRC2012_test_00000001.JPEG
<ROOT>/test/[..]
<ROOT>/test/ILSVRC2012_test_00100000.JPEG
<ROOT>/train/n01440764/n01440764_10026.JPEG
<ROOT>/train/[...]
<ROOT>/train/n15075141/n15075141_9993.JPEG
<ROOT>/val/n01440764/ILSVRC2012_val_00000293.JPEG
<ROOT>/val/[...]
<ROOT>/val/n15075141/ILSVRC2012_val_00049174.JPEG
<ROOT>/labels.txt
```

此外，数据集实现期望在额外目录下存在一些元数据文件：

```
<EXTRA>/class-ids-TRAIN.npy
<EXTRA>/class-ids-VAL.npy
<EXTRA>/class-names-TRAIN.npy
<EXTRA>/class-names-VAL.npy
<EXTRA>/entries-TEST.npy
<EXTRA>/entries-TRAIN.npy
<EXTRA>/entries-VAL.npy
```

可以通过以下 Python 代码生成这些元数据文件：

```python
from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="<ROOT>", extra="<EXTRA>")
    dataset.dump_extra()
```

**注意**：根目录 (`root`) 和额外目录 (`extra`) 不必是不同的目录。

### ImageNet-22k

请根据您的本地设置调整数据集类以匹配 ImageNet-22k 数据集。

## 训练

### 快速设置：在 ImageNet-1k 上训练 DINOv2 ViT-L/16

在一个 SLURM 集群环境中，使用 Submitit 在 4 个 A100-80GB 节点（32 个 GPU）上运行 DINOv2 训练：

```bash
python dinov2/run/train/train.py \
    --nodes 4 \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

训练时间约为 1 天，最终的检查点应在 k-NN 评估中达到 81.6%，在线性评估中达到 82.9%。

训练代码每 12500 次迭代在 `eval` 文件夹中保存教师模型的权重以进行评估。

### 长时间设置：在 ImageNet-22k 上训练 DINOv2 ViT-L/14

在一个 SLURM 集群环境中，使用 Submitit 在 12 个 A100-80GB 节点（96 个 GPU）上运行 DINOv2 训练：

```bash
python dinov2/run/train/train.py \
    --nodes 12 \
    --config-file dinov2/configs/train/vitl14.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet22k:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

训练时间约为 3.3 天，最终的检查点应在 k-NN 评估中达到 82.0%，在线性评估中达到 84.5%。

训练代码每 12500 次迭代在 `eval` 文件夹中保存教师模型的权重以进行评估。

## 评估

训练代码会定期保存教师模型的权重。要评估模型，请在单个节点上运行以下评估：

### k-NN 分类

```bash
python dinov2/run/eval/knn.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/knn \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### 逻辑回归分类

```bash
python dinov2/run/eval/log_regression.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/logreg \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

### 线性分类

```bash
python dinov2/run/eval/linear.py \
    --config-file <PATH/TO/OUTPUT/DIR>/config.yaml \
    --pretrained-weights <PATH/TO/OUTPUT/DIR>/eval/training_24999/teacher_checkpoint.pth \
    --output-dir <PATH/TO/OUTPUT/DIR>/eval/training_24999/linear \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## 使用预训练模型

我们发布了不同模型权重的预训练版本，您可以通过以下命令进行评估：

```bash
python dinov2/run/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_pretrain.yaml \
    --pretrained-weights https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth \
    --train-dataset ImageNet:split=TRAIN:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET> \
    --val-dataset ImageNet:split=VAL:root=<PATH/TO/DATASET>:extra=<PATH/TO/DATASET>
```

## 笔记本

DINOV2提供了一些笔记本，帮助社区利用这些模型和代码：

- **深度估计**：如何加载和使用深度头部，结合匹配的骨干网络，通过 mmcv 实现。
- **语义分割**：如何加载和使用语义分割头部，结合匹配的骨干网络，通过 mmcv 实现；以及如何加载和使用基于 Mask2Former 的在 ADE20K 上训练的分割模型。

## 引用

```bibtex
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
感谢meta的支持！
---

**注意事项：**

- 请将 `<PATH/TO/OUTPUT/DIR>` 和 `<PATH/TO/DATASET>` 替换为您实际的输出目录和数据集路径。
- 确保所有命令在适当的环境中运行，并根据您的硬件配置调整参数（如节点数量、GPU 数量等）。
- 详细的安装和使用说明请参考仓库内的 [Documentation](https://github.com/xloudmax/dinoV2-PCAdemo.git)。
- 为了保证代码的可复现性，请严格按照上述步骤进行操作。

---

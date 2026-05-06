# Skin Lesion Classification

生物医学图像处理 Project 2：皮肤病变图像三分类。

任务是读取皮肤病变图像和对应 mask，预测每张图属于：

- `mel`: melanoma，黑色素瘤
- `nv`: melanocytic nevus，黑色素细胞痣
- `vasc`: vascular lesion，血管性病变

最终需要生成 `output.csv`：

```csv
image_id,dx
1,nv
2,mel
3,vasc
```

## Project Goal

本项目不只做单一深度学习分类器，而是按三条线推进：

1. 传统图像处理 + 机器学习：利用 mask 提取病灶区域，构造颜色、纹理、形状特征，并训练 SVM、Random Forest、XGBoost 等模型。
2. 深度学习：使用预训练 ResNet18、MobileNetV2 或 EfficientNet-B0 微调三分类。
3. 融合方法：结合传统特征、深度特征或模型预测概率，提高泛化能力。

第一周先集中完成机器学习路线，保证项目有稳定 baseline。

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── run.py
├── data/
│   └── README.md
├── docs/
│   ├── TEAM_TASKS.md
│   └── EXPERIMENT_PLAN.md
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── features_color.py
│   ├── features_shape.py
│   ├── features_texture.py
│   ├── features.py
│   ├── train_ml.py
│   └── utils.py
├── scripts/
│   └── check_dataset.py
├── experiments/
└── outputs/
    ├── metrics/
    ├── models/
    └── figures/
```

## Data Placement

不要把课程数据图片上传到 GitHub。请每个人在本地按下面结构放数据：

```text
data/Data_Proj2/
  image/
    1.jpg
    2.jpg
    ...
  mask/
    mask_1.jpg
    mask_2.jpg
    ...
  label.csv
```

`label.csv` 格式：

```csv
image_id,dx
1,nv
2,nv
3,mel
```

## Installation

建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

检查数据：

```bash
python scripts/check_dataset.py --data_dir data/Data_Proj2
```

训练机器学习 baseline：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all
```

生成预测文件：

```bash
python run.py --input_dir data/Data_Proj2 --output_csv output.csv
```

## Metrics

组内实验统一记录：

- Accuracy
- Macro-F1
- Balanced Accuracy
- Confusion Matrix
- Per-class Precision / Recall / F1

由于本地数据类别不平衡，调参时优先看 `macro-F1` 和 `balanced accuracy`，不要只看 accuracy。

## Team Workflow

- `main` 分支只放稳定版本。
- 每个人从自己的分支开发，例如：
  - `feature/color`
  - `feature/texture`
  - `feature/shape`
  - `model/ml`
  - `docs/report`
- 合并前先确认代码能在本地跑通。

Commit message 建议：

```text
feat: add color features
feat: add texture features
fix: handle empty masks
exp: add xgboost baseline
docs: update task plan
```


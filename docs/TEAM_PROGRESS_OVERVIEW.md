# Team Progress Overview

更新日期：2026-05-21

这份文档给 5 人小组同步当前仓库已经完成的工作、最终主线、探索性亮点、每个人可以接手的讲述方向，以及本地复现方式。

## 1. 当前一句话结论

我们已经完成了一个符合课程规则的皮肤病变三分类项目：

```text
image + mask
-> traditional handcrafted image features
-> strict grouped cross-validation
-> traditional ML classifier
-> robustness / stability / error analysis
-> output.csv
```

官方传统机器学习主模型：

```text
feature_set = all_abcd_grouped
classifier  = LogisticRegression(C=0.3, class_weight="balanced")
k_features  = 140
validation  = StratifiedGroupKFold by original lesion id
mask_mode   = raw
```

Seed `127` grouped-CV 结果：

| 指标 | 分数 |
|---|---:|
| Accuracy | 0.7600 |
| Macro-F1 | 0.7715 |
| Balanced Accuracy | 0.7871 |

五种子稳定性：

| 指标 | 分数 |
|---|---:|
| Mean Accuracy | 0.7283 |
| Mean Macro-F1 | 0.7435 |
| Std Macro-F1 | 0.0271 |
| Mean Balanced Accuracy | 0.7512 |

注意：`0.8055` 是 late probability fusion 的 seed `127` 探索高分，不是稳定主模型。

## 2. 我们满足了哪些课程要求

课程要求：

- 从皮肤病变图像提取特征。
- 使用机器学习方法分类不同 lesion。
- 用增强图像识别证明鲁棒性。
- 定量评估阶段不能依赖深度学习或外部数据。

我们对应完成：

| 要求 | 仓库实现 |
|---|---|
| 图像特征 | color, shape, texture, contrast, ABCD, boundary, mel/nv, LBP, Gabor, subregion |
| 机器学习 | Logistic Regression, SVM, Random Forest, ExtraTrees, HistGB, XGBoost cascade |
| 鲁棒性 | original vs augmented accuracy, group-level consistency |
| 定量合规 | stable main model uses traditional features only |
| 可解释性 | ABCD, boundary, confusion/error visualization |
| 防作弊 | grouped CV keeps original and augmented images in the same fold |

## 3. 最重要的方法修正：防止增强泄漏

数据中一张原始图像有两个增强版本：

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

不能随机按图片划分训练集和验证集，否则会出现：

```text
train: 100.jpg
valid: 100_aug1.jpg
```

这会让模型在验证集里看到同一个 lesion 的增强版本，结果虚高。我们现在使用：

```text
StratifiedGroupKFold
group = base_id
```

因此，同一个 lesion 的原图和增强图一定在同一 fold。之前 90%+ 的结果是 image-level split 乐观估计；现在 70%-80% 区间是更可信的 grouped-CV 结果。

## 4. 已完成模块

### 数据与评估

| 文件 | 作用 |
|---|---|
| `src/dataset.py` | 读取 image / mask / label |
| `src/utils.py` | `base_id_from_image_id` 等工具 |
| `src/evaluate.py` | accuracy, macro-F1, balanced accuracy |
| `scripts/check_dataset.py` | 检查数据完整性 |
| `scripts/check_grouped_cv_outputs.py` | 检查 split/OOF 是否有 group leakage |

### 特征

| 文件 | 作用 |
|---|---|
| `src/features_color.py` | RGB / HSV / Lab 颜色统计 |
| `src/features_shape.py` | 面积、周长、形状比例 |
| `src/features_texture.py` | LBP, GLCM, Sobel 等纹理 |
| `src/features_contrast.py` | lesion-background contrast |
| `src/features_abcd.py` | ABCD-inspired v2 |
| `src/features_abcd_grouped.py` | 当前主模型使用的 ABCD grouped 特征 |
| `src/features_boundary.py` | 边界复杂度、凸包、fractal 等 |
| `src/features_melnv.py` | mel/nv 错误驱动特征 |
| `src/features_lbp_multi.py` | 多尺度 LBP |
| `src/features_gabor.py` | Gabor texture |
| `src/features_subregion.py` | 子区域不对称 |
| `src/features.py` | 统一 feature set 入口 |

### 训练、融合、预测

| 文件 | 作用 |
|---|---|
| `src/train_ml.py` | 传统 ML 主训练入口 |
| `src/train_ml_cascade.py` | XGBoost cascade full-data training |
| `src/train_ml_fusion.py` | 原主模型 + cascade probability fusion |
| `run.py` | 生成最终 `output.csv`，支持 normal/cascade/fusion bundle |

### 实验脚本

| 文件 | 作用 |
|---|---|
| `experiments/run_ml_grid.py` | grid search |
| `experiments/run_stability.py` | multi-seed stability |
| `experiments/run_abcd_grouped_integration.py` | ABCD grouped integration |
| `experiments/run_melnv_refinement.py` | mel/nv refinement |
| `experiments/analyze_robustness.py` | augmented robustness |
| `experiments/visualize_errors.py` | 错误样例图 |
| `experiments/run_xgb_cascade_search.py` | XGBoost cascade |
| `experiments/run_early_fusion_search.py` | early feature fusion |
| `experiments/run_fusion_ensemble.py` | late probability fusion |
| `experiments/train_deep.py` | deep learning extension |

## 5. 实验结果总表

| 方法 | 协议 | Accuracy | Macro-F1 | Balanced Accuracy | 角色 |
|---|---|---:|---:|---:|---|
| Earlier boundary LR | seed 127 grouped OOF | 0.7233 | 0.7454 | 0.7535 | 历史 baseline |
| Main ABCD grouped LR03 | seed 127 grouped OOF | 0.7600 | 0.7715 | 0.7871 | 稳定主模型 |
| Main ABCD grouped LR03 | five-seed mean | 0.7283 | 0.7435 | 0.7512 | 稳定性估计 |
| XGBoost cascade | seed 127 grouped OOF | 0.7617 | 0.7802 | 0.8017 | 探索高分 |
| XGBoost cascade | best five-seed bagged | 0.7500 | 0.7655 | 0.7887 | 探索稳定性 |
| Early fusion | seed 127 grouped OOF | 0.7700 | 0.7830 | 0.7997 | 前端融合 ablation |
| Early fusion | best five-seed mean | 0.7450 | 0.7618 | 0.7766 | 前端融合稳定性 |
| Late probability fusion | seed 127 grouped OOF | 0.7717 | 0.7909 | 0.8055 | 单种子最高 |
| Late probability fusion | five-seed bagged | 0.7417 | 0.7607 | 0.7763 | 后端融合稳定性 |
| MobileNetV2 | grouped validation extension | 0.8750 | 0.8623 | 0.8533 | 深度学习扩展 |

## 6. 最值得汇报的亮点

1. **严格 grouped CV 防止增强泄漏**
   这是最核心的方法论贡献。

2. **完整传统特征体系**
   从基础颜色/纹理/形状，到 ABCD、boundary、melnv、LBP、Gabor、subregion。

3. **主模型可解释**
   ABCD grouped + Logistic Regression 比复杂模型更好讲，也更符合课程规则。

4. **鲁棒性分析**
   原图准确率 `0.7800`，增强图准确率 `0.7500`，group consistency `0.7350`。

5. **错误分析**
   主要错误集中在 `mel` 和 `nv`，有可视化样例支撑。

6. **探索性优化完整**
   Cascade、early fusion、late fusion 都测过，并做了稳定性判断。

7. **负结果也记录**  
   Mask cleaning、mel/nv refinement、部分高维特征没有强行采用。

8. **深度学习 extension**
   MobileNetV2 可作为现代方法对照，但不混入传统主线。

## 7. 五个人怎么分工讲

| 成员 | 方向 | 可讲内容 |
|---|---|---|
| A | 数据与评估协议 | 数据结构、增强泄漏、base_id、grouped CV、指标 |
| B | 基础特征工程 | color / shape / texture / contrast 的设计和作用 |
| C | ABCD 与 boundary | 为什么 ABCD grouped 成为主模型，边界特征的医学直觉 |
| D | 模型搜索与稳定性 | grid search、LR/SVM/XGB 对比、multi-seed stability |
| E | 分析与扩展 | robustness、错误可视化、fusion、deep learning extension |

每个人都要能回答：

```text
我负责的模块是什么？
它为什么有价值？
实验结果是什么？
最终是否被采用？如果没有，为什么？
```

## 8. 本地复现主模型

安装：

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

数据放置：

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

训练主模型：

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw
```

检查 grouped split：

```bash
.venv/bin/python scripts/check_grouped_cv_outputs.py \
  --splits_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv \
  --oof_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
```

生成 `output.csv`：

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

## 9. 推荐阅读顺序

```text
README.md
docs/PROJECT_REVIEW_GUIDE.md
docs/EXPERIMENT_LEDGER.md
docs/REPRODUCIBILITY_CHECKLIST.md
docs/DEEP_LEARNING_EXTENSION_RESULTS.md
```

## 10. 最终建议

现在不建议继续盲目刷模型。更重要的是：

```text
统一最终口径
整理图表
准备 PPT
每个人认领一个实质模块
按 REPRODUCIBILITY_CHECKLIST 做一次提交前验收
```

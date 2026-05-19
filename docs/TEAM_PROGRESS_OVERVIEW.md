# 组内当前工作总览

更新日期：2026-05-19

这份文档给组内 5 位成员同步目前仓库已经完成的工作、核心实验结论、各模块文件位置、最终可汇报亮点，以及下一步还能补的传统机器学习优化方向。

## 1. 一句话总结

我们已经完成了一个符合课程规则的皮肤病变三分类传统机器学习项目主线：

```text
image + mask
-> 传统图像处理特征
-> grouped cross-validation 防止增强泄漏
-> grid search 选择模型
-> robustness / error analysis / stability analysis
-> 生成 output.csv
```

当前最终传统模型：

```text
feature_set = all_boundary
classifier  = LogisticRegression(class_weight="balanced")
feature selection = SelectKBest(k=100)
validation = StratifiedGroupKFold by original lesion id
mask_mode = raw
```

严格 grouped CV 结果：

| 指标 | 分数 |
|---|---:|
| Accuracy | 0.7233 |
| Macro-F1 | 0.7454 |
| Balanced Accuracy | 0.7535 |

5 个 seed 稳定性结果：

| 指标 | 分数 |
|---|---:|
| Mean Macro-F1 | 0.7242 |
| Std Macro-F1 | 0.0143 |
| Mean Balanced Accuracy | 0.7301 |

## 2. 项目规则和我们怎么满足

课程要求定量评估阶段使用传统图像处理方法或简单机器学习方法，不能把深度学习或外部数据作为主提交。

因此我们当前主线是传统方法：

- 颜色特征
- 纹理特征
- 形状特征
- lesion-background contrast 特征
- ABCD-inspired 特征
- boundary 特征
- mel/nv 错误驱动特征
- SVM / Random Forest / Logistic Regression / KNN
- SelectKBest 特征选择
- grouped CV 评估

深度学习已经实现，但只作为 extension，不作为课程定量主结果。

## 3. 最关键的方法修正：增强图像泄漏

数据里有 200 张原始 lesion，另外 400 张是增强图。例如：

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

如果随机按图片划分训练集和验证集，就可能出现：

```text
训练集：100.jpg
验证集：100_aug1.jpg
```

这会导致验证结果虚高，因为模型实际上见过同一个 lesion。

我们已经修正为：

```text
StratifiedGroupKFold
group = base_id，也就是原始 lesion id
```

这样同一个 lesion 的原图和增强图一定在同一个 fold 中，不会跨训练集和验证集泄漏。

### 为什么之前 90% 现在变成 70%？

之前 90%+ 的结果来自 image-level random split，里面存在增强图像泄漏。这个结果不是代码错了，而是评估协议太乐观。

现在 70%+ 是 grouped CV 下的结果，更接近真实泛化能力。这个点是我们报告和答辩里最重要的亮点之一。

## 4. 已完成的代码模块

### 数据与基础工具

| 文件 | 作用 |
|---|---|
| `src/dataset.py` | 读取 image / mask / label |
| `src/utils.py` | 工具函数，包括从增强图提取 `base_id` |
| `src/preprocess.py` | mask 清理、二值化、最大连通域、填洞等 |
| `scripts/check_dataset.py` | 检查数据结构 |

### 特征提取

| 文件 | 作用 |
|---|---|
| `src/features_color.py` | RGB / HSV / Lab 颜色统计 |
| `src/features_texture.py` | gray histogram, LBP, GLCM, Sobel gradient |
| `src/features_shape.py` | mask 面积、周长、bbox、形状统计 |
| `src/features_contrast.py` | 病灶与周围皮肤 ring 的对比特征 |
| `src/features_abcd.py` | ABCD-inspired 特征 |
| `src/features_boundary.py` | 边界复杂度、凸包、Feret diameter、fractal 等 |
| `src/features_melnv.py` | 针对 mel/nv 错误的色素不均匀、不对称和局部变化特征 |
| `src/features.py` | 统一特征入口，支持多个 feature set |

### 训练与预测

| 文件 | 作用 |
|---|---|
| `src/train_ml.py` | 传统 ML 训练、grouped CV、保存模型和指标 |
| `run.py` | 生成最终 `output.csv` |
| `experiments/run_ml_grid.py` | 自动化 grid search |
| `experiments/run_stability.py` | 多 seed 稳定性验证 |
| `experiments/run_melnv_refinement.py` | mel-vs-nv 二阶段 refinement 实验 |

### 分析与可视化

| 文件 | 作用 |
|---|---|
| `experiments/analyze_robustness.py` | 原图/增强图 robustness consistency 分析 |
| `experiments/visualize_errors.py` | 按 true/pred 类型生成错误样例图 |

### 深度学习 extension

| 文件 | 作用 |
|---|---|
| `experiments/train_deep.py` | MobileNetV2 / ResNet18 transfer learning |
| `requirements-deep.txt` | PyTorch / torchvision 依赖 |
| `docs/KAGGLE_DEEP_LEARNING.md` | Kaggle GPU 训练指南 |

## 5. 已完成的实验

### 5.1 自动化 grid search

我们比较了：

- feature sets: `all`, `all_contrast`, `all_abcd_v2`, `all_boundary`, `final`
- classifiers: SVM, Random Forest, Logistic Regression, KNN
- feature selection: all / top 60 / top 100
- validation: grouped CV

最佳结果：

| rank | feature set | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---:|---|---|---:|---:|---:|---:|
| 1 | `all_boundary` | LR | 100 | 0.7233 | 0.7454 | 0.7535 |
| 2 | `final` | SVM | 100 | 0.7150 | 0.7370 | 0.7208 |
| 3 | `final` | SVM | 60 | 0.7100 | 0.7369 | 0.7280 |

结论：

`all_boundary + LR + top100` 是当前最优传统模型。

### 5.2 Mask cleaning 消融

我们尝试了 clean mask：

| feature set | mask | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---|---|---|---:|---:|---:|---:|
| `all` | clean | LR | all | 0.7217 | 0.7373 | 0.7350 |

结论：

clean mask 有价值，但没有超过 raw mask 下的最佳模型，所以最终主模型使用 `mask_mode=raw`。

### 5.3 Robustness consistency

模型在原图和增强图上的结果：

| image type | n images | accuracy |
|---|---:|---:|
| original | 200 | 0.7500 |
| augmented | 400 | 0.7100 |

group-level robustness：

| 指标 | 值 |
|---|---:|
| prediction consistency | 0.6800 |
| all images correct in group | 0.5650 |
| any image correct in group | 0.8750 |
| all match original prediction | 0.6800 |

结论：

模型对增强图有一定鲁棒性，但增强图准确率比原图低。这正好回应了课程要求中关于 augmented images robustness 的点。

### 5.4 错误样例可视化

主要错误集中在 `mel` 和 `nv`：

| true label | predicted label | n errors |
|---|---|---:|
| nv | mel | 84 |
| mel | nv | 51 |
| nv | vasc | 14 |
| vasc | nv | 13 |
| mel | vasc | 3 |
| vasc | mel | 1 |

结论：

`vasc` 相对容易，因为血管性病变颜色对比更明显；`mel` 和 `nv` 都是色素性病变，在颜色、纹理和边界上有重叠，所以最容易互相错分。

### 5.5 Mel/NV 错误驱动优化

我们新增了 `src/features_melnv.py`，包含：

- Lab 颜色离散度
- HSV 色相/饱和度变化
- PCA 主轴两侧不对称
- 四象限颜色差异
- core-border 对比
- 3x3 局部色块变化
- dark / high-saturation 连通区域统计

结果：

| feature set | classifier | k | macro-F1 |
|---|---|---:|---:|
| `final_melnv` | SVM | 80 | 0.7314 |
| `all_boundary_melnv` | LR | 100 | 0.7240 |
| 当前最终 `all_boundary` | LR | 100 | 0.7454 |

结论：

mel/nv 特征有医学解释性，但没有超过当前最终模型。这个是一个有价值的负结果，说明盲目加特征可能因为维度增加和信号重复而降低泛化。

### 5.6 二阶段 mel-vs-nv refinement

思路：

1. 主模型先做三分类。
2. 如果预测为 `mel` 或 `nv`，再用一个二分类模型重新判断。
3. `vasc` 预测不变。

最佳结果：

| refinement | main mean macro-F1 | refined mean macro-F1 | delta |
|---|---:|---:|---:|
| `all_boundary` LR top80 | 0.7242 | 0.7241 | -0.0001 |

结论：

二阶段模型看起来更复杂，但没有稳定提升，所以不采用。这也可以作为报告里的负结果分析。

### 5.7 Multi-seed 稳定性验证

使用 seeds：

```text
42, 127, 2024, 3407, 520
```

最佳稳定性：

| model | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|
| `all_boundary` + LR + top100 | 0.7242 | 0.0143 | 0.7301 |

结论：

seed 127 的 0.7454 是一个较好的 split，但不是孤立偶然。更保守地说，模型的真实 grouped CV 期望大约在 0.72 到 0.73 macro-F1。

### 5.8 深度学习 extension

本地 pretrained MobileNetV2 只跑 1 epoch：

| metric | score |
|---|---:|
| Accuracy | 0.6250 |
| Macro-F1 | 0.6093 |

Kaggle GPU 训练说明见：

```text
docs/KAGGLE_DEEP_LEARNING.md
```

注意：

深度学习只作为 extension，不作为课程定量主提交。

## 6. 当前最值得汇报的亮点

1. **修正 augmentation leakage**  
   这是方法论亮点，也是为什么我们从 90% 变成 70% 的关键解释。

2. **严格 grouped CV**  
   不让同一个 lesion 的原图和增强图跨训练/验证集。

3. **传统特征体系完整**  
   颜色、纹理、形状、contrast、ABCD、boundary、mel/nv targeted features 都做过。

4. **自动化 grid search**  
   不只是手动试模型，而是系统比较 feature set、classifier、feature selection。

5. **Robustness consistency**  
   单独分析原图和增强图表现，回应课程要求。

6. **错误可视化和 failure analysis**  
   明确发现 mel/nv 是主要难点。

7. **负结果也记录**  
   mask cleaning、mel/nv features、二阶段 refinement、深度学习 smoke test 都有结论，没有为了复杂而复杂。

8. **Multi-seed stability**  
   证明结果不是只靠一个随机 split。

## 7. 五个人可以怎么分工讲

这不是强制分工，只是建议大家汇报时每个人都有清楚的 substantive contribution。

| 成员 | 可负责方向 | 可讲内容 |
|---|---|---|
| A | 数据协议与评估 | 数据结构、增强泄漏、`base_id`、grouped CV、指标选择 |
| B | 基础特征 | color / texture / shape，尤其 texture 优化和 HOG 不适合小数据的结论 |
| C | 高级传统特征 | contrast、ABCD、boundary，解释为什么 boundary 最有效 |
| D | 模型选择与实验系统 | grid search、classifier 对比、SelectKBest、multi-seed stability |
| E | 分析与扩展 | robustness、错误可视化、mel/nv failure analysis、deep learning extension |

每个人都应该能回答：

- 自己负责了哪个代码/实验模块？
- 为什么这个模块对项目有价值？
- 实验结果是什么？
- 这个模块最终有没有被采用？如果没有，为什么？

## 8. 组员本地怎么复现

### 安装

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

### 数据位置

不要上传数据到 GitHub。本地放：

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

### 检查数据

```bash
.venv/bin/python scripts/check_dataset.py --data_dir data/Data_Proj2
```

### 训练当前最终传统模型

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_boundary \
  --classifier lr \
  --k_features 100 \
  --cv grouped \
  --mask_mode raw
```

### 生成 output.csv

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_boundary_lr_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

## 9. 接下来传统机器学习还可以做什么

如果还要继续优化传统 ML，优先级如下：

### 9.1 数值稳定性优化

当前 LR 有 overflow warning，可以尝试：

- `RobustScaler`
- feature clipping，例如 1%-99% 分位裁剪
- Logistic Regression 不同 solver：`lbfgs`, `liblinear`, `saga`
- 更细的 `C` 搜索：`0.03, 0.1, 0.3, 1, 3`

这部分很适合写成工程稳定性优化。

### 9.2 Ensemble

尝试简单 ensemble，而不是复杂 stacking：

- `all_boundary + LR`
- `final + SVM`
- `all_contrast + SVM`
- `all + LR`
- `final_melnv + SVM`

比较 hard voting / soft voting。如果不涨，也可以作为负结果记录。

### 9.3 Threshold tuning

针对类别不平衡和 mel/nv 错误，可以调概率阈值：

- 默认 argmax
- 以 macro-F1 为目标搜索 class threshold
- 观察 mel recall 和 nv precision 的 tradeoff

### 9.4 Feature importance

输出并解释：

- SelectKBest top features
- Logistic Regression coefficients
- Random Forest feature importance
- 按组汇总 feature importance

这个很适合 PPT 展示，让老师看到我们不仅追分，也解释模型。

## 10. 不建议现在主攻的方向

- 继续大量加高维 HOG/SIFT：样本太小，容易稀释信号。
- 更复杂的 stacking：可能过拟合，不好解释。
- 把深度学习当主提交：课程定量评估不允许。
- 继续押 mask cleaning：目前已经验证没有超过 raw mask。
- 继续押二阶段 mel-vs-nv：已经验证不稳定。

## 11. 最终建议

目前工作量对 5 人大作业已经够，而且亮点比较完整。接下来最值得补的是：

```text
RobustScaler / clipping
simple ensemble
threshold tuning
feature importance
```

这些都是传统机器学习方向，既符合课程规则，也更容易在报告中展示为系统性实验。

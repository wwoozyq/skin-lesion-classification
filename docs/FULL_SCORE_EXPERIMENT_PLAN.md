# Skin Lesion Classification 满分实验规划

## 1. 项目目标与评分约束

本项目目标是对皮肤病变图像进行三分类：

- `mel`: melanoma，黑色素瘤
- `nv`: melanocytic nevus，黑色素细胞痣
- `vasc`: vascular lesion，血管性病变

输入包括 `image/`、`mask/` 和训练阶段的 `label.csv`。最终提交文件为：

```csv
image_id,dx
1,nv
2,mel
3,vasc
```

严格按课程要求，本项目的定量评估阶段必须使用传统图像处理方法或简单机器学习方法。深度学习模型、外部数据和大模型方法不能作为定量评估主提交。它们可以作为展示和报告中的扩展探索，但必须与主线结果清楚分开。

满分目标不是只追求一个最高 accuracy，而是要证明：

1. 方法符合课程规则，TA 给的 demo test files 能直接运行。
2. 特征设计有医学图像解释性，不是盲目堆特征。
3. 验证方式能避免增强图像带来的数据泄漏。
4. 指标完整，特别关注类别不平衡下的 macro-F1 和 balanced accuracy。
5. 有消融实验、错误分析和可复现实验记录。

## 2. 现有工作基础

当前主分支已经完成了一个合格的传统机器学习框架：

- 数据读取：`src/dataset.py`
- 特征入口：`src/features.py`
- 颜色特征：`src/features_color.py`
- 纹理特征：`src/features_texture.py`
- 形状特征：`src/features_shape.py`
- 训练与交叉验证：`src/train_ml.py`
- 指标计算：`src/evaluate.py`
- 隐藏测试预测：`run.py`
- 数据检查脚本：`scripts/check_dataset.py`

当前主线特征包括：

- RGB / HSV / Lab 病灶区域统计特征
- 灰度直方图
- LBP
- GLCM
- Sobel 梯度统计
- mask 面积、周长、外接框比例、extent、中心位置等形状特征

当前主线分类器为：

- `SVM RBF(C=100, class_weight="balanced")`
- 备选 `RandomForest`
- 备选 `LogisticRegression`

已有实验记录显示，纹理优化后主线传统 ML baseline 大约达到：

| 方法 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 旧 baseline | 0.892 | 0.898 | 0.892 |
| 纹理优化 + SVM RBF | 0.903 | 0.911 | 0.907 |

远端 feature 分支中还有可继续合并验证的工作：

- `feature/contrast-example`: 病灶-背景对比特征，记录中 `all_contrast` macro-F1 为 0.9125。
- `feature/abcd-clean`: 增强 ABCD 特征，记录中 `all_abcd_v2` macro-F1 为 0.9128。
- `feature/boundary-integration`: boundary + ABCD 集成，记录中 `all_abcd_v2_boundary` macro-F1 为 0.9191。
- `feature/external-data-research`: ISIC/HAM10000 外部数据调研和伪 mask / crop 脚本，适合作为报告扩展，不作为定量主提交。

这些结果已经说明项目方向是正确的。下一步的重点是：统一评估协议、避免增强泄漏、合并有效特征、补齐实验矩阵和报告证据链。

## 3. 数据协议：最关键的满分点

课程说明中强调测试数据也来自原始数据的增强版本，因此实验中必须避免增强图像泄漏。

如果文件名类似：

```text
23.jpg
23_aug1.jpg
23_aug2.jpg
```

那么 `23`、`23_aug1`、`23_aug2` 必须被视为同一个原始病例组。验证集划分时不能让原图进训练集、增强图进验证集，否则验证结果会虚高。

必须新增或修改验证逻辑：

1. 从 `image_id` 提取 `base_id`。
2. 使用 `StratifiedGroupKFold` 或等价的 group split。
3. 分组依据为 `base_id`，分层标签为 `dx`。
4. 所有增强版本只能同时出现在 train 或 validation 中。

推荐命名规则：

```text
base_id = image_id 去掉 _aug1、_aug2、_rotate、_flip 等增强后缀
```

报告中必须明确说明：

> Because augmented images may be derived from the same original lesion, we used group-aware cross-validation by original image id to avoid data leakage between training and validation.

这是严格老师最可能扣分的地方，也是最容易拉开差距的地方。

## 4. 主线方法设计

### 4.1 预处理

预处理必须服务于传统特征提取，不做复杂黑箱处理：

1. 读取 RGB 图像和 mask。
2. mask 二值化。
3. 若 mask 有噪声，保留最大连通区域，填洞，轻微形态学开闭运算。
4. 若 mask 为空，退化为全图统计，并在日志中记录。
5. 基于 mask 获取：
   - lesion pixels
   - background ring pixels
   - lesion bounding box
   - lesion-centered crop

background ring 不应使用整张非病灶区域，而应使用 mask 膨胀后的一圈周围皮肤。这样更符合皮肤病变与周围正常皮肤对比的医学逻辑。

### 4.2 特征体系

最终主线特征建议由六组构成。

第一组：颜色特征。

- RGB / HSV / Lab 通道均值、标准差、p10、p50、p90
- HSV 色相和饱和度直方图
- Lab 空间颜色离散度
- 病灶内部颜色熵

第二组：纹理特征。

- 灰度直方图
- LBP uniform histogram
- GLCM contrast / dissimilarity / homogeneity / energy / correlation
- Sobel 梯度均值、标准差、skewness、kurtosis

已有实验说明高维 HOG 在 600 张左右数据上容易稀释信号，因此 HOG 不作为主线必选特征；如果使用，只能作为消融项，并控制维度。

第三组：形状特征。

- area ratio
- perimeter
- circularity
- bbox aspect ratio
- extent
- eccentricity
- solidity
- major/minor axis ratio
- boundary irregularity

第四组：病灶-背景对比特征。

- lesion 与 background ring 的 RGB / HSV / Lab 均值差
- lesion 与 background ring 的颜色标准差差异
- 灰度均值差、灰度熵差
- 梯度均值差、梯度标准差差

这组特征对 `vasc` 很有价值，因为血管性病变往往与周围皮肤颜色差异更明显。

第五组：ABCD 规则特征。

ABCD 特征要写成“医学规则到计算特征”的映射：

- A: PCA 主轴对齐后的面积不对称、颜色不对称
- B: radial border irregularity、border roughness、fractal-like complexity
- C: 诊断颜色类别比例，例如 black、white、red、blue-gray、dark brown、light brown
- D: lesion scale、major/minor ratio、diameter ratio

不要只说“我们加了 ABCD 特征”。报告中要说明第一版 ABCD 与已有特征重叠，增强版通过主轴对齐、诊断颜色和径向边界建模带来提升。

第六组：boundary 特征。

如果合并 `feature/boundary-integration`，需要把边界特征单独列为一组消融：

- 边界半径波动
- 边界梯度统计
- 边界颜色过渡
- 边界粗糙度
- 局部边界复杂度

这组特征要强调它补充的是 lesion 内部统计无法表达的边缘信息。

## 5. 模型路线

主提交模型建议保持传统机器学习路线：

1. `StandardScaler`
2. 可选 `SelectKBest(f_classif)` 或 PCA
3. 分类器
4. 保存完整 pipeline

必须比较的分类器：

- Logistic Regression: 线性 baseline，可解释但表达能力有限。
- KNN: 简单非参数 baseline。
- Random Forest: 对非线性和混合特征友好，可给 feature importance。
- SVM RBF: 当前主线最强，适合中小规模人工特征。

可选比较：

- XGBoost / LightGBM: 若环境允许可以加，但不要让项目依赖难安装包。
- Soft voting: 只有在 grouped CV 中稳定提升时才纳入最终提交。
- Stacking: 报告可写尝试，但样本量小，过拟合风险高，不建议作为默认主提交。

最终模型选择标准：

1. grouped CV macro-F1 最高。
2. balanced accuracy 不明显下降。
3. mel 类 recall 不能过低。
4. 在不同随机种子下稳定。
5. `run.py` 能无标签预测并生成正确格式 CSV。

## 6. 评估设计

主指标：

- Macro-F1

辅助指标：

- Accuracy
- Balanced Accuracy
- Per-class Precision / Recall / F1
- Confusion Matrix

推荐验证设置：

1. 使用 grouped 5-fold cross validation。
2. 每个实验至少固定 3 个随机种子，报告 mean ± std。
3. 所有模型使用完全相同的 split。
4. 保存每次实验的 metrics、classification report、confusion matrix。

最低实验矩阵：

| 实验 | 目的 |
|---|---|
| color only | 验证颜色是否为主要信号 |
| texture only | 验证纹理贡献 |
| shape only | 验证 mask 形状贡献 |
| contrast only | 验证病灶-背景差异贡献 |
| abcd_v2 only | 验证医学规则特征是否有独立信息 |
| boundary only | 验证边界信息贡献 |
| all baseline | 当前主线对照 |
| all + contrast | 验证 contrast 是否互补 |
| all + abcd_v2 | 验证 ABCD 是否互补 |
| all + boundary | 验证边界是否互补 |
| all + contrast + abcd_v2 + boundary | 最终候选 |

模型矩阵：

| 特征集 | LR | KNN | RF | SVM RBF |
|---|---:|---:|---:|---:|
| color | 必做 | 可选 | 必做 | 必做 |
| texture | 必做 | 可选 | 必做 | 必做 |
| shape | 必做 | 可选 | 必做 | 必做 |
| all | 必做 | 可选 | 必做 | 必做 |
| final all | 必做 | 可选 | 必做 | 必做 |

最终报告中不要堆太多表。正文放关键表，附录放完整实验 CSV。

## 7. 增强鲁棒性验证

因为课程要求算法要能识别增强图像，所以必须单独设计 robustness experiment。

建议做两类验证：

第一类：增强一致性。

对于同一个 `base_id` 的原图和增强图，统计模型预测是否一致：

```text
consistency = 同组内预测为同一类别的比例
```

第二类：按增强类型分组评估。

如果文件名能识别增强类型，例如 flip、rotate、brightness，则分别统计：

- original accuracy / macro-F1
- flip accuracy / macro-F1
- rotate accuracy / macro-F1
- color-jitter accuracy / macro-F1

如果文件名不能识别增强类型，也至少统计 original vs augmented 两组指标。

报告中应回答：

1. 哪类增强最容易误分类？
2. 颜色特征是否受 brightness/color augmentation 影响？
3. 形状和 ABCD 特征是否对 rotation 更稳定？
4. 最终模型是否比单一颜色/纹理模型更稳？

## 8. 错误分析

必须做错误分析，否则报告会像只跑了脚本。

建议输出：

1. 混淆矩阵热力图。
2. 每个类别最常见的错误方向，例如 `mel -> nv`。
3. 每类挑 3 张错分图，展示原图、mask、预测、真实标签。
4. 分析错分原因：
   - mel 和 nv 颜色/形状相近。
   - vasc 样本少，但颜色对比明显。
   - mask 不准会影响 shape、boundary、ABCD。
   - 增强导致颜色统计漂移。

严格老师喜欢看到“失败原因”和“下一步改进”，不要只展示成功样例。

## 9. 代码实现计划

第一阶段：统一评估协议。

- 新增 `base_id` 解析函数。
- 将 `StratifiedKFold` 改成 `StratifiedGroupKFold`。
- 固定 split 保存到 `outputs/metrics/splits_seed*.csv`。
- 每个实验保存完整 metrics CSV。

第二阶段：合并有效特征。

优先合并顺序：

1. `feature/contrast-example`
2. `feature/abcd-clean` 中的 `abcd_v2`
3. `feature/boundary-integration`

合并后不要直接相信旧分支结果，必须在统一 grouped CV 下重新跑。

第三阶段：实验 runner。

建议新增：

```text
experiments/run_ml_grid.py
```

支持参数：

```bash
python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets color texture shape all all_contrast all_abcd_v2 final \
  --classifiers lr rf svm \
  --cv grouped \
  --seeds 42 2024 2026 \
  --output_dir outputs/metrics
```

第四阶段：最终训练与预测。

最终确定模型后：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set final --classifier svm
python run.py --input_dir path/to/demo_test --output_csv output.csv
```

TA 测试目录可能没有 `label.csv`，所以 `run.py` 必须只依赖 `image/` 和 `mask/`。

## 10. 报告与展示结构

Presentation 建议结构：

1. Problem and requirement
2. Dataset and leakage-safe validation
3. Overall pipeline
4. Feature design: color, texture, shape, contrast, ABCD, boundary
5. Model comparison
6. Ablation study
7. Robustness on augmented images
8. Error analysis
9. Final result and contribution table

Report 建议结构：

1. Introduction
2. Dataset and evaluation protocol
3. Method
4. Experiments
5. Results
6. Discussion
7. Individual contributions
8. Appendix: complete metrics and commands

个人贡献必须具体到实质工作，例如：

- 数据与防泄漏评估协议
- 颜色/对比特征
- 纹理特征优化
- ABCD 医学解释特征
- boundary 特征
- 模型调参与实验记录
- 报告和可视化

不要把“做 PPT”当作主要贡献。

## 11. 外部数据与深度学习扩展

外部数据和深度学习可以作为加分展示，但不能污染主线定量评估。

推荐写法：

> The official quantitative submission follows the course constraint and uses only traditional image processing and simple machine learning. Deep learning and external ISIC/HAM10000 data are treated as additional exploratory experiments for discussion, not as the main quantitative submission.

外部数据建议只做：

- ISIC 2018 / HAM10000 标签映射调研
- 不同数据源 domain shift 分析
- lesion-centered crop 的探索
- 深度学习预训练的后续想法

不要把外部数据混入课程验证集后再报告主指标。

## 12. 时间安排

Day 1:

- 跑通当前主分支。
- 检查本地数据格式。
- 实现 grouped split。
- 复现当前 `all + SVM` baseline。

Day 2:

- 合并 contrast 特征。
- 合并 abcd_v2 特征。
- 在 grouped CV 下重跑消融。

Day 3:

- 合并 boundary 特征。
- 完成模型矩阵：LR / RF / SVM。
- 初步确定 final candidate。

Day 4:

- 做增强鲁棒性实验。
- 做错误分析和错分图可视化。
- 固化最终模型和预测脚本。

Day 5:

- 从干净环境复现安装、训练、预测。
- 检查 `output.csv` 格式。
- 整理所有结果 CSV 和图。

Day 6:

- 写 presentation 主线。
- 写 report 方法与实验部分。
- 明确每位成员实质贡献。

Day 7:

- 模拟答辩。
- 准备老师可能问的问题：
  - 为什么不用 deep learning 做主提交？
  - 为什么 macro-F1 比 accuracy 更重要？
  - 如何避免增强图像泄漏？
  - 哪组特征贡献最大？
  - mel 和 nv 为什么容易混淆？
  - 外部数据为什么没有混进主评估？

## 13. 严格扣分风险清单

以下问题必须避免：

1. 随机划分导致同一原图的增强版本同时出现在 train 和 validation。
2. 只报告 accuracy，不报告 macro-F1、balanced accuracy 和 per-class F1。
3. 定量主提交使用深度学习或外部数据。
4. `run.py` 依赖 `label.csv`，导致 TA hidden test 不能运行。
5. 只给最终结果，没有消融实验。
6. 只说“用了很多特征”，没有解释医学含义。
7. 旧分支结果和主分支结果评估协议不同，却直接横向比较。
8. 数据、模型、脚本路径写死在个人电脑上。
9. 上传课程数据或外部数据图片到 GitHub。
10. 个人贡献写得含糊，无法证明每个人做了实质工作。

## 14. 最终满分版本应达到的状态

代码层面：

- 一条命令能检查数据。
- 一条命令能训练并保存模型。
- 一条命令能对 TA 测试集生成 `output.csv`。
- 所有实验结果可复现。

实验层面：

- 有 grouped CV。
- 有消融实验。
- 有模型对比。
- 有增强鲁棒性验证。
- 有错误分析。

报告层面：

- 方法符合课程规则。
- 特征设计有医学解释。
- 指标选择合理。
- 失败分析诚实。
- 分工清楚。

推荐最终主线表述：

> We built a traditional image-processing and machine-learning pipeline using lesion masks. The model extracts color, texture, shape, lesion-background contrast, ABCD-inspired, and boundary features, then trains a balanced SVM/RF classifier. To avoid leakage from augmented images, we evaluate using group-aware cross-validation by original lesion id. The final model is selected by macro-F1 and balanced accuracy, and its robustness is further verified on augmented images.


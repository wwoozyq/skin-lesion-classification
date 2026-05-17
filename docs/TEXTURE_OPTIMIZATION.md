# 纹理特征优化记录

## 初始 Baseline

运行原始代码（旧纹理 + Random Forest）：

| 配置 | Macro-F1 | Accuracy | Balanced Acc |
|------|----------|----------|-------------|
| 旧纹理（单独） | 0.777 | 0.788 | 0.754 |
| 全部特征 | **0.898** | **0.892** | **0.892** |

旧纹理仅包含灰度直方图16维 + 灰度均值/标准差/分位数 + 梯度均值/标准差，共20维。

## 探索过程

### 第一轮：新增 LBP + GLCM + HOG + 梯度 + 边缘密度

根据调研报告，在 `features_texture.py` 中添加了五个方向的特征。初始实现 HOG 用了 64x64 分辨率，产生约 1764 维特征，严重维度爆炸，导致模型偏向多数类：

| 配置 | Macro-F1 | 问题 |
|------|----------|------|
| HOG 64x64 | 0.515 | 维度爆炸，RF 完全偏向 nv 类 |
| HOG 48x48 | 0.637 | 仍有 144 维 HOG 特征，稀释信号 |

### 第二轮：去掉 HOG，保留 LBP + GLCM + 梯度 + 边缘密度

去掉 HOG 后纹理单独 0.753（略低于旧版 0.777），全部特征 0.886（低于旧版 0.898）。新增的纹理特征维度偏高但信号不够强。

### 第三轮：Random Forest 参数搜索

测试了 12 种 RF 参数组合（n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf）：

| 排名 | 参数 | Macro-F1 |
|------|------|----------|
| 1 | depth=15, max_features=sqrt, n_est=300 | 0.890 |
| 2 | depth=None, n_est=500 | 0.888 |
| 3 | depth=None, n_est=300（默认） | 0.886 |

最优参数 max_depth=15 带来 +0.4% 提升，但仍未超过旧 baseline。

### 第四轮：SelectKBest 特征筛选搜索

使用 f_classif 在不同 k 值（10~99）和多种 RF 配置下搜索，共 108 种组合：

| 排名 | k | RF 配置 | Macro-F1 |
|------|---|---------|----------|
| 1 | 99（全保留） | depth=15 | 0.890 |
| 2 | 15 | depth=20 | 0.882 |
| 3 | 25 | depth=15 | 0.882 |

结论：RF 的 `max_features=sqrt` 已经做了隐式特征选择，显式 SelectKBest 无法进一步改善。

### 第五轮：ANOVA F-score 特征重要性分析

对 99 维特征按组计算 F-score：

| 特征组 | 数量 | 平均 F | 最大 F | 结论 |
|--------|------|--------|--------|------|
| color | 45 | 31.3 | 183.6 | 强信号（HSV/Lab） |
| glcm | 10 | 59.2 | 93.0 | 最强纹理信号 |
| lbp | 28 | 23.6 | 49.5 | 中等偏弱 |
| shape | 6 | 22.0 | 70.5 | 强信号（perimeter） |
| gradient | 7 | 11.2 | 52.9 | 仅 entropy 有用 |
| edge_density | 3 | 8.8 | 10.9 | 噪声 |

LBP 28 维和 edge_density 3 维平均 F-score 很低，在稀释强信号。

### 第六轮：精简纹理特征

去掉 LBP 多尺度和 edge_density，只保留 LBP(R=2, 18维) + GLCM(10维) + 梯度(3维) + 旧 gray histogram(20维)，共 51 维纹理特征：

| 配置 | Macro-F1 |
|------|----------|
| 全保留 | 0.891 |

接近旧版但仍差 0.007。

### 第七轮：多分类器对比

| 分类器 | Macro-F1 | Accuracy | Balanced Acc |
|--------|----------|----------|-------------|
| **SVM RBF(C=100)** | **0.911** | **0.903** | **0.907** |
| SVM RBF(C=10) | 0.905 | 0.895 | 0.904 |
| RF depth=15 | 0.891 | 0.880 | 0.880 |
| RF 无 depth 限制 | 0.889 | 0.878 | 0.875 |
| SVM RBF(C=1) | 0.829 | 0.803 | 0.835 |
| SVM linear | 0.791 | 0.768 | 0.791 |
| LogisticRegression | 0.789 | 0.770 | 0.802 |

**SVM RBF(C=100) 显著超越旧 baseline**，且 mel 类 F1 从 0.86 提升到 0.89。

### 结论

1. HOG 对 600 样本×高维场景不适用，会稀释信号
2. RF 加特征筛选无法突破瓶颈，因为 max_features=sqrt 已做隐式筛选
3. SVM RBF(C=100) 更适合该任务的特征分布
4. GLCM 是最强的纹理信号，edge_density 和 LBP 多尺度贡献有限

## 最终改动

### `src/features_texture.py`

| 特征 | 维度 | 说明 |
|------|------|------|
| gray_histogram | 20 | 16维直方图 + mean/std/p10/p90（保留旧 baseline） |
| LBP | 18 | R=2, P=16, uniform 模式 |
| GLCM | 10 | contrast/dissimilarity/homogeneity/energy/correlation × 2距离 |
| gradient | 6 | mean/std/skew/kurtosis/entropy（用 Sobel 算子） |

总计纹理特征约 54 维，全部特征约 103 维。

### `src/train_ml.py`

- 默认分类器改为 **SVM RBF(C=100)**
- 支持 `--classifier rf` / `--classifier lr` 切换
- 支持 `--k_features N` 控制 SelectKBest 保留维度
- 新增 `sklearn.svm.SVC` 和 `sklearn.linear_model.LogisticRegression` 导入

### `requirements.txt`

- 新增 `scikit-image`

## 最终效果

| 指标 | 旧 baseline | 优化后 |
|------|-------------|--------|
| Macro-F1 | 0.898 | **0.911** |
| Accuracy | 0.892 | **0.903** |
| Balanced Acc | 0.892 | **0.907** |
| mel F1 | 0.86 | **0.89** |
| nv F1 | 0.90 | **0.90** |
| vasc F1 | 0.93 | **0.94** |

注意！新增纹理种类后原分类器不太适用，可能需要调试分类器的同学进一步加工一下

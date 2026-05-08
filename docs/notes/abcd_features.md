# ABCD-rule inspired features

## 做了什么

本次优化参考皮肤镜诊断中的 ABCD 规则，把医学上的不对称性、边界不规则、颜色多样性和病灶尺度/结构复杂度转成可计算的人工特征。

ABCD 含义：

- A: Asymmetry，不对称性
- B: Border，边界不规则性
- C: Color，颜色变化
- D: Differential structures / Diameter，差异结构或病灶尺度

本方向的目的不只是追求分数，而是增强传统机器学习模块的医学解释性。

第一版 ABCD 特征主要是直接从 mask、颜色统计和梯度里取数。跑完以后发现：
单独效果不够强，加入 baseline 后也几乎不提升。原因比较明确：这些特征和原来已有的
shape、color、texture 特征重叠太多。

所以又做了一个增强版 `abcd_v2`，重点把 ABCD 规则做得更像“诊断规则”，而不是简单堆统计量。

## 参考资料

- Nachbar et al., The ABCD rule of dermatoscopy.  
  https://pubmed.ncbi.nlm.nih.gov/8157780/
- Dermoscopedia: ABCD rule.  
  https://dermoscopedia.org/ABCD_rule
- American Academy of Dermatology: ABCDEs of melanoma.  
  https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes

## 新增代码

新增文件：

```text
src/features_abcd.py
```

新增核心函数：

```python
def extract_abcd_features(image, mask):
    ...
    return features

def extract_abcd_v2_features(image, mask):
    ...
    return features
```

接入文件：

```text
src/features.py
src/train_ml.py
```

## 新增特征

### A: Asymmetry

- `abcd_asymmetry_lr`: mask 左右翻转不对称性
- `abcd_asymmetry_ud`: mask 上下翻转不对称性
- `abcd_color_asymmetry_lr`: 病灶左右两半颜色均值差异
- `abcd_color_asymmetry_ud`: 病灶上下两半颜色均值差异

### B: Border

- `abcd_border_irregularity`: 周长平方与面积的比值
- `abcd_circularity`: 圆度
- `abcd_extent`: 病灶面积与外接矩形面积比
- `abcd_perimeter_area_ratio`: 周长面积比

### C: Color

- `abcd_rgb_std_mean`: RGB 通道平均标准差
- `abcd_hsv_std_mean`: HSV 通道平均标准差
- `abcd_lab_std_mean`: Lab 通道平均标准差
- `abcd_gray_entropy`: 灰度熵
- `abcd_color_range_mean`: 颜色分位数范围
- `abcd_color_bin_count`: 粗量化颜色种类数

### D: Diameter / Structures

- `abcd_area_ratio`: 病灶面积占整图比例
- `abcd_bbox_diameter_ratio`: 外接框对角线近似直径
- `abcd_major_minor_ratio`: 长短轴比例
- `abcd_gradient_mean`: 病灶区域梯度均值
- `abcd_gradient_std`: 病灶区域梯度标准差

## v2 增强点

### A: PCA-aligned asymmetry

第一版直接按图像上下、左右切分。问题是病灶可能旋转，左右切不一定对应真实主轴。
v2 先用 mask 像素做 PCA，找到病灶主轴和短轴，再分别计算面积不对称和颜色不对称。

新增特征：

- `abcd_v2_pca_area_asymmetry_major`
- `abcd_v2_pca_area_asymmetry_minor`
- `abcd_v2_pca_color_asymmetry_major`
- `abcd_v2_pca_color_asymmetry_minor`

### B: Radial border irregularity

边界不规则不只看周长面积比。v2 从病灶中心到边界点计算半径序列，
再统计半径波动、半径范围、边界粗糙度和盒计数近似分形维数。

新增特征：

- `abcd_v2_border_radial_cv`
- `abcd_v2_border_radial_range`
- `abcd_v2_border_roughness`
- `abcd_v2_border_fractal_dim`

### C: Diagnostic color categories

ABCD 规则里颜色不是普通 RGB 方差，而是黑、白、红、蓝灰、深棕、浅棕等诊断颜色。
v2 用简单阈值统计这些颜色比例、颜色类别数、颜色熵和主导颜色比例。

新增特征：

- `abcd_v2_diag_color_count`
- `abcd_v2_black_ratio`
- `abcd_v2_white_ratio`
- `abcd_v2_red_ratio`
- `abcd_v2_blue_gray_ratio`
- `abcd_v2_dark_brown_ratio`
- `abcd_v2_light_brown_ratio`
- `abcd_v2_diag_color_entropy`
- `abcd_v2_dominant_color_ratio`

## 怎么运行

只测试 ABCD 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set abcd
```

测试 baseline 特征 + ABCD 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_abcd
```

测试增强版 ABCD：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set abcd_v2
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_abcd_v2
```

## 实验结果

使用同一个 RandomForest baseline 和 Stratified 5-Fold 评估。

| method | accuracy | macro_f1 | balanced_accuracy | mel_f1 | nv_f1 | vasc_f1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_all | 0.8917 | 0.8980 | 0.8908 | 0.86 | 0.90 | 0.93 |
| abcd_only | 0.8033 | 0.7898 | 0.7631 | 0.80 | 0.82 | 0.75 |
| all_abcd | 0.8917 | 0.8980 | 0.8883 | 0.86 | 0.90 | 0.93 |
| abcd_v2 | 0.8733 | 0.8804 | 0.8696 | 0.84 | 0.88 | 0.92 |
| all_abcd_v2 | 0.9100 | 0.9128 | 0.9064 | 0.89 | 0.92 | 0.93 |

## 结果分析

ABCD v1 单独使用时，效果低于 baseline。这说明简单的 mask 形态、颜色复杂度和不对称性特征虽然有医学解释性，但信息量不足以单独完成三分类。

将 ABCD 特征加入原 baseline 后，Accuracy 基本不变，Macro-F1 与 baseline 基本持平，Balanced Accuracy 略低。这说明当前实现的 ABCD 特征与已有颜色、形状、纹理特征存在较多重叠，暂时没有明显带来额外性能提升。

增强版 ABCD v2 后，单独使用的 macro-F1 从 0.7898 提升到 0.8804，说明主轴不对称、边界半径波动和诊断颜色类别确实比第一版更有效。
加入 baseline 后，macro-F1 从 0.8980 提升到 0.9128，balanced accuracy 从 0.8908 提升到 0.9064。
这说明 v2 特征提供了一部分 baseline 原有特征没有覆盖的信息。

这个方向的报告价值比较完整：

1. 它提供了明确的医学规则来源。
2. 它能解释传统特征为什么这样设计。
3. 它可以作为消融实验中的可解释特征组。
4. 它有从 v1 到 v2 的失败分析和改进过程。
5. 它在当前验证设置下带来了实际分数提升。

## 是否建议合并

建议合并增强版 `all_abcd_v2`。

报告里不要只写“我们用了 ABCD 特征”，而应该写成：

1. 先做文献规则映射。
2. 第一版直接实现，发现和已有特征重叠，提升有限。
3. 分析原因后做 v2：主轴对齐、边界半径建模、诊断颜色类别。
4. 用同一套 5-fold 设置比较，证明 v2 比 v1 有效，并且加到 baseline 后提升 macro-F1。

# ABCD-rule inspired features

## 做了什么

本次优化参考皮肤镜诊断中的 ABCD 规则，把医学上的不对称性、边界不规则、颜色多样性和病灶尺度/结构复杂度转成可计算的人工特征。

ABCD 含义：

- A: Asymmetry，不对称性
- B: Border，边界不规则性
- C: Color，颜色变化
- D: Differential structures / Diameter，差异结构或病灶尺度

本方向的目的不只是追求分数，而是增强传统机器学习模块的医学解释性。

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

## 怎么运行

只测试 ABCD 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set abcd
```

测试 baseline 特征 + ABCD 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_abcd
```

## 实验结果

使用同一个 RandomForest baseline 和 Stratified 5-Fold 评估。

| method | accuracy | macro_f1 | balanced_accuracy | mel_f1 | nv_f1 | vasc_f1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_all | 0.8917 | 0.8980 | 0.8908 | 0.86 | 0.90 | 0.93 |
| abcd_only | 0.8033 | 0.7898 | 0.7631 | 0.80 | 0.82 | 0.75 |
| all_abcd | 0.8917 | 0.8980 | 0.8883 | 0.86 | 0.90 | 0.93 |

## 结果分析

ABCD 特征单独使用时，效果低于 baseline。这说明基于 mask 的形态、颜色复杂度和不对称性特征虽然有医学解释性，但信息量不足以单独完成三分类。

将 ABCD 特征加入原 baseline 后，Accuracy 基本不变，Macro-F1 与 baseline 基本持平，Balanced Accuracy 略低。这说明当前实现的 ABCD 特征与已有颜色、形状、纹理特征存在较多重叠，暂时没有明显带来额外性能提升。

不过这个方向仍然有报告价值：

1. 它提供了明确的医学规则来源。
2. 它能解释传统特征为什么这样设计。
3. 它可以作为消融实验中的可解释特征组。
4. 后续可以改进 asymmetry 和 border 的计算方式，例如使用主轴对齐、凸包、边界分段等。

## 是否建议合并

谨慎合并。

如果最终目标是单纯提高分数，目前 ABCD 特征不是最优先选择。

如果最终报告需要强调医学解释性和 ABCD 规则启发，可以保留该模块，并在消融实验中说明：ABCD 特征具备解释性，但在当前数据和当前实现下，性能提升有限。


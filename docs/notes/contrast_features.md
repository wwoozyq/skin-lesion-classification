# Lesion-background contrast features

## 做了什么

本次优化新增了病灶-背景对比特征。原 baseline 主要提取病灶区域内部的颜色、形状和纹理信息；这个方向进一步利用 mask 外的一圈周围皮肤，计算病灶区域和周围皮肤之间的差异。

直观理解：皮肤病变的判断不只看病灶本身，也看它相对于正常皮肤是否颜色异常、纹理异常、边界过渡异常。

## 参考资料

- Dermoscopedia: ABCD rule  
  https://dermoscopedia.org/ABCD_rule
- DermNet: Dermoscopy features of pigmented lesions  
  https://pro.dermnetnz.org/procedures/dermoscopy.html
- American Academy of Dermatology: ABCDEs of melanoma  
  https://www.aad.org/public/diseases/skin-cancer/find/at-risk/abcdes

这些资料都强调了皮肤病变诊断中的颜色、结构、边界和不对称性。基于这个思路，本次优化设计了病灶与周围皮肤之间的颜色/纹理对比特征。

## 新增代码

新增文件：

```text
src/features_contrast.py
```

新增核心函数：

```python
def extract_contrast_features(image, mask):
    ...
    return features
```

接入文件：

```text
src/features.py
src/train_ml.py
docs/EXPERIMENT_PLAN.md
```

## 新增特征

背景区域不是整张图的非病灶区域，而是 mask 膨胀后得到的病灶周围一圈区域。这样更接近病灶附近的正常皮肤。

新增特征包括：

- `contrast_rgb_*`: 病灶和背景在 RGB 通道上的均值差、标准差差、绝对均值差
- `contrast_hsv_*`: 病灶和背景在 HSV 通道上的差异
- `contrast_lab_*`: 病灶和背景在 Lab 通道上的差异
- `contrast_gray_mean_diff`: 灰度均值差
- `contrast_gray_std_diff`: 灰度标准差差
- `contrast_gray_entropy_diff`: 灰度熵差
- `contrast_grad_mean_diff`: 梯度强度均值差
- `contrast_grad_std_diff`: 梯度强度标准差差
- `contrast_lesion_area_ratio`: 病灶面积比例
- `contrast_background_ring_ratio`: 背景环面积比例

## 怎么运行

只测试 contrast 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set contrast
```

测试 baseline 特征 + contrast 特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_contrast
```

## 实验结果

使用同一个 RandomForest baseline 和 Stratified 5-Fold 评估。

| method | accuracy | macro_f1 | balanced_accuracy | mel_f1 | nv_f1 | vasc_f1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_all | 0.8917 | 0.8980 | 0.8908 | 0.86 | 0.90 | 0.93 |
| contrast_only | 0.8783 | 0.8921 | 0.8836 | 0.84 | 0.88 | 0.95 |
| all_contrast | 0.9067 | 0.9125 | 0.9028 | 0.88 | 0.91 | 0.94 |

## 结果分析

contrast 特征单独使用时，整体略低于 baseline，但 vasc 的 F1 达到 0.95，说明病灶-背景差异对血管性病变有一定区分能力。

将 contrast 特征与原有 all 特征合并后，三个主要指标均提升：

- Accuracy: 0.8917 -> 0.9067
- Macro-F1: 0.8980 -> 0.9125
- Balanced Accuracy: 0.8908 -> 0.9028

这说明病灶-背景对比特征和原有颜色/形状/纹理特征存在互补性。

## 是否建议合并

建议合并。

原因：

1. 方法有明确的医学图像解释性。
2. 代码接口符合项目规范。
3. 和 baseline 相比，综合指标有提升。
4. 对 vasc 类别尤其有帮助。


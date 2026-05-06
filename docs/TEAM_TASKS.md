# Team Tasks

## 一周机器学习路线分工

第一周目标：全组先把传统图像处理 + 机器学习 baseline 做稳。

| 成员 | 负责方向 | 主要任务 | 输出 |
|---|---|---|---|
| A | 数据与评估 | 读取 image/mask/label；分层划分；统一指标；混淆矩阵 | `dataset.py`, `evaluate.py` |
| B | 颜色特征 | RGB/HSV/Lab 统计；颜色直方图；病灶与背景颜色差 | `features_color.py` |
| C | 纹理特征 | LBP、GLCM、HOG；比较病灶区域和全图特征 | `features_texture.py` |
| D | 形状特征 | 面积、周长、圆度、长宽比、偏心率、边界不规则度 | `features_shape.py` |
| E | 分类器与调参 | SVM、Random Forest、XGBoost；特征选择；模型融合 | `train_ml.py` |

## 每日节奏

### Day 1

- 建好项目结构
- 所有人 clone 仓库并跑通 baseline
- 统一数据路径和输出格式

### Day 2

- B/C/D 分别完成第一版特征提取
- A 固定 train/val 或 Stratified K-Fold
- E 跑通单类特征 baseline

### Day 3

- 颜色、纹理、形状特征分别训练模型
- 记录每类特征的 accuracy、macro-F1、balanced accuracy

### Day 4

- 合并所有人工特征
- 对比 SVM、Random Forest、XGBoost、Logistic Regression

### Day 5

- 做特征标准化、特征选择、类别不平衡处理
- 输出特征重要性和混淆矩阵

### Day 6

- 做机器学习模型融合：soft voting / stacking
- 固化当前最佳机器学习 baseline

### Day 7

- 清理代码
- 从干净目录复现一遍
- 写第一周实验小结

## 每晚同步模板

```text
今天完成：
遇到问题：
明天计划：
需要别人配合：
```


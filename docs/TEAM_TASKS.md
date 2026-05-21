# Team Tasks

当前阶段不是继续从零开发，而是把已经完成的工作整理成一个完整、可讲、可验收的大作业项目。下面是推荐的 5 人分工。

## 分工总表

| 成员 | 负责模块 | 需要掌握的文件 | 最终产出 |
|---|---|---|---|
| A | 数据协议与防泄漏评估 | `src/utils.py`, `src/train_ml.py`, `scripts/check_grouped_cv_outputs.py` | 讲清楚为什么使用 grouped CV，解释 90% 到 70% 的变化 |
| B | 基础特征工程 | `src/features_color.py`, `src/features_shape.py`, `src/features_texture.py`, `src/features_contrast.py` | 讲颜色、形状、纹理、contrast 特征如何提取 |
| C | ABCD/boundary 主模型 | `src/features_abcd_grouped.py`, `src/features_boundary.py`, `docs/STRICT_GROUPED_RESULTS.md` | 讲最终主模型为什么选择 `all_abcd_grouped + LR03` |
| D | 模型搜索与融合实验 | `experiments/run_ml_grid.py`, `run_xgb_cascade_search.py`, `run_early_fusion_search.py`, `run_fusion_ensemble.py` | 讲 grid search、cascade、early/late fusion 和稳定性 |
| E | 分析与扩展 | `experiments/analyze_robustness.py`, `visualize_errors.py`, `train_deep.py` | 讲 robustness、错误样例、deep learning extension |

## 每个人必须能回答的问题

```text
1. 我负责的模块解决了什么问题？
2. 这个模块符合课程“传统方法”要求吗？
3. 使用了哪些输入和输出？
4. 对应实验结果是多少？
5. 最终是否被主模型采用？如果没有，为什么仍然有价值？
```

## 当前统一口径

主模型：

```text
all_abcd_grouped + LogisticRegression(C=0.3) + SelectKBest(k=140)
```

主模型结果：

| protocol | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| seed 127 grouped OOF | 0.7600 | 0.7715 | 0.7871 |
| five-seed mean | 0.7283 | 0.7435 | 0.7512 |

探索亮点：

| 方法 | 最好结果 | 口径 |
|---|---:|---|
| XGBoost cascade | balanced accuracy 0.8017 | seed 127 exploration |
| Early fusion | balanced accuracy 0.7997 | seed 127 exploration |
| Late probability fusion | balanced accuracy 0.8055 | seed 127 exploration |
| MobileNetV2 | accuracy 0.8750 | deep learning extension only |

注意：

```text
不要把 late fusion 的 0.8055 说成稳定主结果。
不要把 MobileNetV2 混进课程传统 ML 主线。
```

## 组员验收任务

每位组员在提交前至少完成一项验收：

| 成员 | 验收命令或材料 |
|---|---|
| A | 运行 `scripts/check_grouped_cv_outputs.py`，截图/记录 PASS 输出 |
| B | 列出 8-10 个基础特征例子，并说明医学/图像直觉 |
| C | 复现或解释 `all_abcd_grouped + LR03` 的 seed 127 指标 |
| D | 用 `docs/results_summary.csv` 做最终模型对比表 |
| E | 准备 robustness 表格和 mel/nv 错误样例图 |

## PPT 建议结构

1. Problem and data
2. Leakage risk and grouped CV
3. Traditional feature engineering
4. Stable main model and metrics
5. Robustness on augmented images
6. Error analysis
7. Exploration: cascade, early fusion, late fusion
8. Deep learning extension
9. Final conclusion and limitations

## 不建议继续做的事

- 继续盲目大规模刷模型。
- 把单 seed 最高分当作最终稳定结论。
- 把深度学习当作课程定量主提交。
- 上传数据、模型权重、大量 `outputs/` 文件到 GitHub。

## 推荐下一步

```text
1. 每个人按上表认领一块。
2. 按 docs/REPRODUCIBILITY_CHECKLIST.md 做一次验收。
3. 把 docs/results_summary.csv 转成 PPT 总表。
4. 用错误样例图讲清楚 mel/nv 为什么难。
5. 最后统一口径：稳定主模型 + 探索亮点 + 深度学习 extension。
```

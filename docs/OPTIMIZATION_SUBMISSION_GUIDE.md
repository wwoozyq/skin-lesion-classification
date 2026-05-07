# 机器学习优化方向提交指南

这份文档说明：如果你负责一个机器学习优化方向，应该做哪些事、交哪些文件、怎么跑实验、怎么提交到 GitHub。

可以参考已经完成的示范分支：

```text
feature/contrast-example
```

示范方向是：病灶-背景对比特征。

对应文件：

```text
src/features_contrast.py
docs/notes/contrast_features.md
docs/results/contrast_results.csv
```

## 1. 你的任务不是重写项目

每个人只负责一个明确优化方向，例如：

- ABCD 规则启发特征
- 病灶-背景对比特征
- 纹理特征
- 边界特征
- 分类器调参
- 特征选择
- 模型融合

你要做的是在现有 baseline 上增加一个模块，然后和 baseline 对比。

目前 baseline 指标：

| method | accuracy | macro_f1 | balanced_accuracy |
|---|---:|---:|---:|
| baseline_all | 0.8917 | 0.8980 | 0.8908 |

你的优化需要回答：

1. 参考了什么文献或规则？
2. 新增了什么特征或算法？
3. 怎么运行？
4. 和 baseline 相比有没有提升？
5. 是否建议合并进最终模型？

## 2. 开始前先跑通 baseline

先 clone 仓库：

```bash
git clone https://github.com/wwoozyq/skin-lesion-classification.git
cd skin-lesion-classification
```

安装依赖：

```bash
pip install -r requirements.txt
```

放好数据：

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

检查数据：

```bash
python scripts/check_dataset.py --data_dir data/Data_Proj2
```

跑 baseline：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all
```

如果能输出 accuracy、macro-F1、balanced accuracy，说明环境没问题。

## 3. 新建自己的分支

不要直接改 `main`。

如果你做 ABCD 特征：

```bash
git checkout -b feature/abcd
```

如果你做纹理特征：

```bash
git checkout -b feature/texture
```

如果你做分类器优化：

```bash
git checkout -b model/classifiers
```

分支命名建议：

```text
feature/abcd
feature/contrast
feature/texture
feature/border
model/classifiers
model/fusion
```

## 4. 需要交哪些文件

每个优化方向至少交 3 类文件。

### 4.1 代码文件

如果你做特征，新增：

```text
src/features_xxx.py
```

例如：

```text
src/features_abcd.py
src/features_texture_advanced.py
src/features_border.py
```

如果你做模型或调参，新增：

```text
src/train_compare_models.py
src/fusion.py
```

### 4.2 方法说明文件

放在：

```text
docs/notes/xxx_method.md
```

例如：

```text
docs/notes/abcd_features.md
docs/notes/texture_features.md
docs/notes/classifier_comparison.md
```

### 4.3 结果文件

放在：

```text
docs/results/xxx_results.csv
```

例如：

```text
docs/results/abcd_results.csv
docs/results/texture_results.csv
docs/results/classifier_results.csv
```

## 5. 特征代码怎么写

如果你做新特征，请写成这个格式：

```python
def extract_xxx_features(image, mask):
    features = {}
    features["xxx_feature_1"] = ...
    features["xxx_feature_2"] = ...
    return features
```

例子：

```python
def extract_abcd_features(image, mask):
    return {
        "abcd_area_ratio": ...,
        "abcd_asymmetry_lr": ...,
        "abcd_border_irregularity": ...,
        "abcd_color_entropy": ...,
    }
```

要求：

- 不要在特征函数里重新读取数据。
- 不要在特征函数里训练模型。
- 不要在特征函数里保存文件。
- 特征名必须加前缀，避免和别人冲突。

推荐前缀：

| 方向 | 前缀 |
|---|---|
| ABCD 特征 | `abcd_` |
| 病灶-背景对比 | `contrast_` |
| 纹理特征 | `texture_` |
| 边界特征 | `border_` |
| 形状特征 | `shape_` |

## 6. 方法说明文件怎么写

`docs/notes/xxx_method.md` 建议按这个格式：

```markdown
# 方法名称

## 做了什么

## 参考资料
- 链接 1
- 链接 2

## 新增特征或算法

## 怎么运行

## 实验结果
Accuracy:
Macro-F1:
Balanced Accuracy:

## 结果分析

## 是否建议合并
建议 / 不建议
原因：
```

示范文件：

```text
docs/notes/contrast_features.md
```

## 7. 结果表怎么写

`docs/results/xxx_results.csv` 至少包含：

```csv
method,accuracy,macro_f1,balanced_accuracy,mel_f1,nv_f1,vasc_f1,notes
baseline_all,0.8917,0.8980,0.8908,0.86,0.90,0.93,current baseline
your_method,,,,,,,
```

示范文件：

```text
docs/results/contrast_results.csv
```

示范结果：

| method | accuracy | macro_f1 | balanced_accuracy | notes |
|---|---:|---:|---:|---|
| baseline_all | 0.8917 | 0.8980 | 0.8908 | current baseline |
| contrast_only | 0.8783 | 0.8921 | 0.8836 | contrast features only |
| all_contrast | 0.9067 | 0.9125 | 0.9028 | baseline + contrast |

## 8. 怎么接入统一训练入口

如果你新增了特征文件，例如：

```text
src/features_abcd.py
```

先在自己的分支里实现函数：

```python
def extract_abcd_features(image, mask):
    ...
```

然后可以在 `src/features.py` 中接入：

```python
from .features_abcd import extract_abcd_features

if feature_set in {"abcd", "all_abcd"}:
    features.update(extract_abcd_features(image, mask))
```

同时在 `src/train_ml.py` 里把新的 `feature_set` 加到 choices 中。

如果不确定怎么接入，可以先只提交 `src/features_xxx.py` 和说明文档，由组长帮你接。

## 9. 怎么跑你的实验

以 contrast 示例为例：

只跑新增特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set contrast
```

跑 baseline + 新增特征：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_contrast
```

你的方向也应该类似：

```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set your_feature
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_your_feature
```

## 10. 怎么提交到 GitHub

先看改了哪些文件：

```bash
git status
```

添加你要提交的文件：

```bash
git add src/features_xxx.py
git add docs/notes/xxx_method.md
git add docs/results/xxx_results.csv
```

如果你改了统一入口，也加上：

```bash
git add src/features.py src/train_ml.py
```

提交：

```bash
git commit -m "feat: add xxx features"
```

推送分支：

```bash
git push -u origin feature/xxx
```

## 11. Pull Request 怎么写

到 GitHub 上打开自己的分支，点 `Compare & pull request`。

PR 描述按这个模板：

```markdown
## 做了什么
- 

## 参考资料
- 

## 新增文件
- 

## 怎么运行
```bash
python -m src.train_ml --data_dir data/Data_Proj2 --feature_set xxx
```

## 实验结果
| method | accuracy | macro_f1 | balanced_accuracy |
|---|---:|---:|---:|
| baseline_all | 0.8917 | 0.8980 | 0.8908 |
| your_method | | | |

## 是否建议合并
建议 / 不建议

原因：
```

不要自己直接 merge，等组长看完再合并。

## 12. 一个完整示范：contrast features

本仓库已有一个完整示范：

```text
branch: feature/contrast-example
```

它做了：

1. 选择方向：病灶-背景对比特征。
2. 查资料：ABCD/ABCDE、dermoscopy 里都强调颜色、边界和结构差异。
3. 写代码：`src/features_contrast.py`。
4. 接入训练入口：`src/features.py`、`src/train_ml.py`。
5. 跑实验：`contrast` 和 `all_contrast`。
6. 写说明：`docs/notes/contrast_features.md`。
7. 写结果：`docs/results/contrast_results.csv`。
8. commit 并 push 到 GitHub 分支。

大家后面负责其他方向时，就照这个格式来。


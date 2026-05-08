# External Data Report

## 结论

建议从 **ISIC 2018 Task 3 Training / HAM10000** 开始，但先不要把外部数据直接混进
课程数据的本地验证集。

最稳妥的用法是：

1. 课程数据仍然作为主训练和主验证来源。
2. 外部数据先做独立整理，保存为统一的 `label.csv`。
3. 第一阶段用于深度学习额外训练，或传统机器学习里的颜色、纹理特征实验。
4. 形状、边界、ABCD 特征暂时不要依赖外部数据，除非后续拿到或生成可靠 mask。

原因很简单：我们现在的 baseline 很多特征依赖 `mask/`，而 ISIC 2018 Task 3
分类数据本身提供的是图像和分类标签，不提供与每张分类图对应的病灶 mask。

## 本项目需要的数据格式

课程数据格式：

```text
Data_Proj2/
  image/
  mask/
  label.csv
```

`label.csv`：

```csv
image_id,dx
1,nv
2,mel
3,vasc
```

外部数据如果要完整接入传统机器学习，最好也有：

```text
image + mask + label
```

如果只有：

```text
image + label
```

它仍然有用，但用途要收窄。

## 候选 1: ISIC 2018 Task 3 Training

来源：

- [ISIC Challenge Data](https://challenge.isic-archive.com/data/)
- [HAM10000 paper](https://www.nature.com/articles/sdata2018161)

官方页面信息：

- Task 3 training data: 10015 images.
- Training ground truth: one CSV with 10015 label rows.
- License: CC BY-NC.
- 官方要求引用 ISIC 2018 challenge 和 HAM10000 paper。

我已下载官方 ground truth CSV 并做了统计。

原始 7 类分布：

| class | count |
|---|---:|
| nv | 6705 |
| mel | 1113 |
| bkl | 1099 |
| bcc | 514 |
| akiec | 327 |
| vasc | 142 |
| df | 115 |

能映射到本项目的 3 类：

| external label | project label | count |
|---|---|---:|
| MEL | mel | 1113 |
| NV | nv | 6705 |
| VASC | vasc | 142 |

可用总数：

```text
7960 / 10015
```

不建议使用的类别：

```text
BCC, AKIEC, BKL, DF
```

这些类别不属于老师给的 `mel/nv/vasc` 三分类任务。为了避免标签噪声，
第一阶段不要硬映射成其他类。

### 适合做什么

推荐：

- 深度学习额外训练。
- 颜色特征、纹理特征的外部训练实验。
- 作为额外测试集，观察模型跨数据集泛化。
- 写进报告的数据扩充与泛化分析部分。

谨慎：

- 直接和课程数据混合训练。
- 用它评价最终模型分数。

暂时不推荐：

- 用它做形状特征、边界特征、ABCD 特征。

原因是分类数据没有对应 mask。没有 mask 时，病灶区域不明确，形状和边界特征会失真。

## 候选 2: HAM10000 in ISIC Archive

来源：

- [ISIC Archive Collections](https://api.isic-archive.com/collections/)

ISIC Archive 的 HAM10000 collection 页面说明它包含常见色素性皮肤病变类别，
其中包括 Melanoma、Melanocytic Nevi 和 Vascular lesions。

需要注意：

- ISIC Archive 当前 collection 显示的图像数可能和 ISIC 2018 Task 3 的
  10015 张不完全一致。
- 为了复现稳定，第一阶段建议优先使用 ISIC Challenge 页面给出的
  2018 Task 3 固定下载包，而不是动态 Archive 查询结果。

### 适合做什么

和 ISIC 2018 Task 3 类似，但更适合作为后续扩展。

第一阶段先不直接使用 Archive API 批量拉图，因为动态查询会带来版本变化，
也会让组员复现成本变高。

## 候选 3: ISIC 2019 Training

来源：

- [ISIC Challenge Data](https://challenge.isic-archive.com/data/)

官方页面信息：

- Training JPEG: 33126 images.
- Training diagnosis labels: 33126 entries.
- Metadata: patient ID, lesion ID, sex, age, anatomic site.
- License: CC BY-NC.

优点：

- 样本更大。
- 仍然包含 `mel`、`nv`、`vasc` 等诊断类别。
- 有 metadata，后续可以做多模态或分组交叉验证。

风险：

- 数据更大，下载和管理成本明显提高。
- 包含 HAM10000、MSK、BCN20000 等来源，和 2018 数据有重叠风险。
- 第一阶段会增加项目复杂度。

建议：

暂时列为第二优先级。等 baseline 和第一轮优化稳定后，再考虑接入。

## 候选 4: ISIC 2018 Task 1 Segmentation

来源：

- [ISIC Challenge Data](https://challenge.isic-archive.com/data/)

官方页面信息：

- Task 1 training data: 2594 images.
- 每张图有 5 个 segmentation masks。

优点：

- 有 mask。
- 可以用于训练或验证分割模型。

限制：

- 它是 segmentation 任务，不是我们的三分类标签数据。
- 不能直接替代 `mel/nv/vasc` 分类训练集。

建议：

后续如果要给外部分类图生成 mask，可以考虑用 Task 1 训练一个轻量分割模型。
但这不是第一周必须做的事情。

## 推荐接入策略

### 第一步：只整理标签

先把 ISIC 2018 Task 3 的 ground truth 转成我们项目格式：

```csv
image_id,dx
ISIC_0024304,nv
ISIC_0024305,mel
```

只保留：

```text
MEL -> mel
NV -> nv
VASC -> vasc
```

### 第二步：不要上传外部图片

外部图片体积大，而且 license 是 CC BY-NC。仓库只保留：

- 数据来源说明。
- 下载链接。
- 标签转换脚本。
- 统计结果。

### 第三步：单独做实验

建议实验命名：

```text
external_isic2018_color_texture
external_isic2018_deep_pretrain
external_isic2018_generalization_test
```

不要把外部数据和课程验证集混在一起算主分数。

### 第四步：报告里这样写

可以写成：

> 为提高模型泛化性，我们调研了 ISIC 2018 Task 3 / HAM10000 数据集。
> 该数据集包含 10015 张皮肤镜图像，其中 7960 张可映射到本项目的
> mel/nv/vasc 三分类。考虑到外部分类数据缺少与本项目一致的 mask，
> 我们没有直接将其用于所有传统形状特征，而是将其作为深度学习预训练、
> 颜色纹理特征扩展和跨数据集泛化测试的候选数据源。

## 伪 mask 预处理方案

如果后续确实想把 ISIC 2018 Task 3 用到形状、边界或 ABCD 特征里，
可以先生成传统图像处理伪 mask。

已提供脚本：

```text
scripts/generate_pseudo_masks.py
scripts/preview_pseudo_masks.py
```

推荐目录：

```text
data/external/isic2018_task3/
  image/
    ISIC_0024306.jpg
    ...
  mask/
    mask_ISIC_0024306.jpg
    ...
  crop/
    ISIC_0024306.jpg
    ...
  label.csv
  manifest.csv
  summary.csv
```

生成前 100 张伪 mask：

```bash
python scripts/generate_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --output_dir data/external/isic2018_task3/mask \
  --max_images 100
```

生成 overlay 预览图：

```bash
python scripts/preview_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --mask_dir data/external/isic2018_task3/mask \
  --output_dir outputs/figures/pseudo_mask_preview \
  --max_images 30
```

伪 mask 算法大致流程：

1. 读取 RGB 图像。
2. 去掉接近纯黑的边框区域。
3. 从图像边缘估计皮肤背景颜色。
4. 在 Lab 颜色距离、亮度差异和饱和度差异上构造病灶分数。
5. 使用 Otsu 阈值分割候选区域。
6. 保留较大的连通区域，填洞，做形态学开闭运算。
7. 保存为 `mask_<image_id>.jpg`。

使用原则：

- 先生成 30 张 overlay 肉眼检查。
- 如果边界大体覆盖病灶，可以用来生成 lesion-centered crop。
- 不把 pseudo mask 当作真实分割标注使用。
- 不用它做周长、圆度、边界不规则性等精细形状特征。
- 报告中必须说明这是 pseudo mask，不是人工标注真值。

这一步的价值不是替代真实分割标注，而是给外部分类数据提供一个可测试的粗定位入口。

### 推荐用途：病灶中心裁剪

因为目前的传统伪 mask 边界不够精准，更稳妥的用法是把它当作 crop 工具：

```text
原图 -> pseudo mask -> 外接框加 padding -> 方形裁剪 -> resize 到 224x224
```

已提供脚本：

```text
scripts/crop_by_pseudo_masks.py
```

生成裁剪图：

```bash
python scripts/crop_by_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --mask_dir data/external/isic2018_task3/mask \
  --output_dir data/external/isic2018_task3/crop \
  --output_size 224
```

裁剪图适合用于：

- 深度学习分类训练。
- 颜色特征。
- 纹理特征。
- 外部数据泛化测试。

不建议用于：

- 精细 shape 特征。
- ABCD 中依赖边界精度的 B 项。
- 周长、圆度、边界粗糙度等定量结论。

报告中可以写：

> 由于外部分类数据不提供人工分割 mask，传统阈值方法生成的 pseudo mask
> 只能粗略定位病灶区域，边界精度不足。我们因此没有将其作为真实分割标注，
> 而是用于生成 lesion-centered crop，并将裁剪图用于深度学习和颜色纹理特征实验。

## 队长分工建议

你可以把这件事拆给一个同学：

```text
外部数据负责人：
1. 下载 ISIC 2018 Task 3 Training GroundTruth。
2. 用脚本生成三分类 label.csv。
3. 统计 mel/nv/vasc 数量。
4. 确认图片下载路径和本地目录结构。
5. 不提交图片，只提交脚本、统计表和说明文档。
```

交付文件：

```text
docs/external_data_report.md
scripts/prepare_isic2018_labels.py
scripts/generate_pseudo_masks.py
scripts/preview_pseudo_masks.py
scripts/crop_by_pseudo_masks.py
data/external/README.md
```

## 当前决定

建议当前项目采用这个策略：

```text
主线：课程数据 + baseline + 组员优化
支线：ISIC 2018 Task 3 标签整理
后续：深度学习阶段再决定是否下载外部图片
```

这样不会拖慢第一周机器学习优化，同时为后续高分报告留下一个清楚、有依据的数据扩展方向。

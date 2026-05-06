# Data Directory

课程数据不要上传到 GitHub。请在本地放成：

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

图片和 mask 的对应关系：

```text
image/1.jpg      -> mask/mask_1.jpg
image/23_aug1.jpg -> mask/mask_23_aug1.jpg
```

如果后续处理 HAM10000 / ISIC 外部数据，建议放在：

```text
external_data/
  images/
  labels_3class.csv
```

外部数据也不要直接上传到 GitHub，只上传处理脚本和说明。


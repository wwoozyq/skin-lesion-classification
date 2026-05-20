# Kaggle Deep Learning Training Guide

This guide explains how to run the optional deep-learning extension on Kaggle.
The deep-learning result is for presentation/report extension only. The course
quantitative submission should still use the traditional machine-learning
pipeline.

## 1. Create a Kaggle Dataset

Upload the course data as a private Kaggle Dataset. Keep this structure:

```text
Data_Proj2/
  image/
  mask/
  label.csv
```

After attaching the dataset to a Notebook, Kaggle will mount it under
`/kaggle/input/<dataset-slug>/`.

If you are unsure about the final path, run:

```python
!find /kaggle/input -maxdepth 4 -type d
```

The `--data_dir` argument should point to the folder that directly contains
`image/`, `mask/`, and `label.csv`.

## 2. Notebook Settings

In the Kaggle Notebook settings:

```text
Accelerator: GPU
Internet: On
```

Internet is needed for `git clone` and for downloading torchvision pretrained
weights the first time.

## 3. Clone the Project

```python
%cd /kaggle/working
!git clone https://github.com/wwoozyq/skin-lesion-classification.git
%cd /kaggle/working/skin-lesion-classification
```

## 4. Install Dependencies

Kaggle usually already has PyTorch installed, but installing the project
requirements keeps the environment consistent.

```python
!pip install -q -r requirements.txt
!pip install -q -r requirements-deep.txt
```

## 5. Check GPU and Data Path

```python
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

!find /kaggle/input -maxdepth 4 -type d
```

Set the data path according to the printed folder structure:

```python
DATA_DIR = "/kaggle/input/<dataset-slug>/Data_Proj2"
```

If your upload placed `image/`, `mask/`, and `label.csv` directly under the
dataset root, use:

```python
DATA_DIR = "/kaggle/input/<dataset-slug>"
```

## 6. Recommended Training Command

Start with MobileNetV2. It is faster and usually more stable on small datasets.

```python
!python experiments/train_deep.py \
  --data_dir "$DATA_DIR" \
  --model mobilenet_v2 \
  --pretrained \
  --crop \
  --cache_images \
  --epochs 20 \
  --freeze_backbone_epochs 4 \
  --image_size 192 \
  --batch_size 32 \
  --monitor accuracy \
  --output_dir outputs/deep/mobilenet_v2_pretrained_20ep
```

Minimum reference baseline:

```text
validation accuracy >= 0.65
```

This is only the lower bound for a reportable extension. If the validation
accuracy is already well above `0.65`, keep the best run and continue comparing
reasonable variants; the goal becomes the highest validation accuracy and
macro-F1, not merely passing `0.65`.

Local 1-epoch probe with pretrained MobileNetV2 already reached:

```text
Accuracy  = 0.6250
Macro-F1  = 0.6093
```

So 20 epochs on Kaggle GPU should be a realistic next attempt.

## 7. ResNet18 Alternative

If MobileNetV2 is unstable, try ResNet18:

```python
!python experiments/train_deep.py \
  --data_dir "$DATA_DIR" \
  --model resnet18 \
  --pretrained \
  --crop \
  --cache_images \
  --epochs 20 \
  --freeze_backbone_epochs 4 \
  --image_size 224 \
  --batch_size 32 \
  --monitor accuracy \
  --output_dir outputs/deep/resnet18_pretrained_20ep
```

If GPU memory is not enough, reduce batch size:

```text
--batch_size 16
```

## 8. Useful Variants

Lighter augmentation:

```python
!python experiments/train_deep.py \
  --data_dir "$DATA_DIR" \
  --model mobilenet_v2 \
  --pretrained \
  --crop \
  --cache_images \
  --epochs 20 \
  --freeze_backbone_epochs 4 \
  --image_size 192 \
  --batch_size 32 \
  --augment_strength light \
  --monitor accuracy \
  --output_dir outputs/deep/mobilenet_v2_light_aug
```

Try raw masks for cropping:

```python
!python experiments/train_deep.py \
  --data_dir "$DATA_DIR" \
  --model mobilenet_v2 \
  --pretrained \
  --crop \
  --mask_mode raw \
  --cache_images \
  --epochs 20 \
  --freeze_backbone_epochs 4 \
  --image_size 192 \
  --batch_size 32 \
  --monitor accuracy \
  --output_dir outputs/deep/mobilenet_v2_raw_mask
```

Monitor macro-F1 instead of accuracy:

```text
--monitor macro_f1
```

## 9. Output Files

Each run writes:

```text
outputs/deep/<run_name>/deep_history.csv
outputs/deep/<run_name>/deep_best_metrics.csv
outputs/deep/<run_name>/deep_best_classification_report.txt
outputs/deep/<run_name>/deep_best_confusion_matrix.csv
outputs/deep/<run_name>/<model>_best.pt
```

The main file to report is:

```text
deep_best_metrics.csv
```

## 10. Suggested Presentation Wording

Use this result as an extension:

> We also implemented a transfer-learning extension using ImageNet-pretrained
> MobileNetV2/ResNet18 with mask-based lesion cropping. This experiment is not
> used for the course quantitative submission, but it shows that deep features
> can be explored beyond handcrafted traditional features.

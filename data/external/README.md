# External Data

Do not commit external images to this repository.

Recommended local layout:

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
```

Use the script below to convert the official ISIC 2018 Task 3 ground truth file:

```bash
python scripts/prepare_isic2018_labels.py \
  --ground_truth_csv /path/to/ISIC2018_Task3_Training_GroundTruth.csv \
  --output_dir data/external/isic2018_task3
```

Only `MEL`, `NV`, and `VASC` are kept, because they map directly to this project:

```text
MEL  -> mel
NV   -> nv
VASC -> vasc
```

The generated files are small and may be used for local experiments. The image
files themselves should stay local.

## Small Sample Download

Before downloading the full ISIC 2018 image zip, start with a balanced small
sample:

```bash
python scripts/download_isic2018_sample.py \
  --ground_truth_csv /path/to/ISIC2018_Task3_Training_GroundTruth.csv \
  --output_dir data/external/isic2018_task3 \
  --max_per_class 20
```

This creates:

```text
data/external/isic2018_task3/
  image/
  label.csv
  manifest.csv
  summary.csv
```

The script downloads images from:

```text
https://isic-archive.s3.amazonaws.com/images/<image_id>.jpg
```

## Optional Pseudo Masks

ISIC 2018 Task 3 is a classification dataset. It does not provide one lesion
mask for each training image. If you want to test traditional shape or ABCD
features on external images, generate pseudo masks first:

```bash
python scripts/generate_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --output_dir data/external/isic2018_task3/mask \
  --max_images 100
```

Then create overlays for manual inspection:

```bash
python scripts/preview_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --mask_dir data/external/isic2018_task3/mask \
  --output_dir outputs/figures/pseudo_mask_preview \
  --max_images 30
```

Only use pseudo masks for experiments after checking overlay quality. If many
overlays are clearly wrong, use the external data for deep learning or
color/texture features instead of shape/ABCD features.

## Lesion-Centered Crops

Pseudo masks are often not accurate enough for boundary-sensitive features.
The safer use is to crop a lesion-centered image patch:

```bash
python scripts/crop_by_pseudo_masks.py \
  --image_dir data/external/isic2018_task3/image \
  --mask_dir data/external/isic2018_task3/mask \
  --output_dir data/external/isic2018_task3/crop \
  --output_size 224
```

Use these crops for:

- deep learning classification
- color features
- texture features

Do not treat pseudo-mask boundaries as manual segmentation labels.

## Crop Color / Texture Experiment

After generating crops, run a crop-only color/texture experiment:

```bash
python scripts/train_crop_color_texture.py \
  --image_dir data/external/isic2018_task3/crop \
  --label_csv data/external/isic2018_task3/label.csv \
  --feature_set color_texture \
  --output_dir outputs/external/isic2018_crop_color_texture
```

## External Generalization Test

Train color/texture features on the course data, then test on external crops:

```bash
python scripts/evaluate_external_generalization.py \
  --train_data_dir data/Data_Proj2 \
  --external_image_dir data/external/isic2018_task3/crop \
  --external_label_csv data/external/isic2018_task3/label.csv \
  --feature_set color_texture \
  --output_dir outputs/external/isic2018_generalization
```

Treat this as a robustness analysis, not as the main project score.

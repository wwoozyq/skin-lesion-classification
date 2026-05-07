# Experiment Plan

## Task

Input:

- `image/`: skin lesion RGB images
- `mask/`: lesion masks
- `label.csv`: labels for training

Output:

- `output.csv` with columns `image_id,dx`

Classes:

- `mel`
- `nv`
- `vasc`

## Machine Learning Baselines

### Color Features

- RGB mean / std / min / max / percentile
- HSV mean / std
- Lab mean / std
- HSV histogram
- Lesion-region color vs background color difference

### Texture Features

- LBP histogram
- GLCM contrast
- GLCM homogeneity
- GLCM energy
- GLCM correlation
- HOG

### Shape Features

- Mask area ratio
- Perimeter
- Circularity
- Bounding-box width / height
- Aspect ratio
- Eccentricity
- Solidity
- Extent
- Boundary irregularity

## Models

Start simple:

- Logistic Regression
- SVM
- Random Forest
- KNN

Then add:

- XGBoost or LightGBM if environment allows
- Soft voting
- Stacking

## Metrics

Use:

- Accuracy
- Macro-F1
- Balanced Accuracy
- Confusion matrix
- Per-class precision / recall / F1

Primary metric for local experiments:

```text
macro-F1
```

Reason: local data is imbalanced. `vasc` has fewer samples than `nv` and `mel`.

## Ablation Studies

Required:

1. Color features only
2. Texture features only
3. Shape features only
4. Lesion-background contrast features only
5. All baseline features
6. All baseline features + contrast features
7. With mask vs without mask
8. Different classifiers
9. Class weight vs no class weight

Optional:

1. Feature selection vs no feature selection
2. PCA vs no PCA
3. Soft voting vs best single model

## External Datasets

Possible datasets:

- HAM10000
- ISIC 2018 Task 3
- ISIC 2018 Task 1
- PH2

External data should be treated carefully:

- Do not mix external data into local validation.
- Prefer using external data for pretraining or extra training only.
- Keep a clear label mapping:
  - melanoma -> `mel`
  - melanocytic nevus -> `nv`
  - vascular lesion -> `vasc`

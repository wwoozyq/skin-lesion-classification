# Skin Lesion Classification

Biomedical Image Processing Project 2: traditional image processing and machine
learning for three-class skin lesion classification.

Classes:

- `mel`: melanoma
- `nv`: melanocytic nevus
- `vasc`: vascular lesion

The final prediction file is `output.csv`:

```csv
image_id,dx
1,nv
2,mel
3,vasc
```

## Current Status

The current main submission follows the course quantitative-evaluation rule:
**traditional image processing + simple machine learning only**. Deep learning
is implemented only as an optional extension for the report/presentation.

Best selected traditional model:

```text
feature_set = all_abcd_grouped
classifier  = LogisticRegression(C=0.3, class_weight="balanced")
selection   = SelectKBest(f_classif, k=140)
validation  = StratifiedGroupKFold by original lesion id
mask_mode   = raw
```

Strict grouped-CV result on seed `127`:

| metric | score |
|---|---:|
| Accuracy | 0.7600 |
| Macro-F1 | 0.7715 |
| Balanced Accuracy | 0.7871 |

Multi-seed grouped-CV stability over seeds `42, 127, 2024, 3407, 520`:

| metric | score |
|---|---:|
| Mean Macro-F1 | 0.7435 |
| Std Macro-F1 | 0.0271 |
| Mean Balanced Accuracy | 0.7512 |

## Why Grouped CV Matters

The dataset contains original images and augmented versions, for example:

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

Random image-level validation can leak information because an original lesion
may appear in training while its augmented version appears in validation. Early
image-level experiments reported around `0.91` macro-F1, but this was
optimistic. The current protocol groups all augmentations of the same lesion by
`base_id` and evaluates with `StratifiedGroupKFold`.

This is the most important methodological correction in the project.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── requirements-deep.txt
├── run.py
├── docs/
│   ├── CURRENT_EXPERIMENT_PROGRESS.md
│   ├── FULL_SCORE_EXPERIMENT_PLAN.md
│   ├── KAGGLE_DEEP_LEARNING.md
│   ├── ABCD_GROUPED_INTEGRATION.md
│   ├── PRESENTATION_HIGHLIGHTS.md
│   ├── STRICT_GROUPED_RESULTS.md
│   ├── TEAM_PROGRESS_OVERVIEW.md
│   └── TEXTURE_OPTIMIZATION.md
├── experiments/
│   ├── analyze_robustness.py
│   ├── run_ml_grid.py
│   ├── run_stability.py
│   ├── run_melnv_refinement.py
│   ├── train_deep.py
│   └── visualize_errors.py
├── scripts/
│   └── check_dataset.py
└── src/
    ├── dataset.py
    ├── evaluate.py
    ├── features.py
    ├── features_abcd.py
    ├── features_abcd_grouped.py
    ├── features_boundary.py
    ├── features_color.py
    ├── features_contrast.py
    ├── features_melnv.py
    ├── features_shape.py
    ├── features_texture.py
    ├── preprocess.py
    ├── train_ml.py
    └── utils.py
```

## Data Placement

Do not upload course images or masks to GitHub. Put the data locally as:

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

For this local workspace, the data directory is symlinked from:

```text
/Users/wlm/Downloads/Project/Data_Proj2
```

## Installation

This project is managed with `uv`.

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Optional deep-learning extension:

```bash
uv pip install --python .venv/bin/python -r requirements-deep.txt
```

## Quick Start

Check the dataset:

```bash
.venv/bin/python scripts/check_dataset.py --data_dir data/Data_Proj2
```

Train the selected traditional model:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw
```

Generate `output.csv`:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

## Main Experiments

Automated grid search:

```bash
.venv/bin/python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all,all_contrast,all_abcd_v2,all_boundary,final \
  --classifiers svm,rf,lr,knn \
  --k_features all,60,100 \
  --mask_modes raw \
  --output_csv outputs/metrics/ml_grid_raw.csv
```

ABCD grouped integration experiment:

```bash
.venv/bin/python experiments/run_abcd_grouped_integration.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_boundary,all_abcd_grouped,final_abcd_grouped \
  --classifiers lr,rf \
  --k_features 80,100,140 \
  --output_csv outputs/metrics/abcd_grouped_integration_lr_rf.csv
```

Mask-cleaning ablation:

```bash
.venv/bin/python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all,all_boundary,final \
  --classifiers svm,lr \
  --k_features all,60,100 \
  --mask_modes clean \
  --output_csv outputs/metrics/ml_grid_clean.csv
```

Robustness consistency:

```bash
.venv/bin/python experiments/analyze_robustness.py \
  --prediction_csv outputs/metrics/ml_all_boundary_lr_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/metrics/robustness_best
```

Error visualization:

```bash
.venv/bin/python experiments/visualize_errors.py \
  --data_dir data/Data_Proj2 \
  --prediction_csv outputs/metrics/ml_all_boundary_lr_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/figures/errors_best \
  --mask_mode raw
```

Multi-seed stability:

```bash
.venv/bin/python experiments/run_stability.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_boundary,all_boundary_melnv,final_melnv \
  --classifiers lr,svm \
  --k_features 80,100,160 \
  --mask_modes raw \
  --seeds 42,127,2024,3407,520 \
  --output_csv outputs/metrics/stability_melnv.csv
```

Optional low-load deep-learning night run:

```bash
caffeinate -dimsu .venv/bin/python experiments/run_deep_lowload_night.py \
  --data_dir data/Data_Proj2 \
  --device auto \
  --threads 2 \
  --max_hours 8
```

For Kaggle GPU training instructions, see
`docs/KAGGLE_DEEP_LEARNING.md`.

## Worth Reporting

The strongest presentation/report points are recorded in:

- `docs/PRESENTATION_HIGHLIGHTS.md`
- `docs/TEAM_PROGRESS_OVERVIEW.md`
- `docs/CURRENT_EXPERIMENT_PROGRESS.md`
- `docs/STRICT_GROUPED_RESULTS.md`
- `docs/FULL_SCORE_EXPERIMENT_PLAN.md`
- `docs/KAGGLE_DEEP_LEARNING.md`

Key messages:

1. We corrected augmentation leakage with grouped CV.
2. The old 90%+ result was inflated; strict validation gives a more honest
   70%+ result.
3. Boundary features plus balanced Logistic Regression are the most stable
   traditional baseline; the selectively integrated ABCD grouped features give
   the strongest current traditional score.
4. Most errors are between `mel` and `nv`, which matches the medical difficulty
   of distinguishing pigmented lesions.
5. Mask cleaning, mel/nv refinement, and deep learning were tested, but not
   adopted as the main quantitative submission because they did not improve the
   strict traditional evaluation.

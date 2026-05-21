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

For review, start with these tracked documents:

| document | purpose |
|---|---|
| `docs/PROJECT_REVIEW_GUIDE.md` | final project story, main model, evidence chain |
| `docs/EXPERIMENT_LEDGER.md` | all experiments, positive and negative conclusions |
| `docs/REPRODUCIBILITY_CHECKLIST.md` | commands for setup, validation, leakage check, output generation |
| `docs/results_summary.csv` | compact results table for PPT/report tables |

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

Experimental XGBoost cascade candidate, selectively integrated from
`origin/model`:

| protocol | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| Integrated smoke, seed `127` | 0.8017 | 0.7802 | 0.7617 |
| `origin/model` reproduction, seed `127` | 0.8006 | 0.7792 | 0.7600 |
| Best 5-seed bagged | 0.7887 | 0.7655 | 0.7500 |

This is a promising traditional-ML extension based on handcrafted
texture/asymmetry features plus a two-stage XGBoost cascade. It improves
class-balanced recognition, but it should be described carefully: balanced
accuracy can reach or approach `0.80`, while ordinary accuracy remains around
`0.75-0.76`.

Actual fusion candidate that preserves the original main model:

| method | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| Original `all_abcd_grouped + LR03` | 0.7871 | 0.7715 | 0.7600 |
| Pure cascade, best seed `127` smoke | 0.8017 | 0.7802 | 0.7617 |
| `0.5 original + 0.5 cascade` fusion | 0.8055 | 0.7909 | 0.7717 |

The fusion result is currently a seed `127` exploratory result. It is the
first real `1 + 1 > 1` result on that split. A follow-up five-seed check of
the same config gives bagged balanced accuracy `0.7763`, macro-F1 `0.7607`,
and accuracy `0.7417`, so the `0.8055` number should be reported as a
single-seed exploratory high point rather than a stable final score.

Feature-level early fusion was also tested as a front-end fusion baseline:

| method | protocol | balanced accuracy | macro-F1 | accuracy |
|---|---|---:|---:|---:|
| Best early fusion, `early_fusion_core + LR03` | seed `127`, OOF-tuned weights | 0.7997 | 0.7830 | 0.7700 |
| Best early fusion, same family | 5-seed mean, OOF-tuned weights | 0.7766 | 0.7618 | 0.7450 |

This supports an important report point: simply concatenating all handcrafted
features can approach `0.80` balanced accuracy on one split, but it is not
clearly more stable than model-level late fusion. Early fusion is therefore a
valuable ablation, while the current strongest exploratory candidate remains
`0.5 original + 0.5 cascade` late fusion.

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
│   ├── EARLY_FUSION_RESULTS.md
│   ├── EXPERIMENT_LEDGER.md
│   ├── FULL_SCORE_EXPERIMENT_PLAN.md
│   ├── FUSION_ENSEMBLE_RESULTS.md
│   ├── KAGGLE_DEEP_LEARNING.md
│   ├── DEEP_LEARNING_EXTENSION_RESULTS.md
│   ├── PROJECT_REVIEW_GUIDE.md
│   ├── REPRODUCIBILITY_CHECKLIST.md
│   ├── results_summary.csv
│   ├── ABCD_GROUPED_INTEGRATION.md
│   ├── PRESENTATION_HIGHLIGHTS.md
│   ├── STRICT_GROUPED_RESULTS.md
│   ├── TEAM_PROGRESS_OVERVIEW.md
│   ├── TEXTURE_OPTIMIZATION.md
│   └── XGB_CASCADE_INTEGRATION.md
├── experiments/
│   ├── analyze_robustness.py
│   ├── run_early_fusion_search.py
│   ├── run_ml_grid.py
│   ├── run_stability.py
│   ├── run_fusion_ensemble.py
│   ├── run_melnv_refinement.py
│   ├── run_xgb_cascade_search.py
│   ├── train_deep.py
│   └── visualize_errors.py
├── scripts/
│   ├── check_dataset.py
│   └── check_grouped_cv_outputs.py
└── src/
    ├── dataset.py
    ├── evaluate.py
    ├── features.py
    ├── features_abcd.py
    ├── features_abcd_grouped.py
    ├── features_boundary.py
    ├── features_color.py
    ├── features_contrast.py
    ├── features_gabor.py
    ├── features_lbp_multi.py
    ├── features_melnv.py
    ├── features_shape.py
    ├── features_subregion.py
    ├── features_texture.py
    ├── preprocess.py
    ├── train_ml_cascade.py
    ├── train_ml_fusion.py
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

Verify that grouped-CV artifacts have no augmentation leakage:

```bash
.venv/bin/python scripts/check_grouped_cv_outputs.py \
  --splits_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv \
  --oof_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
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

Experimental XGBoost cascade search:

```bash
.venv/bin/python experiments/run_xgb_cascade_search.py \
  --data_dir data/Data_Proj2 \
  --seeds 42,127,2024,3407,520 \
  --stage2_variants strong-reg,d2-more-trees,deeper \
  --k_features 100,120,all \
  --output_csv outputs/metrics/xgb_cascade_search.csv \
  --output_per_seed_csv outputs/metrics/xgb_cascade_search_per_seed.csv
```

Train and run the cascade candidate:

```bash
.venv/bin/python -m src.train_ml_cascade \
  --data_dir data/Data_Proj2 \
  --stage2_variant deeper \
  --stage2_k_features 120

.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/xgb_cascade_deeper_k120_soft.joblib \
  --output_csv output.csv
```

Fusion ensemble that preserves the original main model:

```bash
.venv/bin/python experiments/run_fusion_ensemble.py \
  --data_dir data/Data_Proj2 \
  --seeds 127 \
  --cascade_variants strong-reg,d2-more-trees,deeper \
  --cascade_k_features 100,120,all \
  --weights 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --output_csv outputs/metrics/fusion_ensemble_seed127_grid.csv \
  --output_per_seed_csv outputs/metrics/fusion_ensemble_seed127_grid_per_seed.csv

.venv/bin/python -m src.train_ml_fusion \
  --data_dir data/Data_Proj2 \
  --cascade_variant d2-more-trees \
  --cascade_k_features all \
  --cascade_weight 0.5

.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_fusion_candidate.joblib \
  --output_csv output.csv
```

Optional low-load deep-learning night run:

```bash
caffeinate -dimsu .venv/bin/python experiments/run_deep_lowload_night.py \
  --data_dir data/Data_Proj2 \
  --device auto \
  --threads 2 \
  --max_hours 8
```

The `0.65` accuracy line is only a minimum reference baseline; the runner keeps
comparing variants and reports the best validation accuracy/macro-F1 it finds.

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
- `docs/DEEP_LEARNING_EXTENSION_RESULTS.md`
- `docs/XGB_CASCADE_INTEGRATION.md`
- `docs/FUSION_ENSEMBLE_RESULTS.md`

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
6. A teammate XGBoost cascade idea was selectively integrated. It is worth
   presenting as an exploratory traditional-ML improvement because the
   integrated script reaches `0.8017` single-seed balanced accuracy and the
   reproduced 5-seed bagged result reaches `0.7887` balanced accuracy under
   grouped validation.
7. The real fusion version preserves the original `all_abcd_grouped + LR03`
   model and averages its OOF probabilities with the XGBoost cascade. On seed
   `127`, the best fusion reaches `0.8055` balanced accuracy, `0.7909`
   macro-F1, and `0.7717` ordinary accuracy.

# Project Review Guide

This document is the first stop for a teacher, teammate, or future reviewer who
wants to understand what this repository does and how the reported results are
verified.

## Project Goal

The assignment asks us to classify skin lesion images into three classes:

```text
mel   melanoma
nv    melanocytic nevus
vasc  vascular lesion
```

The course quantitative-evaluation rule requires traditional image processing
or simple machine learning methods. Therefore, the official project main line is
handcrafted features plus traditional machine learning. Deep learning is kept as
a report/presentation extension only.

## Final Main Line

The stable traditional model selected for the main submission is:

```text
feature_set = all_abcd_grouped
classifier  = LogisticRegression(C=0.3, class_weight="balanced")
selection   = SelectKBest(f_classif, k=140)
validation  = StratifiedGroupKFold by original lesion id
mask_mode   = raw
```

Seed `127` strict grouped-CV result:

| metric | score |
|---|---:|
| Accuracy | 0.7600 |
| Macro-F1 | 0.7715 |
| Balanced Accuracy | 0.7871 |

Five-seed stability over `42, 127, 2024, 3407, 520`:

| metric | mean | std |
|---|---:|---:|
| Accuracy | 0.7283 | 0.0289 |
| Macro-F1 | 0.7435 | 0.0271 |
| Balanced Accuracy | 0.7512 | 0.0297 |

This model is not the single highest number in the repository. It is selected
because it is simple, rule-compliant, interpretable, and already validated with
strict grouped CV and multi-seed stability.

## What We Explored

The repository records both positive and negative experiments. This is
intentional: a strong project should show that we tested ideas and rejected the
ones that did not generalize.

| direction | status | takeaway |
|---|---|---|
| Strict grouped CV | adopted | Prevents augmented-image leakage and explains the drop from 90%+ to 70%+. |
| Color/shape/texture baseline | adopted as foundation | Gives a simple handcrafted-feature baseline. |
| Contrast and boundary features | adopted as evidence | Helped the first strict model, especially for vascular lesions. |
| ABCD grouped features | adopted in main model | Improved seed `127` and five-seed stability over the earlier boundary model. |
| Mel/NV targeted features | recorded as negative/diagnostic | Interpretable, but did not beat the main model consistently. |
| Mask cleaning | recorded as ablation | Useful to test, but raw mask performed better for the selected model. |
| Robustness consistency | adopted as analysis | Measures original-vs-augmented behavior required by the assignment. |
| Error visualization | adopted as analysis | Shows most remaining errors are `mel`/`nv` confusions. |
| XGBoost cascade | exploratory extension | Reached higher single-seed balanced accuracy, but more complex. |
| Early feature fusion | exploratory ablation | Nearly reaches `0.80` balanced accuracy on seed `127`, less stable. |
| Late probability fusion | exploratory ablation | Best seed `127` result, but five-seed result is more modest. |
| MobileNetV2 deep learning | presentation extension | Strong score, but not used for the official traditional main line. |

## Key Result Table

| method | protocol | accuracy | macro-F1 | balanced accuracy | role |
|---|---|---:|---:|---:|---|
| Earlier boundary LR | seed `127` grouped OOF | 0.7233 | 0.7454 | 0.7535 | historical baseline |
| Main ABCD grouped LR03 | seed `127` grouped OOF | 0.7600 | 0.7715 | 0.7871 | stable main model |
| Main ABCD grouped LR03 | five-seed mean | 0.7283 | 0.7435 | 0.7512 | stability estimate |
| XGBoost cascade | seed `127` grouped OOF | 0.7617 | 0.7802 | 0.8017 | exploratory high-score |
| XGBoost cascade | best five-seed bagged | 0.7500 | 0.7655 | 0.7887 | exploratory stability |
| Early fusion | seed `127` grouped OOF | 0.7700 | 0.7830 | 0.7997 | feature-fusion ablation |
| Early fusion | best five-seed mean | 0.7450 | 0.7618 | 0.7766 | feature-fusion stability |
| Late probability fusion | seed `127` grouped OOF | 0.7717 | 0.7909 | 0.8055 | best single-seed result |
| Late probability fusion | five-seed bagged | 0.7417 | 0.7607 | 0.7763 | fusion stability check |
| MobileNetV2 | grouped validation extension | 0.8750 | 0.8623 | 0.8533 | deep-learning extension only |

## Why 90% Became 70%

The dataset contains one original image and two augmented images per original
lesion. A random image-level split can put the original lesion in training and
an augmentation of the same lesion in validation:

```text
train: 100.jpg
valid: 100_aug1.jpg
```

This leaks lesion identity and inflates validation results. The corrected
protocol uses `StratifiedGroupKFold` with `base_id` as the group, so all images
derived from the same original lesion stay in the same fold.

This correction is the most important methodological contribution of the
project. It makes the lower strict-CV score more honest and defensible.

## Evidence Chain

Use these files to audit the project:

| evidence | file |
|---|---|
| Main model metric | `outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_metrics.csv` |
| Main model fold assignment | `outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv` |
| Main model OOF predictions | `outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv` |
| Main model confusion matrix | `outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_confusion_matrix.csv` |
| Five-seed stability | `outputs/metrics/stability_abcd_grouped_argmax_summary.csv` |
| Robustness consistency | `outputs/metrics/robustness_abcd_grouped/robustness_group_summary.csv` |
| Error visualization figures | `outputs/figures/errors_abcd_grouped/` |
| Cascade result | `outputs/metrics/xgb_cascade_smoke_seed127.csv` |
| Early fusion result | `outputs/metrics/early_fusion_seed127.csv` |
| Late fusion five-seed result | `outputs/metrics/fusion_ensemble_d2_all_w05_multiseed.csv` |
| Deep learning extension | `docs/DEEP_LEARNING_EXTENSION_RESULTS.md` |

Local output artifacts under `outputs/` are intentionally ignored by Git because
they can be regenerated and may contain large files. The important summary
numbers and commands are recorded in tracked documentation.

## How To Review The Repo

Recommended reading order:

1. `README.md` for setup and quick start.
2. `docs/PROJECT_REVIEW_GUIDE.md` for the final project story.
3. `docs/EXPERIMENT_LEDGER.md` for all experiments and conclusions.
4. `docs/REPRODUCIBILITY_CHECKLIST.md` for exact validation commands.
5. `docs/DEEP_LEARNING_EXTENSION_RESULTS.md` only if reviewing the extension.

Recommended grading position:

```text
Official quantitative line:
  traditional handcrafted features + grouped CV + Logistic Regression

Additional report/presentation highlights:
  cascade, early fusion, late fusion, robustness, error visualization,
  deep-learning extension
```

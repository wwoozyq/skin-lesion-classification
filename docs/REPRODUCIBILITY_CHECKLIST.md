# Reproducibility And Acceptance Checklist

Use this checklist before submitting the repository, sending it to teammates, or
showing it to the teacher.

## 1. Environment

The project uses `uv` for dependency management.

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Optional deep-learning extension:

```bash
uv pip install --python .venv/bin/python -r requirements-deep.txt
```

If XGBoost fails to import on macOS because of OpenMP, use the local sklearn
OpenMP library path:

```bash
DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
.venv/bin/python -c "import xgboost; print('xgboost ok')"
```

## 2. Data Placement

Do not commit the course images or masks.

Expected local layout:

```text
data/Data_Proj2/
  image/
  mask/
  label.csv
```

Check dataset integrity:

```bash
.venv/bin/python scripts/check_dataset.py --data_dir data/Data_Proj2
```

Acceptance:

```text
The script should find all expected images, masks, and labels.
```

## 3. Main Model Training

Train the stable traditional main model:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw
```

Expected artifacts:

```text
outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib
outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_metrics.csv
outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv
outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_confusion_matrix.csv
```

Expected seed `127` metrics:

| metric | expected value |
|---|---:|
| Accuracy | 0.7600 |
| Macro-F1 | 0.7715 |
| Balanced Accuracy | 0.7871 |

Small floating-point differences are acceptable, but a large drop should be
investigated.

## 4. Grouped-CV Leakage Check

Verify that no original lesion group crosses validation folds:

```bash
.venv/bin/python scripts/check_grouped_cv_outputs.py \
  --splits_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv \
  --oof_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
```

Acceptance:

```text
PASS: no base_id appears in more than one fold.
PASS: OOF predictions cover the split image set.
```

This is the most important anti-leakage check in the project.

## 5. Prediction File Generation

Generate the final course-style prediction file:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

Acceptance:

```text
output.csv has exactly two columns: image_id, dx
dx values are limited to mel, nv, vasc
For the local 600-image dataset, output.csv has 601 lines including header.
```

Quick check:

```bash
head output.csv
wc -l output.csv
```

## 6. Multi-Seed Stability

The stable model has already been checked over seeds:

```text
42, 127, 2024, 3407, 520
```

Recorded summary:

```text
outputs/metrics/stability_abcd_grouped_argmax_summary.csv
```

Expected five-seed summary:

| metric | mean | std |
|---|---:|---:|
| Accuracy | 0.7283 | 0.0289 |
| Macro-F1 | 0.7435 | 0.0271 |
| Balanced Accuracy | 0.7512 | 0.0297 |

Acceptance:

```text
Report both seed-127 selected split and five-seed stability.
Do not claim the seed-127 number is the stable expected score.
```

## 7. Robustness Analysis

Run robustness analysis on grouped OOF predictions:

```bash
.venv/bin/python experiments/analyze_robustness.py \
  --prediction_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/metrics/robustness_abcd_grouped
```

Recorded interpretation:

```text
Original image accuracy: 0.7800
Augmented image accuracy: 0.7500
Prediction consistency:  0.7350
```

Acceptance:

```text
The report should include original-vs-augmented performance, not only overall
accuracy.
```

## 8. Error Visualization

Generate error panels:

```bash
.venv/bin/python experiments/visualize_errors.py \
  --data_dir data/Data_Proj2 \
  --prediction_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/figures/errors_abcd_grouped \
  --mask_mode raw \
  --max_per_pair 6
```

Acceptance:

```text
The presentation/report should show that most remaining errors are mel/nv
confusions, with example images.
```

## 9. Extension Checks

These are not the official traditional main line, but they are useful for the
report.

XGBoost cascade:

```bash
DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
.venv/bin/python experiments/run_xgb_cascade_search.py \
  --data_dir data/Data_Proj2 \
  --seeds 127 \
  --stage2_variants strong-reg \
  --k_features 120 \
  --cascade_modes soft \
  --n_jobs 2
```

Early fusion:

```bash
DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
.venv/bin/python experiments/run_early_fusion_search.py \
  --data_dir data/Data_Proj2 \
  --feature_sets early_fusion_core,early_fusion_full \
  --classifiers lr03,et,hgb,xgb_strong_reg,xgb_d2_more_trees \
  --k_features 100,140,180,220,all \
  --seeds 127 \
  --output_csv outputs/metrics/early_fusion_seed127.csv \
  --n_jobs 2
```

Late fusion five-seed check:

```bash
DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
.venv/bin/python experiments/run_fusion_ensemble.py \
  --data_dir data/Data_Proj2 \
  --seeds 42,127,2024,3407,520 \
  --cascade_variants d2-more-trees \
  --cascade_k_features all \
  --weights 0.5 \
  --output_csv outputs/metrics/fusion_ensemble_d2_all_w05_multiseed.csv \
  --output_per_seed_csv outputs/metrics/fusion_ensemble_d2_all_w05_multiseed_per_seed.csv \
  --n_jobs 2
```

Acceptance:

```text
These results may be shown as exploration and ablation.
Only the grouped traditional main model should be described as the stable
official submission.
```

## 10. Repository Cleanliness Before GitHub

Run:

```bash
.venv/bin/python -m compileall src experiments scripts
git diff --check
git status --short
```

Check that:

```text
Data files are not staged.
Large model files are not staged unless explicitly required.
Generated outputs remain ignored.
Docs contain the key numbers needed for review.
README points reviewers to the right documents.
```

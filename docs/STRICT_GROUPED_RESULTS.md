# Strict Grouped CV Results

## Why This Evaluation Matters

The dataset contains original images and augmented versions:

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

A random image-level split leaks information because images from the same
original lesion can appear in both training and validation folds. The project
therefore uses:

```text
StratifiedGroupKFold
group = base_id
```

This keeps each original lesion and its augmentations in one fold.

## Split Protocol

| item | setting |
|---|---|
| CV | grouped 5-fold |
| group key | original `base_id` |
| selected split seed | `127` |
| stability seeds | `42, 127, 2024, 3407, 520` |
| primary reported metrics | accuracy, macro-F1, balanced accuracy |
| official model family | traditional features + simple ML |

The seed `127` split is used as the selected split for detailed analysis. The
five-seed result is used to avoid over-claiming stability.

## Historical Baseline

Earlier strict grouped-CV result:

| model | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| `all_boundary + LR + k=100` | 0.7233 | 0.7454 | 0.7535 |

Five-seed stability:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_boundary + LR + k=100` | 0.7077 | 0.7242 | 0.0143 | 0.7301 |

This was the first solid strict model after correcting augmentation leakage.

## Current Main Model

Selected stable traditional model:

```text
feature_set = all_abcd_grouped
classifier  = LogisticRegression(C=0.3, class_weight="balanced")
k_features  = 140
mask_mode   = raw
```

Seed `127` grouped-CV result:

| model | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| `all_abcd_grouped + LR03 + k=140` | 0.7600 | 0.7715 | 0.7871 |

Five-seed stability:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_abcd_grouped + LR03 + k=140` | 0.7283 | 0.7435 | 0.0271 | 0.7512 |

The selected main model improves over the earlier boundary LR model in both
seed `127` and five-seed stability.

## Related Exploratory Results

| method | protocol | accuracy | macro-F1 | balanced accuracy |
|---|---|---:|---:|---:|
| XGBoost cascade | seed `127` grouped OOF | 0.7617 | 0.7802 | 0.8017 |
| Early fusion | seed `127` grouped OOF | 0.7700 | 0.7830 | 0.7997 |
| Late probability fusion | seed `127` grouped OOF | 0.7717 | 0.7909 | 0.8055 |
| Late probability fusion | five-seed bagged | 0.7417 | 0.7607 | 0.7763 |

These are valuable exploratory results, but the final official main line remains
the simpler ABCD grouped LR03 model.

## Verification Commands

Train the main model:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw
```

Verify grouped split artifacts:

```bash
.venv/bin/python scripts/check_grouped_cv_outputs.py \
  --splits_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_splits.csv \
  --oof_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
```

Expected check:

```text
PASS: no base_id appears in more than one fold.
PASS: OOF predictions cover the split image set.
```

Generate predictions:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

# Strict Grouped CV Results

## Why This Evaluation Matters

The dataset contains original images and augmented versions such as:

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

A random image-level split leaks information because augmented versions of the
same original lesion can appear in both training and validation folds. The
current evaluation therefore uses `StratifiedGroupKFold`, where the group is the
original `base_id`.

## Split Protocol

- CV: grouped 5-fold
- Group key: original image id after removing augmentation suffixes
- Split seed: `127`
- Reason for seed: selected by label balance only, not by model performance
- Classifier search: balanced RBF SVM, Random Forest, Logistic Regression, KNN
- Feature selection search: all features, top 60, top 100 by `SelectKBest`
- Primary metric: macro-F1

Each fold contains 40 original groups and 120 images.

## Results

| feature set | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| `all` | 0.6517 | 0.6697 | 0.6578 |
| `all_boundary` | 0.6733 | 0.6847 | 0.6666 |
| `all_abcd_v2` | 0.6867 | 0.7088 | 0.6894 |
| `all_contrast` | 0.6900 | 0.7090 | 0.6895 |
| `final` + SVM | 0.7083 | 0.7327 | 0.7119 |
| `all_boundary` + LR + top 100 | **0.7233** | **0.7454** | **0.7535** |

The old image-level validation reported macro-F1 around 0.911. Under grouped
validation, the baseline drops to 0.6697, showing that the original validation
protocol overestimated performance because of augmentation leakage.

## Stability Check

The best single-seed model was also tested across five grouped split seeds
(`42`, `127`, `2024`, `3407`, `520`):

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_boundary` + LR + top 100 | 0.7077 | 0.7242 | 0.0143 | 0.7301 |

This means the seed-127 score is a strong but plausible split. For the report,
use 0.7454 as the selected split result and 0.7242 +/- 0.0143 as the stability
estimate.

## Final Feature Set

`final` combines:

- baseline color, texture, and shape features
- lesion-background contrast features
- ABCD-inspired v2 features
- boundary features

The automated grid search further improves strict macro-F1 from 0.7327 to
0.7454 by selecting `all_boundary` features with Logistic Regression and the top
100 univariate features.

## Commands

```bash
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all --classifier svm --cv grouped
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_contrast --classifier svm --cv grouped
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_abcd_v2 --classifier svm --cv grouped
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_boundary --classifier svm --cv grouped
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set final --classifier svm --cv grouped
.venv/bin/python -m src.train_ml --data_dir data/Data_Proj2 --feature_set all_boundary --classifier lr --k_features 100 --cv grouped --mask_mode raw
```

Generate predictions with the final model:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_boundary_lr_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

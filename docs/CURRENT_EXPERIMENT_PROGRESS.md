# Current Experiment Progress

This document records the current strict grouped-validation results and the
files generated after the latest experiment round.

## 1. Automated Grid Search

Main grid search:

```bash
.venv/bin/python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all,all_contrast,all_abcd_v2,all_boundary,final \
  --classifiers svm,rf,lr,knn \
  --k_features all,60,100 \
  --mask_modes raw \
  --output_csv outputs/metrics/ml_grid_raw.csv
```

Best strict grouped-CV result:

| rank | feature set | mask | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `all_boundary` | raw | Logistic Regression, C=1.0 | 100 | 0.7233 | 0.7454 | 0.7535 |
| 2 | `final` | raw | SVM, C=100 | 100 | 0.7150 | 0.7370 | 0.7208 |
| 3 | `final` | raw | SVM, C=30 | 100 | 0.7150 | 0.7370 | 0.7208 |

Final traditional model selected for now:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_boundary \
  --classifier lr \
  --k_features 100 \
  --cv grouped \
  --mask_mode raw
```

Saved model:

```text
outputs/models/ml_all_boundary_lr_grouped_raw_seed127.joblib
```

Strict grouped-CV classification summary:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| mel | 0.65 | 0.74 | 0.69 | 210 |
| nv | 0.76 | 0.67 | 0.71 | 300 |
| vasc | 0.82 | 0.84 | 0.83 | 90 |
| macro avg | 0.74 | 0.75 | 0.75 | 600 |

## 2. Mask Cleaning Experiment

Mask-cleaning grid:

```bash
.venv/bin/python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all,all_boundary,final \
  --classifiers svm,lr \
  --k_features all,60,100 \
  --mask_modes clean \
  --output_csv outputs/metrics/ml_grid_clean.csv
```

Best clean-mask result:

| feature set | mask | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---|---|---|---:|---:|---:|---:|
| `all` | clean | Logistic Regression, C=1.0 | all | 0.7217 | 0.7373 | 0.7350 |

Conclusion:

The current mask-cleaning strategy is useful as an ablation but does not beat
the best raw-mask model. The selected final traditional model therefore keeps
`mask_mode=raw`.

## 3. Robustness Consistency Analysis

Command:

```bash
.venv/bin/python experiments/analyze_robustness.py \
  --prediction_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/metrics/robustness_abcd_grouped
```

Image-level robustness:

| image type | n images | accuracy |
|---|---:|---:|
| original | 200 | 0.7800 |
| augmented | 400 | 0.7500 |

Group-level robustness:

| n groups | prediction consistency | all images correct | any image correct | all match original |
|---:|---:|---:|---:|---:|
| 200 | 0.7350 | 0.6300 | 0.8750 | 0.7350 |

Interpretation:

The new ABCD grouped model improves both original-image and augmented-image
accuracy. The original/augmented gap remains, but prediction consistency
improves from 0.6800 to 0.7350, making this a stronger robustness story than
the previous candidate model.

## 4. Error Example Visualization

Command:

```bash
.venv/bin/python experiments/visualize_errors.py \
  --data_dir data/Data_Proj2 \
  --prediction_csv outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv \
  --output_dir outputs/figures/errors_abcd_grouped \
  --mask_mode raw \
  --max_per_pair 6
```

Error distribution:

| true label | predicted label | n errors | figure |
|---|---|---:|---|
| nv | mel | 59 | `outputs/figures/errors_abcd_grouped/errors_true_nv_pred_mel.png` |
| mel | nv | 53 | `outputs/figures/errors_abcd_grouped/errors_true_mel_pred_nv.png` |
| nv | vasc | 14 | `outputs/figures/errors_abcd_grouped/errors_true_nv_pred_vasc.png` |
| mel | vasc | 9 | `outputs/figures/errors_abcd_grouped/errors_true_mel_pred_vasc.png` |
| vasc | nv | 6 | `outputs/figures/errors_abcd_grouped/errors_true_vasc_pred_nv.png` |
| vasc | mel | 3 | `outputs/figures/errors_abcd_grouped/errors_true_vasc_pred_mel.png` |

Main observation:

Most remaining errors are still between `nv` and `mel`, which is expected
because both can be pigmented lesions with overlapping color and boundary
patterns. The new ABCD grouped features substantially reduce `nv -> mel`
errors from 84 to 59, but introduce a small increase in `mel -> vasc` and
`vasc -> mel` mistakes.

## 5. Deep Learning Extension

Installed optional dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements-deep.txt
```

Smoke experiment:

```bash
.venv/bin/python experiments/train_deep.py \
  --data_dir data/Data_Proj2 \
  --crop \
  --epochs 2 \
  --output_dir outputs/deep/resnet18_smoke
```

Result:

| epoch | validation accuracy | validation macro-F1 |
|---:|---:|---:|
| 1 | 0.3417 | 0.2862 |
| 2 | 0.4250 | 0.4197 |

Interpretation:

This is only a pipeline smoke test, not a final deep-learning result. It proves
that grouped split, mask-based crop, training, validation, and model saving all
work. Deep learning should remain an extension in the presentation/report, not
the main quantitative submission, because the course quantitative assessment
requires traditional image processing or simple machine learning.

## 6. Mel/NV Error-Driven Optimization

Motivation:

The previous error visualization showed that most mistakes are between `mel`
and `nv`:

| true label | predicted label | n errors |
|---|---|---:|
| nv | mel | 84 |
| mel | nv | 51 |

Therefore, we tested two error-driven improvements.

### 6.1 Mel/NV-Focused Features

New feature module:

```text
src/features_melnv.py
```

New feature sets:

| feature set | content |
|---|---|
| `melnv` | pigment/asymmetry/variegation features only |
| `all_melnv` | baseline color, shape, texture plus mel/nv features |
| `all_boundary_melnv` | current best `all_boundary` plus mel/nv features |
| `final_melnv` | full feature set plus mel/nv features |

The mel/nv features include Lab color spread, HSV variegation, PCA-axis
asymmetry, quadrant color imbalance, lesion core vs border color differences,
3x3 local color heterogeneity, and dark/high-saturation component statistics.

Command:

```bash
.venv/bin/python experiments/run_ml_grid.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_boundary,all_melnv,all_boundary_melnv,final_melnv \
  --classifiers svm,lr \
  --k_features all,80,100,120,160 \
  --mask_modes raw \
  --output_csv outputs/metrics/ml_grid_melnv.csv
```

Top single-seed results:

| feature set | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---|---|---:|---:|---:|---:|
| `all_boundary` | LR | 100 | 0.7233 | 0.7454 | 0.7535 |
| `final_melnv` | SVM | 80 | 0.6967 | 0.7314 | 0.7206 |
| `final_melnv` | LR | 80 | 0.7100 | 0.7265 | 0.7420 |
| `all_boundary_melnv` | LR | 100 | 0.7067 | 0.7240 | 0.7344 |

Conclusion:

The mel/nv-focused features are interpretable, but they do not improve the
current best strict grouped-CV model. The likely reason is that they partly
duplicate information already captured by boundary, color, texture, and ABCD
features, while also increasing dimensionality on a small dataset.

### 6.2 Multi-Seed Stability

Command:

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

Best stability results:

| feature set | classifier | k | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---|---:|---:|---:|---:|
| `all_boundary` | LR | 100 | 0.7242 | 0.0143 | 0.7301 |
| `final_melnv` | SVM | 100 | 0.7167 | 0.0181 | 0.7069 |
| `all_boundary` | LR | 80 | 0.7156 | 0.0123 | 0.7253 |
| `final_melnv` | LR | 80 | 0.7147 | 0.0078 | 0.7261 |

Conclusion:

Before the ABCD grouped branch integration, the `all_boundary` LR model was the
most stable candidate. Its seed-127 score of 0.7454 was a strong fold split,
while the more honest multi-seed expectation was around 0.72-0.73 macro-F1.

### 6.3 Two-Stage Mel/NV Refinement

We also tested a hierarchical approach:

1. Train the main 3-class model.
2. If the main model predicts `mel` or `nv`, use a second binary `mel-vs-nv`
   classifier to refine the prediction.
3. Keep `vasc` predictions unchanged.

Command:

```bash
.venv/bin/python experiments/run_melnv_refinement.py \
  --data_dir data/Data_Proj2 \
  --main_feature_set all_boundary \
  --main_classifier lr \
  --main_k_features 100 \
  --ref_feature_sets all_boundary,all_boundary_melnv,final_melnv \
  --ref_classifiers lr,svm \
  --ref_k_features 80,100,160 \
  --mask_mode raw \
  --seeds 42,127,2024,3407,520 \
  --output_csv outputs/metrics/melnv_refinement.csv
```

Best refinement result:

| ref feature set | ref classifier | ref k | main mean macro-F1 | refined mean macro-F1 | mean delta |
|---|---|---:|---:|---:|---:|
| `all_boundary` | LR | 80 | 0.7242 | 0.7241 | -0.0001 |

Conclusion:

Two-stage refinement should not be adopted. It is useful as an experiment, but
it does not consistently improve the main model and often over-corrects mel/nv
predictions.

## 7. ABCD Grouped Branch Integration

After inspecting `origin/abcd-grouped-optimization`, we did not merge the branch
directly because it committed generated `outputs/` files and rewrote core
training code. Instead, we selectively integrated its useful ideas as an
independent feature module and experiment.

New files / interfaces:

```text
src/features_abcd_grouped.py
experiments/run_abcd_grouped_integration.py
feature sets: abcd_grouped, all_abcd_grouped, final_abcd_grouped
classifier: lr03 = LogisticRegression(C=0.3)
```

Best seed-127 strict grouped-CV result:

| model | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| previous `all_boundary` + LR + top100 | 0.7233 | 0.7454 | 0.7535 |
| new `all_abcd_grouped` + `lr03` + top140 | **0.7600** | **0.7715** | **0.7871** |
| threshold ablation, `mel_threshold=0.45` | 0.7600 | 0.7730 | 0.7910 |

Five-seed stability:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| previous `all_boundary` + LR + top100 | 0.7077 | 0.7242 | 0.0143 | 0.7301 |
| new `all_abcd_grouped` + `lr03` + top140 | **0.7283** | **0.7435** | 0.0271 | **0.7512** |
| threshold ablation, `mel_threshold=0.45` | 0.7257 | 0.7426 | 0.0284 | 0.7520 |

Conclusion:

The selectively integrated ABCD grouped features are currently the strongest
traditional-method candidate. The no-threshold model is preferred for the main
report because it has cleaner decision logic and slightly better multi-seed
mean macro-F1. The threshold version can be presented as an error-driven
ablation.

## Final Prediction File

Generated with:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_all_abcd_grouped_lr03_grouped_raw_seed127.joblib \
  --output_csv output.csv
```

Output:

```text
output.csv
```

The file has 601 lines: one header plus 600 image predictions.

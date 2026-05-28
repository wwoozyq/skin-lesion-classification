# Experiment Ledger

This ledger summarizes the experiments already run in the project, what each
experiment tested, where the implementation lives, and how the result should be
used in the report.

## Evaluation Rule Used Throughout

Unless explicitly marked as an extension, results use:

```text
CV protocol: StratifiedGroupKFold
Group key:   base_id from image_id
Reason:      keep original images and augmentations in the same fold
Metrics:     accuracy, macro-F1, balanced accuracy, confusion matrix
```

We do not use test labels for tuning, deep features for the traditional main
line, or external medical data.

## 1. Strict Grouped-CV Baseline

Purpose:

```text
Replace image-level random split with group-aware validation.
```

Implementation:

```text
src/train_ml.py
src/utils.py::base_id_from_image_id
```

Core command:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all \
  --classifier svm \
  --cv grouped
```

Conclusion:

The apparent 90%+ image-level result was too optimistic because augmented
images leaked across train/validation folds. Under grouped CV, the baseline is
much lower but more honest.

Report role:

```text
Major methodological correction and reviewer-safety point.
```

## 2. Handcrafted Feature Grid Search

Purpose:

```text
Compare feature sets, classifiers, and feature-selection sizes under the
strict grouped protocol.
```

Implementation:

```text
experiments/run_ml_grid.py
src/features_color.py
src/features_shape.py
src/features_texture.py
src/features_contrast.py
src/features_abcd.py
src/features_boundary.py
```

Main early result:

| method | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| `all_boundary + LR + k=100` | 0.7233 | 0.7454 | 0.7535 |

Conclusion:

Boundary and contrast features were useful, but the later ABCD-grouped model
became the stable main model.

Report role:

```text
Shows systematic model/feature search instead of hand-picking one pipeline.
```

## 3. Mask Cleaning Ablation

Purpose:

```text
Test whether cleaning masks improves feature quality.
```

Implementation:

```text
src/preprocess.py
experiments/run_ml_grid.py --mask_modes clean
```

Best clean-mask result:

| method | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| `all + LR + clean mask` | 0.7217 | 0.7373 | 0.7350 |

Conclusion:

Mask cleaning is a meaningful ablation, but it did not beat the selected raw
mask model. The final main model keeps `mask_mode=raw`.

Report role:

```text
Negative result showing that preprocessing was tested rather than assumed.
```

## 4. Robustness Consistency

Purpose:

```text
Measure whether predictions remain consistent across original and augmented
versions of the same lesion.
```

Implementation:

```text
experiments/analyze_robustness.py
```

ABCD-grouped model result:

| image type | n images | accuracy |
|---|---:|---:|
| original | 200 | 0.7800 |
| augmented | 400 | 0.7500 |

| n groups | prediction consistency | all images correct | any image correct | all match original |
|---:|---:|---:|---:|---:|
| 200 | 0.7350 | 0.6300 | 0.8750 | 0.7350 |

Conclusion:

The model is reasonably robust to augmentation, but augmented images remain
harder than originals.

Report role:

```text
Directly addresses the project requirement to demonstrate robustness on
augmented images.
```

## 5. Error Visualization

Purpose:

```text
Inspect what the model still gets wrong.
```

Implementation:

```text
experiments/visualize_errors.py
```

ABCD-grouped model error distribution:

| true label | predicted label | n errors |
|---|---|---:|
| nv | mel | 59 |
| mel | nv | 53 |
| nv | vasc | 14 |
| mel | vasc | 9 |
| vasc | nv | 6 |
| vasc | mel | 3 |

Conclusion:

Most remaining errors are between `mel` and `nv`, which is expected because
both are pigmented lesions with overlapping color, texture, and boundary
patterns.

Report role:

```text
Failure analysis and visual explanation for the remaining 20%-25% error.
```

## 6. Mel/NV Error-Driven Features

Purpose:

```text
Add features specifically designed for melanoma-vs-nevus confusion.
```

Implementation:

```text
src/features_melnv.py
experiments/run_melnv_refinement.py
```

Tested ideas:

- Lab color spread
- HSV variegation
- PCA-axis asymmetry
- quadrant color imbalance
- core-border color contrast
- 3x3 local color heterogeneity
- dark/high-saturation component statistics

Representative result:

| method | macro-F1 |
|---|---:|
| `final_melnv + SVM + k=80` | 0.7314 |
| `all_boundary_melnv + LR + k=100` | 0.7240 |
| older `all_boundary + LR + k=100` | 0.7454 |

Conclusion:

The features are interpretable, but they did not improve grouped-CV
performance consistently. They are kept as diagnostic work, not adopted into
the final main model.

Report role:

```text
Error-driven optimization attempt with a clear negative conclusion.
```

## 7. Two-Stage Mel/NV Refinement

Purpose:

```text
Use a second binary classifier to refine predictions only when the main model
predicts mel or nv.
```

Best stability result:

| setup | main mean macro-F1 | refined mean macro-F1 | delta |
|---|---:|---:|---:|
| `all_boundary` LR top80 refinement | 0.7242 | 0.7241 | -0.0001 |

Conclusion:

The refinement stage over-corrects and does not improve stability. It should
not be used in the final pipeline.

Report role:

```text
Useful negative result: more complex is not automatically better.
```

## 8. ABCD Grouped Integration

Purpose:

```text
Integrate interpretable ABCD-style grouped features and retest the main model.
```

Implementation:

```text
src/features_abcd_grouped.py
experiments/run_abcd_grouped_integration.py
src/train_ml.py --classifier lr03
```

Best seed `127` result:

| model | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| older `all_boundary + LR + k=100` | 0.7233 | 0.7454 | 0.7535 |
| `all_abcd_grouped + LR03 + k=140` | 0.7600 | 0.7715 | 0.7871 |
| threshold ablation `mel_threshold=0.45` | 0.7600 | 0.7730 | 0.7910 |

Five-seed stability:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| older `all_boundary + LR + k=100` | 0.7077 | 0.7242 | 0.0143 | 0.7301 |
| `all_abcd_grouped + LR03 + k=140` | 0.7283 | 0.7435 | 0.0271 | 0.7512 |

Conclusion:

This is the selected stable traditional main model. The threshold ablation is
interesting, but the no-threshold version is cleaner and easier to defend.

Report role:

```text
Final official traditional-machine-learning method.
```

## 9. XGBoost Cascade

Purpose:

```text
Use a clinically motivated two-stage structure:
Stage 1 separates vasc, Stage 2 separates mel and nv.
```

Implementation:

```text
experiments/run_xgb_cascade_search.py
src/train_ml_cascade.py
run.py cascade support
```

Design:

```text
P(vasc) = Stage 1 probability
P(mel)  = (1 - P(vasc)) * Stage 2 P(mel | non-vasc)
P(nv)   = (1 - P(vasc)) * Stage 2 P(nv  | non-vasc)
```

Results:

| protocol | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| integrated seed `127` smoke | 0.7617 | 0.7802 | 0.8017 |
| reproduced teammate branch seed `127` | 0.7600 | 0.7792 | 0.8006 |
| best five-seed bagged | 0.7500 | 0.7655 | 0.7887 |

Conclusion:

The cascade is a strong exploratory traditional model and a good presentation
highlight, but it is more complex than the stable main LR model.

Report role:

```text
Advanced traditional ML extension.
```

## 10. Early Feature Fusion

Purpose:

```text
Test front-end feature fusion by concatenating many handcrafted features into
one large vector before training one classifier.
```

Implementation:

```text
experiments/run_early_fusion_search.py
docs/EARLY_FUSION_RESULTS.md
```

Best result:

| protocol | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| seed `127`, OOF-tuned class weights | 0.7700 | 0.7830 | 0.7997 |
| best five-seed mean | 0.7450 | 0.7618 | 0.7766 |

Conclusion:

Early fusion nearly reaches `0.80` balanced accuracy on one split, but it is
not clearly more stable than late fusion or the cascade.

Report role:

```text
Feature-level fusion ablation.
```

## 11. Late Probability Fusion

Purpose:

```text
Preserve the original main model and the cascade model, then fuse their
probabilities.
```

Implementation:

```text
experiments/run_fusion_ensemble.py
src/train_ml_fusion.py
run.py fusion support
docs/FUSION_ENSEMBLE_RESULTS.md
```

Formula:

```text
P_final = 0.5 * P_original + 0.5 * P_cascade
```

Results:

| protocol | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| seed `127` | 0.7717 | 0.7909 | 0.8055 |
| five-seed bagged | 0.7417 | 0.7607 | 0.7763 |

Conclusion:

Late fusion is the best single-seed result, but the five-seed check shows it
should be reported as exploratory rather than final.

Report role:

```text
Model-level fusion highlight and honest stability discussion.
```

## 12. TTA + Nested-CV Fusion Weight

Purpose:

```text
Attack the fusion-weight in-sample overfit and the augmentation-sensitive
error bucket at the same time. Honest nested-CV picks the fusion weight on
inner folds; six-transform TTA averages probabilities on the outer val fold.
```

Implementation:

```text
experiments/run_tta_nested_fusion.py   (branch tta-nested-fusion)
```

Design:

```text
Outer: 5 seeds × 5 StratifiedGroupKFold folds (grouped by base_id)
Inner: 3-fold grouped CV on the outer train set
w     : argmax over {0, 0.05, ..., 1.0} of inner-OOF macro-F1
TTA   : {identity, hflip, vflip, rot90, rot180, rot270}, mean of softmax
```

Four-cell ablation, five-seed bagged:

| cell | TTA | tune w | accuracy | macro-F1 | balanced accuracy |
|---|---|---|---:|---:|---:|
| A | no | fixed `0.5` | 0.7417 | 0.7607 | 0.7763 |
| B | yes | fixed `0.5` | 0.7433 | 0.7610 | 0.7753 |
| C | no | nested | 0.7383 | 0.7571 | 0.7715 |
| D | yes | nested | 0.7400 | 0.7584 | 0.7726 |

Cell `A` reproduces section 11 bagged numbers exactly, confirming the
pipeline. Cells `B C D` move within `0.005` balanced accuracy of `A`.

Mechanism check on seed `127`, cell `D` against the LR main model:

| error bucket | LR main | cell D |
|---|---:|---:|
| total | 144 | 138 |
| augmentation-sensitive | 69 | 57 |
| lesion-hard | 75 | 81 |
| all-augs-wrong lesions | 25 | 27 |

Nested-CV honestly selects weight mean `0.562 ± 0.143`, not `0.5`, which
confirms that the single-seed `0.8055` peak in section 11 was in-sample
noise rather than an undertuned weight.

Conclusion:

The mechanism works directionally on a single seed — TTA does shrink the
augmentation-sensitive error bucket by about 17%. But five-seed bagging
absorbs that gain, because bagging already averages out the same
augmentation-direction variance. The five-seed bagged numbers do not beat
section 11, so this is kept as a negative result on branch
`tta-nested-fusion` and is not merged into `main`.

Report role:

```text
Negative result confirming that the late-fusion ceiling on this dataset is
the bagged 0.7763 BalAcc number from section 11, not the single-seed 0.8055.
```

## 13. Deep Learning Extension

Purpose:

```text
Show a modern extension separate from the official traditional ML line.
```

Implementation:

```text
experiments/train_deep.py
experiments/run_deep_lowload_night.py
docs/DEEP_LEARNING_EXTENSION_RESULTS.md
docs/KAGGLE_DEEP_LEARNING.md
```

Current best extension result:

| model | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| MobileNetV2, ImageNet pretrained, crop, TTA | 0.8750 | 0.8623 | 0.8533 |

Conclusion:

Deep learning works well, but it is not used as the official quantitative
traditional-method submission.

Report role:

```text
Optional extension and comparison against traditional image processing.
```

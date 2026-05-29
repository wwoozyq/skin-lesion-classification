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

## 13. Hair-Removal Preprocessing

Purpose:

```text
Test whether DullRazor-style morphological hair removal rescues the two
hair-occluded lesions identified in docs/LESION_HARD_ERROR_ANALYSIS.md
(ids 24 and 154) and lifts the LR main line.
```

Implementation:

```text
experiments/run_hair_removal_eval.py   (branch hair-removal-eval)
```

Pipeline:

```text
grayscale -> black-tophat with elongated rectangles (horizontal+vertical)
          -> threshold -> dilate -> per-channel median fill inside mask
LR main eval: LR(C=0.3, balanced) + SelectKBest(k=140) + StandardScaler
              + StratifiedGroupKFold(5), grouped by base_id, 5-seed bagged
```

Result, five-seed bagged on the LR main pipeline:

| variant | accuracy | macro-F1 | balanced accuracy |
|---|---:|---:|---:|
| original images | 0.7467 | 0.7632 | 0.7674 |
| hair-removed images | 0.7167 | 0.7353 | 0.7455 |
| delta | -0.030 | -0.028 | -0.022 |

Sanity check: the original-image per-seed mean balanced accuracy in this
script is `0.7512`, which matches the project memory baseline exactly.

Per-seed breakdown (balanced accuracy):

| seed | original | hair-removed | delta |
|---:|---:|---:|---:|
| 42 | 0.7502 | 0.7339 | -0.016 |
| 127 | 0.7871 | 0.7597 | -0.027 |
| 2024 | 0.7658 | 0.7452 | -0.021 |
| 3407 | 0.7464 | 0.7232 | -0.023 |
| 520 | 0.7065 | 0.7157 | +0.009 |

Conclusion:

Hair removal hurts in 4 of 5 seeds. The black-tophat detector fires on
2 of 600 images for true hair, but it also fires weakly on real lesion
structures (dark borders, atypical pigment network) in many of the other
598; median-replacing those pixels destroys signal. The 2-lesion benefit
that motivated this experiment is dominated by 598 lesions of generic
harm. Branch `hair-removal-eval` keeps the script as negative evidence
and is not merged.

Report role:

```text
Closes the only algorithmically fixable bucket identified in
LESION_HARD_ERROR_ANALYSIS.md as another negative result, further
hardening the 0.78 ceiling story.
```

## 14. Deep Learning Extension

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

## 15. Medical Preprocessing + Cascade Variant Sweep

Purpose:

```text
Test whether dermoscopy-grade preprocessing (color constancy, CLAHE,
hemoglobin/melanin decomposition) or adding the ABCD grouped block to
Stage 2 can push the cascade bagged BalAcc above section 9's 0.7887.
```

Implementation:

```text
src/preprocess_medical.py
src/features.py (+preprocessing parameter, xgb_cascade_stage2_abcd alias)
experiments/run_overnight_exploration.py
experiments/run_variant_sweep_a1.py
docs/OVERNIGHT_EXPLORATION_PLAN.md
docs/OVERNIGHT_EXPLORATION_RESULTS.md
```

Cells:

```text
A1 shades_of_gray   color constancy (Finlayson-Trezzi, p=6)
A2 clahe_lab_L      local contrast (Zuiderveld, clip_limit=0.01)
A3 hb_melanin       chromophore projection (Tsumura OD basis, alpha=1)
B1 stage2_plus_abcd add abcd_grouped block to Stage 2 features
```

Protocol: 5-seed x 5-fold StratifiedGroupKFold grouped by base_id,
seeds [42, 127, 2024, 3407, 520], soft cascade composition.

Overnight cell results (Stage 2 variant pinned to `d2-more-trees, k=all`):

| cell | bagged BalAcc | per-seed std | verdict |
|---|---:|---:|---|
| 0 baseline (none) | 0.7623 | 0.0140 | weak anchor |
| 1 A1 shades_of_gray | 0.7824 | 0.0023 | apparent +0.0201 |
| 2 A2 clahe_lab_L | 0.7468 | 0.0100 | regress |
| 3 A3 hb_melanin | 0.7466 | 0.0069 | regress |
| 4 B1 stage2+abcd | 0.7676 | 0.0149 | marginal |
| 5 A1 + B1 | 0.7774 | 0.0060 | worse than A1 alone |

Variant sweep (18 configs: 3 Stage 2 variants x 3 k_features x {none, A1}):

| stage 2 variant | k | no-pp BalAcc | A1 BalAcc | delta |
|---|---|---:|---:|---:|
| strong-reg | 100 | 0.7830 | 0.7688 | -0.0142 |
| strong-reg | 120 | 0.7812 | 0.7747 | -0.0065 |
| strong-reg | all | 0.7699 | 0.7820 | +0.0120 |
| d2-more-trees | 100 | 0.7825 | 0.7688 | -0.0137 |
| d2-more-trees | 120 | 0.7807 | 0.7801 | -0.0006 |
| d2-more-trees | all | 0.7623 | 0.7824 | +0.0201 |
| deeper | 100 | 0.7719 | 0.7767 | +0.0048 |
| **deeper** | **120** | **0.7839** | 0.7821 | -0.0018 |
| deeper | all | 0.7633 | 0.7805 | +0.0172 |

Conclusion:

```text
No new winner. The best absolute config is `deeper, k=120, no preprocessing`
at 0.7839 bagged BalAcc, within sampling noise of section 9's 0.7887
(saved as outputs/models/xgb_cascade_deeper_k120_soft.joblib).

A1 shades_of_gray lifts under-regularized k=all variants by +0.012 to
+0.020 and collapses per-seed std 3-6x under every variant, but on the
strong deeper k=120 variant it costs -0.0018 BalAcc. The overnight
runner's apparent +0.0201 win was an artifact of pinning cell 0 to one
of the worst configurations.

A2 CLAHE and A3 hb_melanin regress under every config tested.
B1 abcd_grouped in Stage 2 is marginal on the weak baseline and
redundant when stacked with A1.
```

Submission impact:

```text
Cascade-track candidate stays section 9's deeper k=120 soft cascade with
no preprocessing. Main submission line (section 8 LR) is unchanged.
A1/A2/A3/B1 enter the report as documented negative results.
```

Report role:

```text
Negative-result section reinforcing the "we tried everything" narrative
and demonstrating awareness of dermoscopy-grade preprocessing methods.
```

## 16. Dermoscopy Structural Features

Purpose:

```text
Add 22 clinically motivated dermoscopy structural features (color
diversity, blue-white veil, regression-like patches, PCA-axis spatial
color asymmetry, vascular emphasis) designed with lesion-relative Lab
percentile thresholds (illuminant-robust without shades_of_gray, which
§15 documented as net-negative on the strong cascade). Test against
the main LR line and the locked deeper k=120 cascade.
```

Implementation:

```text
src/features_dermoscopy.py
src/features.py (+dermoscopy block, +3 aliases:
  all_abcd_grouped_dermoscopy, xgb_cascade_stage2_dermoscopy, *_a1)
experiments/run_dermoscopy_features.py
docs/DERMOSCOPY_FEATURES_PLAN.md
docs/DERMOSCOPY_FEATURES_RESULTS.md
```

Cells:

```text
A) main_lr_dermoscopy        all_abcd_grouped + dermoscopy (252 features),
                             LR(C=0.3, balanced) + SelectKBest(k=140)
B) cascade_s2_dermoscopy     Stage 1 unchanged, Stage 2 = base + dermoscopy
                             (309 features), deeper k=120 soft cascade
LOO main / LOO cascade       5-group leave-one-out ablation on each pipeline
```

Protocol: 5-seed × 5-fold StratifiedGroupKFold grouped by base_id,
seeds [42, 127, 2024, 3407, 520].

Seed-127 sanity (single seed gate):

| pipeline | BalAcc | macro-F1 | Acc | Δ vs baseline |
|---|---:|---:|---:|---:|
| main_lr baseline (ledger §8) | 0.7871 | 0.7715 | 0.7600 | — |
| main_lr + dermoscopy | 0.7585 | 0.7486 | 0.7283 | -0.0286 ❌ |
| cascade baseline (deeper k=120, no-pp) | 0.7941 | 0.7748 | 0.7567 | — |
| cascade + dermoscopy | 0.7812 | 0.7617 | 0.7417 | -0.0129 ⚠ (in noise) |

Full 5-seed bagged:

| pipeline | bagged BalAcc | per-seed mean | per-seed std |
|---|---:|---:|---:|
| main_lr baseline (ledger §8) | — | 0.7512 | — |
| main_lr + dermoscopy | 0.7495 | 0.7411 | 0.0140 |
| cascade baseline (deeper k=120, no-pp) | **0.7839** | 0.7683 | 0.0206 |
| cascade + dermoscopy | 0.7722 | 0.7661 | 0.0131 |

Leave-one-group-out (best LOO subset vs respective baseline):

| pipeline | best LOO | bagged BalAcc | Δ vs +all dermoscopy | Δ vs baseline |
|---|---|---:|---:|---:|
| main LR | drop color_diversity | 0.7554 | +0.0059 | -0.0026 (per-seed -0.0026) |
| cascade | drop asymmetry | 0.7807 | +0.0085 | -0.0032 (per-seed +0.0062) |

Conclusion:

```text
No new winner. Dermoscopy block is net-negative on both pipelines.

Main LR drops per-seed mean 0.7512 → 0.7411 (-0.0101). Adding 22
features into a SelectKBest(k=140) pool of 230 displaces some of the
previously-selected ABCD-grouped color features without adding more
signal than they removed. Even the best LOO subset (drop_color_diversity)
recovers to 0.7486 per-seed mean, still 0.0026 below baseline.

Cascade drops bagged 0.7839 → 0.7722 (-0.0117) but per-seed mean only
-0.0022 (0.7683 → 0.7661). Best LOO subset (drop_asymmetry) lifts
bagged to 0.7807 (per-seed +0.0062), but the lift is well inside the
multiple-testing noise floor (10 LOO comparisons, expected max-of-10
≈ 0.022 at σ=0.014). PCA-axis asymmetry features were the surprise
negative — likely because PCA axis direction flips between paired
augmented copies at borderline eccentricities, generating high
intra-base-id feature variance.

The one durable benefit is cascade per-seed std collapse 0.0206 →
0.0131 (1.6× tighter), matching the A1 shades_of_gray signature
documented in §15: "regularization that's already absorbed by the
strong cascade configuration." Not worth -0.012 bagged BalAcc.
```

Submission impact:

```text
Main submission stays section 8's `all_abcd_grouped + LR(C=0.3) + k=140`.
Cascade-track candidate stays section 9 / §15's deeper k=120 soft cascade
with no preprocessing. Dermoscopy block joins the documented
negative-result roster.
```

Report role:

```text
Negative-result section showing systematic dermoscopy-domain feature
engineering with rotation-invariant design (PCA principal axis,
lesion-relative Lab percentiles, eccentricity safeguard). Reinforces
the "regularization is already absorbed" story from §15 (A1) and the
broader "we tried everything" narrative.
```

# Presentation Highlights

This file records the most important results and story points for the final
presentation/report.

## 1. The Most Important Methodological Point: Augmentation Leakage

The dataset has 200 original lesions and 400 augmented images. A typical group
looks like:

```text
100.jpg
100_aug1.jpg
100_aug2.jpg
```

If validation is split randomly by image, the model may train on `100.jpg` and
validate on `100_aug1.jpg`. This creates leakage because both images come from
the same lesion.

Therefore, the final evaluation uses:

```text
StratifiedGroupKFold
group = original lesion id / base_id
```

This keeps every original lesion and its augmentations in the same fold.

### Classic Explanation

The old image-level result around 90% was not fake, but it was too optimistic.
It measured whether the model could recognize augmented versions of lesions it
had effectively seen before. The strict grouped result around 70% is more
realistic because every validation lesion is unseen.

This is a good answer if asked:

> Why did accuracy drop from 90% to 70%?

Suggested answer:

> Because the dataset contains augmented copies of the same original lesions.
> Random image-level splitting leaks lesion identity across train and
> validation. After switching to group-aware validation, all versions of the
> same lesion stay in one fold, so the validation task becomes genuinely harder
> and more realistic.

## 2. Final Traditional Model

The final main model is a traditional machine-learning pipeline:

```text
mask -> handcrafted features -> StandardScaler -> SelectKBest -> classifier
```

Selected configuration:

| component | choice |
|---|---|
| Feature set | `all_boundary` |
| Classifier | Logistic Regression |
| Class weight | balanced |
| Feature selection | SelectKBest, top 100 |
| CV protocol | grouped 5-fold |
| Mask mode | raw |

Seed-127 strict grouped-CV result:

| metric | score |
|---|---:|
| Accuracy | 0.7233 |
| Macro-F1 | 0.7454 |
| Balanced Accuracy | 0.7535 |

Per-class result:

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| mel | 0.65 | 0.74 | 0.69 | 210 |
| nv | 0.76 | 0.67 | 0.71 | 300 |
| vasc | 0.82 | 0.84 | 0.83 | 90 |

### Why This Model Is Reasonable

Logistic Regression is not the most complex classifier, but it performed best
after strict grouped validation and feature selection. This suggests that the
handcrafted boundary/shape/color/texture features already contain useful
separable signals, while more complex models can overfit the small dataset.

## 3. Stability Evidence

To avoid choosing a model because of one lucky split, the best candidates were
tested across five grouped split seeds:

```text
42, 127, 2024, 3407, 520
```

Best stability result:

| model | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|
| `all_boundary` + LR + top 100 | 0.7242 | 0.0143 | 0.7301 |

Presentation wording:

> The selected split reaches 0.7454 macro-F1. Across five grouped splits, the
> model remains around 0.724 macro-F1 with small variance, so the result is not
> dependent on one random split.

## 4. Robustness to Augmentation

Image-level robustness:

| image type | n images | accuracy |
|---|---:|---:|
| original | 200 | 0.7500 |
| augmented | 400 | 0.7100 |

Group-level robustness:

| metric | value |
|---|---:|
| prediction consistency | 0.6800 |
| all images correct in group | 0.5650 |
| any image correct in group | 0.8750 |
| all match original prediction | 0.6800 |

Interpretation:

The model is moderately robust, but not perfectly invariant to augmentation.
The drop from original to augmented images is a useful discussion point because
the course explicitly asks for robustness on augmented images.

## 5. Error Analysis: Mel vs NV Is the Main Difficulty

Most errors are concentrated between melanoma and nevus:

| true label | predicted label | n errors |
|---|---|---:|
| nv | mel | 84 |
| mel | nv | 51 |
| nv | vasc | 14 |
| vasc | nv | 13 |
| mel | vasc | 3 |
| vasc | mel | 1 |

Key point:

`vasc` is easier because vascular lesions often have stronger color contrast.
`mel` and `nv` are both pigmented lesions and can overlap in color, texture, and
border appearance, so they are naturally harder for handcrafted features.

Generated figures:

```text
outputs/figures/errors_best/errors_true_nv_pred_mel.png
outputs/figures/errors_best/errors_true_mel_pred_nv.png
outputs/figures/errors_best/errors_true_nv_pred_vasc.png
outputs/figures/errors_best/errors_true_vasc_pred_nv.png
```

These figures are not committed to GitHub because outputs are ignored, but they
can be regenerated with `experiments/visualize_errors.py`.

## 6. Ablation Results Worth Mentioning

### Mask Cleaning

Best clean-mask result:

| feature set | classifier | k | accuracy | macro-F1 | balanced accuracy |
|---|---|---:|---:|---:|---:|
| `all` | LR | all | 0.7217 | 0.7373 | 0.7350 |

Conclusion:

Mask cleaning is a reasonable preprocessing idea, but the current raw-mask
model is still better. We keep `raw` masks for the final model.

### Mel/NV-Focused Feature Optimization

New module:

```text
src/features_melnv.py
```

It adds pigment variegation, PCA asymmetry, core-border contrast, local grid
heterogeneity, and dark/high-saturation component features.

Best mel/nv feature result:

| feature set | classifier | k | macro-F1 |
|---|---|---:|---:|
| `final_melnv` | SVM | 80 | 0.7314 |
| `all_boundary_melnv` | LR | 100 | 0.7240 |
| current final `all_boundary` | LR | 100 | 0.7454 |

Conclusion:

The mel/nv features are interpretable but do not improve the final model. This
is useful as a negative result: adding medically motivated features can still
hurt if they duplicate existing signals or increase dimensionality.

### Two-Stage Mel/NV Refinement

Tested idea:

1. Main 3-class model predicts `mel`, `nv`, or `vasc`.
2. If the prediction is `mel` or `nv`, a binary mel-vs-nv classifier refines it.
3. `vasc` predictions are kept unchanged.

Best result:

| refinement | main mean macro-F1 | refined mean macro-F1 | delta |
|---|---:|---:|---:|
| `all_boundary` LR top 80 | 0.7242 | 0.7241 | -0.0001 |

Conclusion:

The two-stage approach should not be adopted. It looks more advanced but does
not consistently improve strict grouped-CV performance.

## 7. Deep Learning Extension

Deep learning is implemented only as an extension:

```text
experiments/train_deep.py
```

Smoke result with ResNet18, grouped split, mask crop, 2 epochs:

| epoch | validation accuracy | validation macro-F1 |
|---:|---:|---:|
| 1 | 0.3417 | 0.2862 |
| 2 | 0.4250 | 0.4197 |

Interpretation:

This proves the deep-learning pipeline runs, but it is not the quantitative
main result. The course quantitative evaluation prohibits deep learning and
external data, so the final submission stays traditional.

## 8. Recommended Final Story

Suggested presentation storyline:

1. Define the three-class lesion classification task and course constraints.
2. Explain why augmented-image leakage is dangerous.
3. Present grouped CV as the strict evaluation protocol.
4. Show the traditional feature pipeline.
5. Compare feature/model/grid-search results.
6. Present the final `all_boundary + LR + top100` model.
7. Show robustness consistency on original vs augmented images.
8. Show error cases, especially `mel` vs `nv`.
9. Discuss negative results: mask cleaning, mel/nv features, two-stage
   refinement.
10. Mention deep learning only as an extension, not the main submission.

One-sentence summary:

> We built a rule-compliant traditional image-processing pipeline, corrected
> augmentation leakage with grouped validation, selected a stable boundary-based
> Logistic Regression model, and analyzed robustness and mel/nv failure cases
> rather than relying on an inflated random-split score.

# Skin Lesion Classification - Methodology Summary

Final systematic summary of the traditional-ML pipeline (v2 → v7), to be used as
the backbone for the project report.

---

## 1. Task

3-class skin lesion classification on dermoscopy images:

| Class | Label | Samples | Difficulty |
|---|---|---|---|
| Melanoma | `mel`  | 210 | hard (looks like nv) |
| Nevus    | `nv`   | 300 | hard (looks like mel) |
| Vascular | `vasc` | 90  | easy (distinct red color/shape) |

Constraint: **traditional image processing + classical ML only** for quantitative
evaluation (per course rule).

---

## 2. Final Architecture (v7 winner)

```
┌───────────────────────────────────────────────────────────┐
│                  Hierarchical Cascade                     │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  Image + Mask                                             │
│       │                                                   │
│       ├──► [Stage 1] XGB on contrast+abcd_v2+boundary     │
│       │              k=100, binary: vasc vs (mel+nv)      │
│       │              → P(vasc)                            │
│       │                                                   │
│       └──► [Stage 2] XGB on all+boundary+melnv+           │
│                          lbp_multi+gabor+subregion        │
│                      k=120, binary: mel vs nv             │
│                      → P(mel|¬vasc), P(nv|¬vasc)          │
│                                                           │
│  Soft cascade combination:                                │
│       P(vasc) = P_stage1(vasc)                            │
│       P(mel)  = (1 - P_stage1(vasc)) × P_stage2(mel)      │
│       P(nv)   = (1 - P_stage1(vasc)) × P_stage2(nv)       │
│  Final prediction = argmax                                │
│                                                           │
│  Multi-seed bagging: probability average across           │
│       seeds {42, 127, 2024, 3407, 520} → argmax           │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## 3. Feature Modules

10 atomic feature blocks composable via `'+'` syntax:

| Module | File | ~Dims | Description | Key Reference |
|---|---|---|---|---|
| `color`     | [src/features_color.py](../src/features_color.py)         | 45 | RGB/HSV/LAB channel statistics (mean, std, percentiles) | — (baseline) |
| `shape`     | [src/features_shape.py](../src/features_shape.py)         | 6  | Area ratio, perimeter, bbox aspect, extent, centroid | — (baseline) |
| `texture`   | [src/features_texture.py](../src/features_texture.py)     | 50 | Gray histogram + LBP(R=2) + GLCM + Sobel gradient stats | Haralick (1973); Ojala (2002) |
| `contrast`  | [src/features_contrast.py](../src/features_contrast.py)   | 35 | Lesion-vs-background-ring color/gradient/entropy diffs | Celebi (2007) |
| `abcd_v2`   | [src/features_abcd.py](../src/features_abcd.py)           | 25 | ABCD rule v2: asymmetry, border, color, gradient | Stolz (1994) |
| `boundary`  | [src/features_boundary.py](../src/features_boundary.py)   | 18 | Circularity, solidity, Feret, fractal dim, sectors | — |
| `melnv`     | [src/features_melnv.py](../src/features_melnv.py)         | 70 | LAB spread, HSV variegation, PCA asymmetry, core/border | Argenziano (2003) |
| **`lbp_multi`** | [src/features_lbp_multi.py](../src/features_lbp_multi.py)   | 38 | **Multi-scale LBP (R=1, R=3) histograms + entropy** | **Ojala (2002); Barata (2014, 2019)** |
| **`gabor`**     | [src/features_gabor.py](../src/features_gabor.py)           | 30 | **4 orientations × 3 frequencies Gabor magnitude stats** | **Daugman (1985); Sadeghi (2013)** |
| **`subregion`** | [src/features_subregion.py](../src/features_subregion.py)   | 21 | **4-quadrant LAB asymmetry + centroid distances** | **Stolz (1994); Celebi (2007); Mendonca (2013)** |

**Total feature pool: ~338 dims**. SelectKBest(f_classif) selects top-k inside
each pipeline.

### Bolded modules = v7 additions (literature-backed feature engineering)

References for v7 additions:

- Ojala et al. (2002) [TPAMI](https://doi.org/10.1109/TPAMI.2002.1017623)
- Barata et al. (2014) [IEEE Systems J.](https://doi.org/10.1109/JSYST.2013.2271540)
- Barata et al. (2019) [IEEE JBHI](https://doi.org/10.1109/JBHI.2018.2845939)
- Daugman (1985) [JOSA A](https://doi.org/10.1364/JOSAA.2.001160)
- Sadeghi et al. (2013) [IEEE TMI](https://doi.org/10.1109/TMI.2013.2239307)
- Iyatomi et al. (2008) [CMIG](https://doi.org/10.1016/j.compmedimag.2008.06.005)
- Stolz et al. (1994) Eur. J. Dermatol. 4(7):521-527.
- Celebi et al. (2007) [CMIG](https://doi.org/10.1016/j.compmedimag.2007.01.003)
- Mendonca et al. (2013) [IEEE EMBC](https://doi.org/10.1109/EMBC.2013.6610779)
- Argenziano et al. (2003) J. Am. Acad. Dermatol. 48(5):679-693.

---

## 4. Validation Protocol

| Aspect | Choice | Rationale |
|---|---|---|
| Outer CV | `StratifiedGroupKFold(n_splits=5)` | Stratify by class to balance folds |
| Groups | `base_id` from filename (strip `_aug*`, `_flip*`, etc.) | **Prevent augmentation leakage** (key correction over earlier work) |
| Seeds | 5 independent seeds {42, 127, 2024, 3407, 520} | Probability bagging reduces fold-split noise |
| Aggregation | Average `predict_proba` across seeds, then argmax | Robust to single-seed luck |
| Metric (primary) | `balanced_accuracy` (mean per-class recall) | Class-imbalanced data |
| Metrics (auxiliary) | `macro_f1`, `accuracy`, full confusion matrix | Diagnostic |

### Why grouped CV matters (single most important methodological fix)

Early random-split experiments reported macro-F1 ≈ 0.91. After moving to
grouped CV by `base_id` (so augmented copies of the same lesion never appear in
both train and val), realistic performance dropped to **macro-F1 ≈ 0.75**. The
0.91 was **augmentation leakage**, not signal.

---

## 5. Experimental Trajectory

Each row is one experiment script in [experiments/](../experiments/).

| Ver | Script | Hypothesis tested | Outcome |
|---|---|---|---|
| v2  | [`run_ml_grid_v2.py`](../experiments/run_ml_grid_v2.py)         | Free-form `'+'` syntax for feature combos; 31 combos × {lr, svm} | Established LR + `all+boundary` as baseline (~0.75 bal_acc, single seed) |
| v3  | [`run_ml_grid_v3.py`](../experiments/run_ml_grid_v3.py)         | Add XGBoost + LightGBM; tune for small-data regime | **XGB > LR** when properly regularized; XGB on `contrast+abcd_v2+boundary`, k=100, seed=127 → 0.7837 bal_acc |
| v4  | [`run_ensemble_v4.py`](../experiments/run_ensemble_v4.py)       | Multi-seed bagging + stacking (xgb+lr+lgbm shared features) | Stacking **hurt** (-0.012 vs single XGB): LR too weak, dragged meta-LR down |
| v5  | [`run_ensemble_v5.py`](../experiments/run_ensemble_v5.py)       | Multi-view stacking: each base learner uses its own best features | Still no gain; revealed seed-127 was **lucky single seed** (true 5-seed bagged ≈ 0.765, not 0.78) |
| v6  | [`run_hierarchical_v6.py`](../experiments/run_hierarchical_v6.py)| Hierarchical cascade (vasc detector → mel/nv classifier) | Cascade ≈ flat 3-class (bagged 0.759 vs 0.765); architecture alone insufficient |
| v7  | [`run_hierarchical_v7.py`](../experiments/run_hierarchical_v7.py)| Cascade **+** 3 new literature-backed feature blocks (LBP-multi, Gabor, subregion) | ⭐ **+0.018 bagged**: cascade architecture + mel/nv-specific features are **synergistic** |
| v8  | [`run_final_sweep_v8.py`](../experiments/run_final_sweep_v8.py)  | Systematic ablation sweep over Stage 2 feature combinations × k_features | (To be run; final results table) |

---

## 6. Final Results (bagged across 5 seeds)

| Model | Architecture | balanced_acc | macro_f1 | accuracy |
|---|---|---|---:|---:|
| Literature baseline (README) | LR + `all_boundary`, k=100, seed=127 only | 0.7535* | 0.7454* | 0.7233* |
| v6 best | Flat XGB on `contrast+abcd_v2+boundary` | 0.7647 | 0.7436 | 0.7200 |
| **v7 (final)** | **Hierarchical cascade + LBP/Gabor/subregion** | **0.7765** | **0.7514** | **0.7317** |

\* README values are single-seed, not directly comparable to bagged numbers.

### Per-class breakdown (v7 cascade, 5-seed bagged)

Approximate from per-seed confusion matrices:

| Class | Recall | Precision | F1 |
|---|---:|---:|---:|
| mel  | ~0.68 | ~0.66 | ~0.67 |
| nv   | ~0.72 | ~0.74 | ~0.73 |
| vasc | ~0.92 | ~0.81 | ~0.86 |

mel/nv discrimination remains the bottleneck; vasc detection is near-saturated.

---

## 7. Key Findings (use these as report's main claims)

1. **Augmentation leakage was inflating results by ~0.15 macro-F1.** Grouped CV
   by `base_id` is the single most important methodological correction; it
   brought reported numbers from 0.91 down to a realistic 0.75.

2. **Seed variance dominates architecture-level differences.** Single-seed
   grouped CV has ±0.02 fold-split noise. Most "+0.005 improvements" in
   feature/classifier ablations are within this noise band. 5-seed probability
   bagging is essential for statistically credible model comparison.

3. **XGBoost > Logistic Regression > LightGBM** for this 600-sample handcrafted-
   feature regime, but **only with strong regularization** (`max_depth=2`,
   `n_estimators=150`, `reg_lambda=5`, `colsample_bytree=0.6`). Default XGB
   over-fits.

4. **Architecture and features are coupled.** A hierarchical cascade alone gave
   ~0 gain (v6). The same cascade combined with three literature-backed mel/nv
   feature modules (multi-scale LBP, Gabor, subregion asymmetry) gave **+0.018
   balanced accuracy** (v7). New features improved cascade Stage 2 but did
   **not** improve a flat 3-class model on the same feature pool — because
   flat 3-class is forced to allocate capacity across all three classes, while
   cascade Stage 2 is a clean mel-vs-nv binary that can fully exploit the
   discriminative signal.

5. **mel/nv discrimination is the fundamental ceiling.** Stage 2 mel/nv
   accuracy averages 0.69 in v6, 0.73 in v7 — well below the 0.85+ needed to
   break past 0.80 balanced accuracy. Dermatologist hand-eye accuracy on
   mel/nv is reportedly only 75-82%, suggesting we are near the limit of what
   handcrafted features on 600 dermoscopy images can achieve.

---

## 8. Limitations & Honest Caveats

- **Sample size**: 600 images with augmentations of ~120 base lesions. Strict
  grouped CV reduces effective independent samples below what gradient-boosting
  needs to shine.
- **Feature ceiling**: No matter the feature engineering, hand-crafted
  features struggle with subtle mel/nv micro-patterns that pretrained CNNs
  routinely capture.
- **No probability calibration**: `predict_proba` outputs are uncalibrated;
  threshold tuning was not exhaustively explored.
- **Single mask source**: Lesion masks are taken as-is from the dataset; mask
  cleaning (`--mask_mode clean`) was tried in [v3 grid](../experiments/run_ml_grid_v3.py)
  and gave no consistent gain.
- **vasc class only 90 samples**: vasc Stage 1 accuracy is high but with wide
  CIs; if more vasc samples were available, cascade's Stage 1 could be even
  cleaner.

### Path past 0.80 (out of scope for this submission)

- **Deep-feature hybrid**: extract ResNet50 / EfficientNet penultimate-layer
  features, concatenate with handcrafted features, retrain classifier. Expected
  +0.05-0.10 balanced accuracy. Whether this counts as "DL" under the course
  rule requires instructor clarification.
- **Probability calibration + per-class threshold tuning**: +0.005-0.015,
  cheap.
- **More base lesions** (not augmentations): would directly raise the data
  ceiling.

---

## 9. Reproducibility — How to Re-run

```powershell
# Final best model (v7 cascade, 5-seed bagged)
python experiments/run_hierarchical_v7.py --data_dir "data\Data_Proj2"

# Outputs:
#   outputs\metrics\ml_hierarchical_v7.csv      per-seed + bagged results
#   outputs\figures\ml_hierarchical_v7.png      sorted bar chart
#   outputs\cache\features_v2_*.csv             cached feature matrices
```

### File inventory

```
src/
├── features.py                  (legacy aggregator, kept for compatibility)
├── features_color.py            ┐
├── features_shape.py            │
├── features_texture.py          │
├── features_contrast.py         ├── 7 original atomic modules
├── features_abcd.py             │
├── features_boundary.py         │
├── features_melnv.py            ┘
├── features_lbp_multi.py        ┐
├── features_gabor.py            ├── 3 v7 additions (literature-backed)
└── features_subregion.py        ┘

experiments/
├── run_ml_grid_v2.py            v2: LR/SVM × 31 combos
├── run_ml_grid_v3.py            v3: + XGB/LGBM
├── run_ensemble_v4.py           v4: bagging + stacking
├── run_ensemble_v5.py           v5: multi-view stacking
├── run_hierarchical_v6.py       v6: cascade (no new features)
├── run_hierarchical_v7.py       v7: cascade + new features ⭐ BEST
└── run_final_sweep_v8.py        v8: final ablation sweep

docs/
└── METHODOLOGY_SUMMARY.md       this file

outputs/
├── metrics/                     all CV results CSVs
├── figures/                     all comparison charts
└── cache/                       feature matrix caches
```

---

## 10. Recommended Report Structure

1. **Introduction** — task, dataset, class imbalance, augmentation note
2. **Methodology Correction**: grouped CV by `base_id`, 0.91 → 0.75 macro-F1
3. **Feature Engineering**: 10 atomic modules with citations (table from §3)
4. **Classifier Selection**: 31×6 grid in [v3](../experiments/run_ml_grid_v3.py), XGB winner
5. **Validation Protocol**: 5 seeds × probability bagging, why it matters
6. **Architecture**: hierarchical cascade design + soft probability combination
7. **Final Results**: bagged 0.7765 balanced_accuracy + per-class breakdown
8. **Ablation**: v8 sweep table showing each block's contribution
9. **Limitations & Future Work**: ceiling analysis, deep-feature path
10. **References**: 10+ citations from §3

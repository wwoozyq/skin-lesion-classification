# Dermoscopy Structural Features — Results

Outcome of the dermoscopy-inspired structural feature experiment described in
`src/features_dermoscopy.py` and run through
`experiments/run_dermoscopy_features.py`.

## TL;DR

- **No new submission winner.** 22 new clinically-motivated dermoscopy
  features (color diversity, blue-white veil, regression-like patches,
  PCA-axis spatial color asymmetry, vascular emphasis) were added to both
  the main LR line and the cascade Stage 2 block. Both pipelines regressed.
- **Main LR + dermoscopy** moves per-seed mean BalAcc from ledger §8's
  **0.7512 → 0.7411** (Δ -0.0101, ~0.7σ of per-seed std). Adding 22 features
  into the `SelectKBest(k=140)` pool dilutes the previously-selected ABCD-grouped
  signal.
- **Cascade + dermoscopy** moves bagged BalAcc from `deeper, k=120, no-pp`
  baseline's **0.7839 → 0.7722** (Δ -0.0117). Per-seed mean drops only
  -0.0022 (0.7683 → 0.7661) — most of the bagged loss is from re-ranking
  the cross-fold ensemble, not from per-seed performance.
- **Cascade per-seed std collapses 0.0206 → 0.0131** (1.6× tighter) — same
  "stability gain without mean improvement" signature documented for A1
  shades_of_gray in `docs/OVERNIGHT_EXPLORATION_RESULTS.md`. Dermoscopy
  acts as a soft regularizer.
- **A1 stacking** partially recovers the dermoscopy loss (cascade +0.0040,
  main LR +0.0068) and collapses cascade per-seed std another 5× to
  **0.0028**, but still loses to the dermoscopy-free baselines. See §6.
- **Leave-one-group-out (LOO) ablation finds no actionable subset.** Best
  cascade variant `drop_asymmetry` lifts to **0.7807** (per-seed mean
  0.7745, +0.0062 over baseline), but bagged still loses by -0.0032 and
  the LOO improvement is well inside multiple-testing noise (10 paired
  comparisons, max-of-10 expected ~0.022 BalAcc just from random
  variance at σ=0.014).
- **Submission decision**: do not adopt. Main submission stays
  `all_abcd_grouped + LR(C=0.3) + k=140`; cascade-track candidate stays
  `xgb_cascade_stage2 + deeper k=120 soft, no preprocessing`. Dermoscopy
  block is documented as another negative result.

## 1. Method recap

### 1.1 Feature design

22 features in 5 clinical groups (see `src/features_dermoscopy.py`):

| group | n | description |
|---|---:|---|
| color_diversity | 6 | Lab L/a/b p5-95 ranges, octant entropy, PCA-1 std, max-octant fraction |
| blue_white | 3 | lesion-relative L > p60 ∧ b < p40 ∧ chroma < p50 area / max-component / n-components |
| regression | 4 | white (L > p75 ∧ chroma < p25) + blue-gray (L < p50 ∧ chroma < p25) |
| asymmetry | 7 | eccentricity + 6× |mean_side_a − mean_side_b| per Lab channel along PCA1/PCA2 |
| vascular | 2 | lesion-relative a* > p75 area (with 20 ≤ L ≤ 85 clamp) + max a* in lesion |

All thresholds are computed **per-lesion** from each lesion's own Lab
percentiles, so the features are robust to global illumination without
relying on shades_of_gray (which is documented as net-negative on the
strong cascade — see overnight exploration results).

Asymmetry features return zero when lesion eccentricity < 0.3 (PCA axis
direction unstable for near-round lesions). Smoke-tested for rotation
invariance: identity / rot90 / fliplr produce numerically identical
feature values on 5 sample lesions.

### 1.2 Pipelines tested

```text
Experiment A — main_lr_dermoscopy
  Features: all_abcd_grouped + dermoscopy (252 total)
  Classifier: StandardScaler + SelectKBest(f_classif, k=140) + LR(C=0.3, balanced)
  Matches ledger §8 main LR exactly except for the dermoscopy block addition.

Experiment B — cascade_s2_dermoscopy
  Stage 1: xgb_cascade_stage1 (79 features), k=100, XGB(n=150, depth=2, lr=0.05, …)
            — unchanged from ledger §9 / §15.
  Stage 2: xgb_cascade_stage2 + dermoscopy (309 features), k=120,
            XGB(n=200, depth=4, lr=0.05, subsample=0.80, colsample=0.60, reg_lambda=5.0)
            — matches the `deeper k=120` variant locked in §15.
  Composition: soft cascade
            P(vasc) = stage1
            P(mel)  = (1 - P_vasc) * stage2_mel
            P(nv)   = (1 - P_vasc) * stage2_nv
```

CV: 5-seed × 5-fold StratifiedGroupKFold grouped by base_id.
Seeds: [42, 127, 2024, 3407, 520].

## 2. Seed-127 sanity (gate)

| pipeline | BalAcc | macro-F1 | Acc |
|---|---:|---:|---:|
| main_lr baseline (ledger §8) | 0.7871 | 0.7715 | 0.7600 |
| main_lr + dermoscopy | **0.7585** | 0.7486 | 0.7283 |
| Δ | **-0.0286** | -0.0229 | -0.0317 |
| cascade baseline (deeper k=120, no-pp) | 0.7941 | 0.7748 | 0.7567 |
| cascade + dermoscopy | **0.7812** | 0.7617 | 0.7417 |
| Δ | **-0.0129** | -0.0131 | -0.0150 |

Main LR's -0.0286 BalAcc is ~1.5σ (per-seed std ≈ 0.014-0.020); cascade's
-0.0129 is well inside single-seed noise. Per locked plan the gate is
"seed127 not regressing"; both technically failed, but cascade's delta
was ambiguous enough to justify running the 5-seed for both.

## 3. Full 5-seed bagged results

| pipeline | bagged BalAcc | bagged F1 | bagged Acc | per-seed BalAcc mean | per-seed std |
|---|---:|---:|---:|---:|---:|
| main_lr baseline (ledger §8) | — | — | — | **0.7512** | — |
| main_lr + dermoscopy | 0.7495 | 0.7515 | 0.7350 | **0.7411** | 0.0140 |
| cascade baseline (deeper k=120, no-pp) | **0.7839** | 0.7609 | 0.7450 | 0.7683 | **0.0206** |
| cascade + dermoscopy | 0.7722 | 0.7501 | 0.7317 | 0.7661 | **0.0131** |

**Read this honestly:**

- Main LR per-seed mean drops 0.7512 → 0.7411 (-0.0101). At per-seed std
  0.014 this is ~0.7σ — not statistically distinguishable from noise on
  any single seed, but the **5/5 seeds all underperformed** the ledger §8
  baseline mean.
- Cascade bagged drops 0.7839 → 0.7722 (-0.0117). Per-seed mean only
  drops 0.7683 → 0.7661 (-0.0022). The bagged-vs-per-seed-mean gap exists
  because the bagged ensemble averages over 25 OOF probability matrices;
  dermoscopy slightly re-orders which folds are confident, shrinking the
  bagging gain from 0.0156 (baseline 0.7839 − 0.7683) to 0.0061 (dermoscopy
  0.7722 − 0.7661).
- Cascade per-seed std collapses 0.0206 → 0.0131 (1.6× tighter). This is
  the **only positive effect** of dermoscopy — it acts as a regularizer
  that tightens cross-seed variance.

## 4. Leave-one-group-out (LOO) ablation

5-seed bagged, one dermoscopy group dropped at a time. Baseline rows are
the "+ all dermoscopy" rows from §3.

### 4.1 Main LR

| drop | bagged BalAcc | per-seed mean | per-seed std | Δ vs "+ all" 0.7411 | Δ vs ledger §8 0.7512 |
|---|---:|---:|---:|---:|---:|
| (none, +all dermoscopy) | 0.7495 | 0.7411 | 0.0140 | — | -0.0101 |
| **drop color_diversity** | **0.7554** | **0.7486** | 0.0113 | **+0.0075** | -0.0026 |
| drop blue_white | 0.7457 | 0.7442 | 0.0158 | +0.0031 | -0.0070 |
| drop regression | 0.7495 | 0.7411 | 0.0140 | 0.0000 | -0.0101 |
| drop asymmetry | 0.7455 | 0.7466 | 0.0146 | +0.0055 | -0.0046 |
| drop vascular | 0.7495 | 0.7414 | 0.0140 | +0.0003 | -0.0100 |

`color_diversity` is the largest negative contributor. Dropping it
recovers most of the dermoscopy loss but **still ends 0.0026 below the
ledger §8 baseline**. `regression` and `vascular` are zero-impact —
SelectKBest doesn't pick them at k=140 because their f_classif scores
fall below the threshold set by 230 existing features.

### 4.2 Cascade Stage 2

| drop | bagged BalAcc | per-seed mean | per-seed std | Δ vs "+ all" 0.7722 | Δ vs baseline 0.7839 |
|---|---:|---:|---:|---:|---:|
| (none, +all dermoscopy) | 0.7722 | 0.7661 | 0.0131 | — | -0.0117 |
| drop color_diversity | 0.7684 | 0.7657 | 0.0150 | -0.0038 | -0.0155 |
| drop blue_white | 0.7733 | 0.7619 | 0.0160 | +0.0011 | -0.0106 |
| drop regression | 0.7706 | 0.7665 | 0.0139 | -0.0016 | -0.0133 |
| **drop asymmetry** | **0.7807** | **0.7745** | 0.0148 | **+0.0085** | **-0.0032** |
| drop vascular | 0.7688 | 0.7687 | 0.0163 | -0.0034 | -0.0151 |

`asymmetry` (7 features) is the largest negative contributor to cascade
performance. Dropping it lifts bagged BalAcc by +0.0085 and per-seed
mean by +0.0084, landing 0.0032 below baseline.

### 4.3 Multiple-testing sanity check

10 paired LOO comparisons (5 groups × 2 pipelines) against their
respective baselines. At per-seed σ ≈ 0.014, the maximum of 10 i.i.d.
zero-mean comparisons has expected magnitude
`σ · Φ⁻¹(1 − 0.5/10) ≈ 0.014 · 1.6 ≈ 0.022 BalAcc`. The best observed
LOO lift is `drop_asymmetry` cascade per-seed +0.0062 — well below the
random-max expectation. **No LOO subset is statistically distinguishable
from baseline.**

## 5. Per-cell interpretation

**color_diversity (6 features) — net-negative for main LR, neutral for
cascade.** LR with SelectKBest re-ranks the 252-feature pool; the 6 color
diversity features displace some of the previously-selected ABCD-grouped
color features without adding more signal than they removed. XGBoost
ignores low-information features at split time, so cascade is unaffected.

**blue_white (3 features) — neutral.** Documented dermoscopic structure,
but at lesion-relative thresholds (p60/p40/p50) the detector fires on
many non-blue-white lesions too; the resulting feature is barely
informative beyond what `boundary_color_contrast` and `melnv_lab_spread`
already capture.

**regression (4 features) — neutral.** Same root cause as blue_white:
lesion-relative percentile thresholds make the detector fire on any
lesion with a bright-low-chroma patch (which is common in NV with
hypopigmented centers, not just regression structures). Tightening to
absolute Lab thresholds would change this but would also reintroduce
illuminant dependency.

**asymmetry (7 features) — net-negative for cascade.** This was the
surprise. PCA-axis spatial color asymmetry was meant to be the most
physically meaningful feature (rotation-invariant by construction, with
low-eccentricity safeguard). On the cascade it actively hurts:
`drop_asymmetry` recovers +0.0085 bagged. Two likely causes: (i)
PCA-axis orientation flips between paired augmented copies (eccentricity
0.31 vs 0.29 can produce 90° rotation), generating high variance under
the `groups=base_id` rule; (ii) XGBoost trees over-fit on the 7 strongly
correlated asymmetry channels at depth 4.

**vascular (2 features) — neutral.** Dermatoscopes with red-shifted
illumination push a* upward for many non-vascular lesions; the
lesion-relative p75 threshold then fires on any lesion. Not a useful
discriminator.

## 6. A1 stacking ablation

To test the "two regularizers stack" hypothesis, both pipelines were
re-run with `shades_of_gray` preprocessing on top of the dermoscopy
features (mode `a1_ablation`).

| pipeline | bagged BalAcc | bagged F1 | bagged Acc | per-seed std | Δ vs no-A1 |
|---|---:|---:|---:|---:|---:|
| main_lr + dermoscopy (no A1) | 0.7495 | 0.7515 | 0.7350 | 0.0140 | — |
| **main_lr + dermoscopy + A1** | **0.7563** | 0.7566 | 0.7417 | 0.0175 | **+0.0068** |
| cascade + dermoscopy (no A1) | 0.7722 | 0.7501 | 0.7317 | 0.0131 | — |
| **cascade + dermoscopy + A1** | **0.7762** | 0.7639 | 0.7417 | **0.0028** | **+0.0040** |

A1 **partially recovers** the dermoscopy loss on both pipelines:

- Main LR + dermoscopy + A1 = 0.7563 lands 0.0068 above no-A1 (0.7495),
  but still 0.0026 below the dermoscopy-free ledger §8 baseline mean.
- Cascade + dermoscopy + A1 = 0.7762 lands 0.0040 above no-A1 (0.7722),
  still 0.0077 below the dermoscopy-free `deeper k=120 no-pp` baseline
  (0.7839).
- Cascade per-seed std collapses **0.0206 → 0.0131 → 0.0028** as we
  stack dermoscopy then A1 — a 7× tightening over baseline. This is
  consistent with the "two regularizers" story but the cost in bagged
  mean BalAcc is real.

§15 documented A1 as net-positive on under-regularized cascade variants
(+0.012 to +0.020 on `k=all`) and net-negative on the strong `deeper
k=120` variant (-0.0018). Adding dermoscopy slightly under-regularizes
the strong cascade (loosens the SelectKBest+tree-depth pinch by adding
noisy features), so A1 once again becomes net-positive when stacked on
top. Both interventions act on the same "feature noise from illumination
and weak descriptors" axis. The net result is still a net-negative stack
because the underlying dermoscopy block dominates.

**Comparison to A1-alone (from §15 overnight results):**

| configuration | bagged BalAcc | per-seed std |
|---|---:|---:|
| baseline (no dermoscopy, no A1) | 0.7839 | 0.0206 |
| + dermoscopy | 0.7722 | 0.0131 |
| + A1 (no dermoscopy) | 0.7821 | 0.0066 |
| + dermoscopy + A1 | 0.7762 | 0.0028 |

A1 alone is the strongest variance-reducing intervention (still 0.0018
below baseline mean). Adding dermoscopy on top makes both mean and
variance worse than A1 alone. **Do not stack.**

## 7. Submission impact

| component | before | after | change |
|---|---|---|---|
| main submission | `all_abcd_grouped + LR(C=0.3) + k=140` | (unchanged) | — |
| cascade-track candidate | `deeper k=120 soft, no-pp` | (unchanged) | — |
| documented negative results | A1, A2, A3, B1, hair-removal, TTA+nested, 2-stage refinement, mel/nv error features | + dermoscopy block | +1 |

Dermoscopy joins the documented negative-result roster. No code paths
in `run.py` / `src/train_ml.py` / `src/train_ml_cascade.py` change.

## 8. Files produced

| path | purpose |
|---|---|
| `src/features_dermoscopy.py` | feature extractor (227 lines) |
| `experiments/run_dermoscopy_features.py` | runner (5 modes) |
| `outputs/metrics/dermoscopy_results.csv` | all 5-seed rows (12 experiments) |
| `outputs/cache/features_cascade_all_abcd_grouped_dermoscopy_raw_pp_none.csv` | main-LR feature cache (252 cols) |
| `outputs/cache/features_cascade_xgb_cascade_stage2_dermoscopy_raw_pp_none.csv` | cascade-S2 feature cache (309 cols) |

## 9. Negative-result framing for the report

> "We added 22 clinically-motivated dermoscopy structural features — color
> diversity, blue-white veil, regression-like patches, PCA-axis spatial
> color asymmetry, and vascular emphasis — designed with lesion-relative
> Lab percentile thresholds to avoid the illuminant dependency that made
> shades_of_gray preprocessing net-negative on our strong cascade. The
> features were rotation/flip-invariant by construction and passed smoke
> tests on individual lesions. Under 5-seed bagged StratifiedGroupKFold
> evaluation, dermoscopy was net-negative on both the main LR pipeline
> (per-seed mean −0.0101 BalAcc) and the cascade (bagged −0.0117 BalAcc).
> Leave-one-group-out ablation found that color_diversity and asymmetry
> were the largest negative contributors to LR and cascade respectively,
> but even the best LOO subset still lost to baseline within the
> multiple-testing noise floor. The one durable benefit was per-seed
> variance reduction in the cascade (std 0.0206 → 0.0131, 1.6× tighter),
> consistent with the A1 result: features that look like regularization
> are redundant once the model is well-regularized via SelectKBest and
> tree depth/L2."

# Pre-Submission Research Plan

Experimental track for pushing the 5-seed bagged XGBoost cascade beyond its
current 0.7887 BalAcc ceiling. All work targets the **2026-06-10 13:25**
code zip deadline (`Project-2_王礼铭.zip`).

## Decision Context

- Final submission model: **option 3 — XGBoost cascade, 5-seed bagged,
  trained on full 600 images**, saved as `joblib` bundle.
- Headline number to beat: cascade 5-seed bagged
  **0.7887 BalAcc / 0.7655 macro-F1 / 0.7500 accuracy** (ledger §9).
- Single-seed peak 0.8055 (ledger §11) is in-sample noise — proven negative
  in ledger §12 nested-CV / TTA experiment. Do not chase it again.
- Group leader: 王礼铭. Filename format `Project-2_王礼铭.{zip,pptx,pdf}`.

## Track A — Medically Motivated Preprocessing

Apply standard dermoscopy preprocessing before feature extraction. Each
candidate gets one CV sweep with the cascade pipeline; positive findings
stack into the final submission model, negative findings get logged in
`docs/EXPERIMENT_LEDGER.md` and not merged.

### A1: Shades-of-Gray color constancy — PRIORITY 1, doing now

| field | value |
|---|---|
| Hypothesis | Cross-image white-balance variance dilutes mel/nv color comparisons. Normalizing per-image to a Minkowski-norm gray world removes the lighting confound. |
| Method | `e_k = ((1/N) ∑ I_k^p)^(1/p)` with p=6, then per-channel gain `gain_k = (1/√3) / (e_k / ‖e‖)`, applied to RGB. |
| Expected | +0.01 ~ +0.03 BalAcc on 5-seed bagged cascade |
| Cost | ~3 hours (impl + CV sweep + write-up) |
| Files | new `src/preprocess_color.py`; touch `src/features.py` cache key |

### A2: CLAHE on Lab L channel — PRIORITY 3

| field | value |
|---|---|
| Hypothesis | Local contrast enhancement makes pigment network and border irregularity visible to boundary/texture features. |
| Method | RGB → Lab; `skimage.exposure.equalize_adapthist(L, clip_limit=0.01)`; Lab → RGB. |
| Expected | +0.005 ~ +0.02 BalAcc |
| Cost | ~2 hours |

### A3: Hemoglobin / Melanin channel separation — PRIORITY 4 (optional)

| field | value |
|---|---|
| Hypothesis | Beer-Lambert decomposition gives Stage 1 a near-optimal vasc signal (vasc is essentially the hemoglobin channel). |
| Method | Tsumura projection: `log(RGB) → hemoglobin + melanin maps`. Use as extra inputs for Stage 1 features. |
| Expected | +0.005 ~ +0.015 BalAcc, mostly Stage 1 recall |
| Cost | ~3 hours |

## Track B — Feature Engineering

### B1: add `abcd_grouped` to cascade Stage 2 — PRIORITY 2

| field | value |
|---|---|
| Hypothesis | ABCD-grouped features were never tested inside the cascade, but they are exactly the clinical mel-vs-nv markers. Stage 2 currently uses heavy texture engineering and no ABCD-grouped block. |
| Method | edit `xgb_cascade_stage2` in `src/features.py` to include `"abcd_grouped"`; re-fit cascade; compare bagged BalAcc. |
| Expected | +0 ~ +0.02 BalAcc |
| Risk | XGBoost feature dilution; should not hurt but might not help. |
| Cost | ~1 hour |

## Track C — Deployment Plumbing

Required regardless of A/B results. Build last, after preprocessing and
features stabilize, to avoid re-serializing models.

### C1: 5-seed bagging trainer

Train 5 cascades with seeds [42, 127, 2024, 3407, 520] on full 600
images. Save as `outputs/models/cascade_bagged/seed_{seed}.joblib`.

### C2: `run.py` — new `cascade_bagged` model type

New branch in `run.py`: load 5 bundles, average `predict_proba`, argmax.

### C3: submission zip

`Project-2_王礼铭.zip`. Structure follows the assignment demo example;
includes `run.py`, `src/`, weights, `readme.txt` (env + invocation).
No data shipped.

## Closed Tracks (do not retry)

| track | result | source |
|---|---|---|
| Hair removal | -0.022 BalAcc | ledger §13 |
| Mask cleaning | marginal negative | ledger §3 |
| TTA + nested-CV fusion weight | no bagged gain | ledger §12 |
| Early feature fusion | 0.7766 < 0.7887 | ledger §10 |
| Two-stage mel/nv refinement | no improvement | ledger §7 |
| Mel/nv error-driven features | no improvement | ledger §6 |

## Evaluation Protocol (constant across A and B)

```text
CV:      5-seed × 5-fold StratifiedGroupKFold, grouped by base_id
Seeds:   [42, 127, 2024, 3407, 520]
Metric:  bagged accuracy, macro-F1, balanced accuracy
Sanity:  cascade baseline must reproduce 0.7887 ± 0.005 BalAcc per run
Pass:    > 0.7887 BalAcc bagged → adopt
Fail:    < 0.7860 BalAcc bagged → revert, log as negative
```

## Working Timeline

| date | task |
|---|---|
| 5/29 – 5/30 | A1 color constancy: impl + CV + write-up |
| 5/31 – 6/01 | A1 positive → B1 cascade Stage 2 augmentation. A1 negative → A2 directly. |
| 6/02 – 6/03 | A2 CLAHE |
| 6/04 | A3 if A1+A2+B1 still leave headroom |
| 6/05 – 6/06 | C1 + C2 deployment |
| 6/07 | final 5-seed validation, zip dry-run |
| 6/08 – 6/10 | buffer + submission |

## Result Logging

Each experiment lands a new section in `docs/EXPERIMENT_LEDGER.md`
(continuing current numbering at §15+). Positive findings additionally
update the cascade pipeline on `main`; negative findings stay on the
experiment branch.

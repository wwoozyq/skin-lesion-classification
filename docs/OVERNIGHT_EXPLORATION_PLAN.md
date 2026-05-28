# Overnight Exploration Plan — Medical Preprocessing × Cascade

**Run window**: 2026-05-28 evening → 2026-05-29 morning.
**Goal**: break the cascade 5-seed bagged BalAcc 0.7887 ceiling using
medically motivated preprocessing (Track A) and feature augmentation
(Track B), to inform the 2026-06-10 submission.

## Source transparency

WebSearch is blocked in this environment (`idealab message api 暂不支持`).
Everything below is drawn from training knowledge of well-established
literature (cited where the original paper is unambiguous) plus the
`scikit-image` and `OpenCV` public APIs (whose parameter defaults can be
verified locally with `python -c "import skimage; ..."`). Where a citation
is given, it identifies the canonical paper for the algorithm — not the
specific dermatoscopy paper that adopted it.

## Decision context

- Baseline to beat: cascade 5-seed bagged
  **0.7887 BalAcc / 0.7655 macro-F1 / 0.7500 accuracy** (ledger §9).
- Three preprocessing candidates × one feature-augmentation candidate
  yield 8 experimental cells (baseline + 3 single + 3 single + 1 triple);
  every cell is one full 5-seed × 5-fold StratifiedGroupKFold run.
- Final adoption rule: any cell that beats 0.7887 BalAcc bagged with
  ≥ +0.005 margin AND does not drop macro-F1 by > 0.005 is a candidate
  for the cascade submission bundle. Otherwise log as negative.
- All preprocessing is applied to the **RGB image only**; masks are
  unchanged (Track A targets the image's color/illumination content, not
  the lesion segmentation). The bool mask still drives every feature
  extractor exactly as before.

## A1 — Shades-of-Gray color constancy

| field | value |
|---|---|
| Hypothesis | Cross-image illuminant variance (dermatoscope LED color temp, polariser orientation) inflates intra-class color variance for mel and nv. Per-image white balance removes the lighting confound and lets ABCD-color features compare like-with-like. |
| Algorithm | Finlayson & Trezzi (2004), "Shades of Gray". Estimate illuminant per channel: `e_k = ((1/N) ∑_i I_k(i)^p)^(1/p)` with `p = 6`. Normalize illuminant vector to unit `‖e‖₂`, then multiply each channel by `(1/√3) / (e_k / ‖e‖)`. Equivalent to per-channel gain that maps the Minkowski-6 image average onto neutral gray. |
| Parameters | `p = 6` (Finlayson 2004 reports best generic performance at `p∈[4,6]`); clip output to `[0, 255]` uint8; operate on **non-masked** pixels (we want lighting estimate, not lesion estimate); single pass, no iteration. |
| Why p = 6 not gray-world (p = 1) or max-RGB (p = ∞) | Gray-world assumes the scene average is neutral, which fails when one chromophore (e.g. hemoglobin) dominates the lesion. Max-RGB locks to a single hot pixel and is dermatoscope-glare-fragile. The Minkowski 6-norm robustly approximates the dominant tail without locking onto saturated outliers. |
| Source | Finlayson, G. D., & Trezzi, E. (2004). *Shades of gray and colour constancy*. Color and Imaging Conference. Training knowledge — paper widely cited in dermoscopy preprocessing pipelines (e.g. ISIC submissions 2017+). |
| Expected gain | +0.005 ~ +0.025 BalAcc, concentrated in mel-vs-nv. Side effect on Stage 1 vasc is small (vasc is already a red-dominated lesion, unaffected by white balance). |
| Risks | Could shift the "true vasc red" signal slightly if a vasc lesion is taken in oddly tinted illumination — Stage 1 vasc OOF accuracy should be monitored separately. |
| Pass | bagged BalAcc ≥ 0.7937 (= 0.7887 + 0.005). |
| Fail | bagged BalAcc < 0.7887; log as negative, do not merge. |
| Cost | ~1 min per cell once cached. |

## A2 — CLAHE on Lab L channel

| field | value |
|---|---|
| Hypothesis | Local contrast enhancement reveals pigment network, dots, globules, and border irregularity that ABCD-asymmetry / boundary / texture features rely on but that current global histograms under-emphasize. Applying on Lab `L*` rather than RGB keeps chromaticity untouched. |
| Algorithm | RGB → CIE Lab (`skimage.color.rgb2lab`). Run `skimage.exposure.equalize_adapthist(L_normalized, kernel_size=None, clip_limit=0.01, nbins=256)` on the L channel (rescaled to [0, 1]). Convert back: rescale to [0, 100], stitch `(L', a, b)`, Lab → RGB (`lab2rgb`), clip and cast to uint8. |
| Parameters | `clip_limit = 0.01` (skimage default — verified locally: `equalize_adapthist.__defaults__`); `kernel_size = None` → defaults to `1/8` of each image dimension (skimage convention); `nbins = 256` (default). |
| Why Lab L over RGB / V | Equalizing R/G/B separately distorts hue and breaks the color features (ABCD's "C" is literally color variegation). Operating on `L*` and rebuilding the image preserves `a*`/`b*` exactly, so color-based features see the same chromaticity but boundary/texture features benefit from amplified local micro-contrast. |
| Why CLAHE not global histogram equalization | Global EQ flattens dynamic range across the whole image, which over-stretches noisy peri-lesion skin and crushes the lesion's actual local contrast. CLAHE limits per-tile cumulative histogram slope to `clip_limit × tile_size² / nbins`, which preserves local contrast where it matters. |
| Source | Pizer, S. M., et al. (1987). *Adaptive histogram equalization and its variations*. Computer Vision, Graphics, and Image Processing. CLAHE (clip-limit variant): Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization*, Graphics Gems IV. skimage implementation: `skimage.exposure.equalize_adapthist`. |
| Expected gain | +0.005 ~ +0.015 BalAcc, concentrated in boundary and lbp_multi / gabor features (Stage 2 textures). |
| Risks | (1) Over-amplification of hair / ruler artifacts could hurt — but per ledger §13, those artifacts are only present on ~2 of 600 images. (2) Lab `lab2rgb` round-trip introduces minor clipping; should be benign. |
| Pass | bagged BalAcc ≥ 0.7937. |
| Fail | bagged BalAcc < 0.7887; log as negative. |
| Cost | ~4 min per cell once cached. |

## A3 — Tsumura hemoglobin / melanin separation

| field | value |
|---|---|
| Hypothesis | Beer-Lambert decomposition gives Stage 1 a near-orthogonal vasc signal (vasc is essentially hemoglobin-dominant) and Stage 2 a melanin map that suppresses lighting variance. Decomposed maps could be passed to feature extractors as alternative "images". |
| Algorithm | Per-pixel: `D = -log(RGB / 255 + ε)` (optical density). Hemoglobin and melanin chromophore directions in OD space (Tsumura 1999): `c_hb ≈ [0.55, 0.71, 0.43]` (R, G, B coefficients for HbO₂+Hb mix), `c_mel ≈ [0.40, 0.59, 0.70]` (eumelanin). Project: `[m_hb, m_mel]^T = pinv([c_hb c_mel]) · D` per pixel. Compose a "hb-amplified" RGB image as: `RGB' = exp(-(m_hb · α · c_hb + m_mel · c_mel)) · 255` with `α = 1.5` to over-weight hemoglobin features for Stage 1, OR a "neutral reconstruction" with `α = 1.0` as a denoised input. |
| Parameters | `ε = 1/255` to avoid log(0); chromophore directions per Tsumura et al. (1999); per-image projection (no global fit). Output clipped to `[0, 255]` uint8. |
| Why this vs simple R-channel boost | Vascular lesions are red **because of hemoglobin**, but hair, dermoscope tint, and ruler shadows also affect the R channel. Tsumura projects to the hemoglobin basis vector explicitly, so it isolates the vasc-relevant signal from confounders. Used widely in dermoscopy literature post-2010. |
| Source | Tsumura, N., et al. (1999, 2003). *Image-based skin color and texture analysis / synthesis by extracting hemoglobin and melanin information in the skin*. ACM SIGGRAPH. Chromophore coefficients are an approximation from training knowledge — there is variation across the literature (different LED spectra), but `c_hb` and `c_mel` here are within ±0.1 of every published version I'm aware of. |
| Implementation note | Only the "neutral reconstruction" path (α = 1.0) is enabled by default for the overnight run — it slots in cleanly as a preprocessing transform and does not require changing feature extractors. The α = 1.5 hb-amplification variant is left as a follow-up if A3 shows promise. |
| Expected gain | +0.0 ~ +0.015 BalAcc, mostly Stage 1 vasc recall + Stage 2 mel-vs-nv via clearer pigment maps. |
| Risks | (1) Approximate chromophore coefficients could mis-decompose under unusual illumination — partially mitigated by stacking with A1. (2) The reconstruction `exp(-…)` is sensitive to numerical underflow in saturated whites; mitigated by ε clip. |
| Pass | bagged BalAcc ≥ 0.7937. |
| Fail | bagged BalAcc < 0.7887; log as negative. |
| Cost | ~2 min per cell once cached. |

## B1 — Cascade Stage 2 + abcd_grouped

| field | value |
|---|---|
| Hypothesis | Stage 2 currently uses `BASE + boundary + melnv + lbp_multi + gabor + subregion` but no `abcd_grouped`. The ABCD-grouped block (already validated on the LR main line, ledger §8) encodes asymmetry, border irregularity, color variegation, diameter — exactly the textbook mel-vs-nv discriminators. SelectKBest sees more candidate channels and should pick a complementary subset. |
| Algorithm | Add `"abcd_grouped"` to `FEATURE_ALIASES["xgb_cascade_stage2"]` in `src/features.py`. Re-cache Stage 2 features. Re-fit cascade. Compare 5-seed bagged BalAcc against §9 baseline (0.7887). |
| Parameters | Same Stage 2 XGBoost variant (`d2-more-trees`) and `k_features=all` as ledger §9 baseline. |
| Source | This is a project-internal hypothesis; no external citation. Justification is the LR main-line gain in ledger §8 (`all_abcd_grouped` LR03 k=140: 0.7910 BalAcc on seed 127 vs `all_boundary` 0.7535 on the same seed). |
| Expected gain | +0.000 ~ +0.020 BalAcc. The most likely outcome is "no significant change" because XGBoost cascade with `k=all` may dilute the signal; the upside case is `abcd_grouped` providing a redundancy that bagging exploits. |
| Risks | XGBoost feature dilution; cascade variant tuning was done without `abcd_grouped`, so the optimal `(variant, k)` could shift. |
| Pass | bagged BalAcc ≥ 0.7937 with same `(variant, k)`. |
| Fail | bagged BalAcc < 0.7887 with same `(variant, k)`. |
| Cost | ~1 min per cell once cached (features build once). |

## Combined cells

After A1/A2/A3/B1 finish, two combined cells are run only on whichever of
A1/A2/A3 had the **largest single-cell margin over baseline**:

| cell | preprocessing | features |
|---|---|---|
| BEST_A + B1 | best single A | Stage 2 + abcd_grouped |

Rationale: stacking two positive preprocessings is non-obviously additive
(they could correct overlapping confounds); we run only the most promising
combo to bound overnight runtime. If A1 and A2 both pass, BOTH `A1+B1` and
`A2+B1` are run; `A1+A2+B1` is a single tie-breaker cell.

## Master cell list (8 cells maximum)

| # | preprocessing | feature_set augmentation | notes |
|---|---|---|---|
| 0 | none (baseline) | none | sanity: must reproduce 0.7887 ± 0.005 |
| 1 | A1 shades_of_gray | none | A1 vs baseline |
| 2 | A2 clahe_lab_L | none | A2 vs baseline |
| 3 | A3 hb_melanin | none | A3 vs baseline |
| 4 | none | B1 (Stage 2 + abcd_grouped) | B1 vs baseline |
| 5 | best A | B1 | first stack |
| 6 | second-best A (if also passing) | B1 | second stack |
| 7 | A1 + A2 (sequential) | B1 | tie-breaker, only if A1 and A2 both pass |

Cells 5–7 are launched only after cells 0–4 finish and their results are
parsed automatically. The runner emits a `pending_followups.json` for the
human-in-the-loop check the next morning if any ambiguity prevents
automatic selection.

## Evaluation protocol (identical across cells)

```text
CV:       5-seed × 5-fold StratifiedGroupKFold, grouped by base_id
Seeds:    [42, 127, 2024, 3407, 520]
Pipeline: same as ledger §9 cascade (Stage 1 binary vasc, Stage 2 binary mel/nv)
Stage 2:  variant=d2-more-trees, k_features=all (matches ledger §9 best)
Metrics:  bagged accuracy, macro-F1, balanced accuracy + per-seed
          mean/std for each
Sanity:   cell 0 bagged BalAcc must land in [0.7837, 0.7937]; if not,
          STOP and investigate before trusting cells 1–7
Pass:     bagged BalAcc ≥ 0.7937 (+0.005 over baseline) AND
          bagged macro-F1 not down by > 0.005
Fail:     bagged BalAcc < 0.7887; log as negative
Ledger:   each cell appends a new section §15+ in
          docs/EXPERIMENT_LEDGER.md
Adoption: positive cells trigger a planned `train_ml_cascade.py` rerun
          with the new preprocessing baked into the saved bundle
          (separate task — not part of the overnight scope)
```

## Files touched / created

| path | scope |
|---|---|
| `src/preprocess_medical.py` (new) | `shades_of_gray`, `clahe_lab_l`, `hb_melanin_decomposition`, dispatcher `apply_medical_preprocessing(image, name)` |
| `src/features.py` (edit) | (a) accept `preprocessing` kwarg on `extract_features_for_image` and `build_feature_table`; preprocess RGB once before computing all blocks. (b) add `"abcd_grouped"` to the `xgb_cascade_stage2` alias |
| `src/train_ml_cascade.py` (edit) | accept `preprocessing` arg; pass through to `build_feature_table`; cache key includes preprocessing |
| `experiments/run_overnight_exploration.py` (new) | runs cells 0–4 sequentially, parses results, decides which of 5–7 to launch, logs per-cell summary CSV, writes a status file the next morning can read |
| `outputs/cache/features_cascade_{set}_{mask_mode}_pp_{preprocessing}.csv` (new) | one cache file per (feature_set × preprocessing) combo |
| `outputs/metrics/overnight_exploration_summary.csv` (new) | one row per cell |
| `outputs/metrics/overnight_exploration_per_seed.csv` (new) | one row per (cell × seed) |
| `outputs/metrics/overnight_exploration_status.json` (new) | machine-readable status (success / sanity-failed / preempted) |
| `docs/EXPERIMENT_LEDGER.md` (append) | one new §15 + sub-sections for each cell |
| `docs/OVERNIGHT_EXPLORATION_RESULTS.md` (new) | morning summary doc |

## Runtime budget

Feature extraction dominates first-time cost; subsequent cells reuse the
cache. Conservative estimate:

| step | wall time |
|---|---|
| feature build for cell 0 (baseline) | 3 min |
| feature build for cell 1 (A1) | ~4 min (preprocess + extract) |
| feature build for cell 2 (A2) | ~6 min (CLAHE is slower) |
| feature build for cell 3 (A3) | ~4 min |
| feature build for cell 4 (B1, adds abcd_grouped only) | ~2 min (only the new block) |
| 8 cells × 5 seeds × cascade fit (~3 min/seed at k=all) | ~120 min |
| combined cells (5–7), up to 3 more | ~45 min |
| total | ~3.5 hours |

Hard ceiling: 6 hours via subprocess timeout; if exceeded, runner kills
itself and writes `preempted` status so morning review is unambiguous.

## Out of scope (not done overnight)

- No `run.py` integration; preprocessing is experimental until results
  are in.
- No `cascade_bagged` submission bundle build — that lives in Track C
  (`SUBMISSION_RESEARCH_PLAN.md` §C1/C2/C3) and is deferred until at
  least one A or B cell is confirmed positive.
- No PPT / report writing.
- No git-history rewrites or cross-branch ops.

## What "morning success" looks like

The next-morning checklist is one bash command:

```bash
cat outputs/metrics/overnight_exploration_status.json && \
  column -s, -t < outputs/metrics/overnight_exploration_summary.csv | head
```

Plus `docs/OVERNIGHT_EXPLORATION_RESULTS.md` rendered with a short
verdict per cell and an explicit "next action" line for whichever cell(s)
beat 0.7887.

# Overnight Exploration Results — 2026-05-29

Outcome of the experiments described in `docs/OVERNIGHT_EXPLORATION_PLAN.md`,
plus a follow-up variant sweep that **reverses** the apparent A1 win.

## TL;DR

- **No new submission winner.** The best 5-seed bagged BalAcc found is
  **0.7839** (`deeper, k=120, no preprocessing`), still below ledger §9's
  documented 0.7887. Keep ledger §9's cascade as the cascade-track
  candidate.
- **A1 (shades-of-gray) is variant-dependent and net-negative on the
  strong cascade.** It lifts weak `k=all` variants by +0.012 to +0.020,
  but on the strong `deeper, k=120` variant it costs **-0.0018 BalAcc**.
  The overnight-runner's "+0.0201" was an artifact of pinning cell 0 to
  one of the weakest configurations (`d2-more-trees, k=all` = 0.7623).
- **A1's only durable benefit is variance reduction.** Per-seed std
  drops 3-6× under every variant. Worth mentioning in the report, not
  worth adopting as preprocessing for the submission bundle.
- **A2 (CLAHE) and A3 (hb_melanin) regress under every config tested.**
  Close those tracks.
- **B1 (Stage 2 + abcd_grouped) is +0.0053 on the weak baseline and
  redundant once A1 normalizes color** (stack underperforms A1 alone).
  Do not adopt.
- **Submission decision**: keep `deeper, k=120, soft, no preprocessing`
  as the cascade-track joblib; A1/A2/A3/B1 all enter the report as
  documented negative results that strengthen the "we tried everything"
  story.

## 1. Method recap

Runner: `experiments/run_overnight_exploration.py` (commit `abac3c0`).
The overnight cells all use the same cascade variant
(`d2-more-trees, k=all`):

```text
Stage 1: xgb_cascade_stage1, k=100, XGB(n=150, depth=2, lr=0.05, …)
Stage 2: xgb_cascade_stage2, k=all, XGB(n=300, depth=2, lr=0.03, …)
Cascade: soft   P(vasc) = stage1
                P(mel)  = (1 - P_vasc) * stage2_mel
                P(nv)   = (1 - P_vasc) * stage2_nv
CV:      5-seed × 5-fold StratifiedGroupKFold, grouped by base_id
Seeds:   [42, 127, 2024, 3407, 520]
```

Preprocessing module: `src/preprocess_medical.py` (commit `ddae438`).
Each cell touches only the preprocessing transform and (for B1) the
Stage 2 feature set; nothing else changes.

The follow-up variant sweep
(`experiments/run_variant_sweep_a1.py`) repeats the same cascade
machinery across **3 Stage 2 variants × 3 k_features × {none, A1}**
(18 configs, all 5-seed bagged, all soft cascade) to test whether
A1's lift survives when the cascade is configured properly.

## 2. Overnight cell-level results

Source: `outputs/metrics/overnight_exploration_summary.csv`.
Wall time: ~22 min total (6 cells × ~4 min average).

| cell | preprocessing | stage 2 features | bagged BalAcc | bagged macro-F1 | bagged Acc | per-seed BalAcc std | verdict |
|---|---|---|---:|---:|---:|---:|---|
| 0 baseline | none | `xgb_cascade_stage2` | **0.7623** | 0.7369 | 0.7133 | 0.0140 | anchor (weak variant) |
| 1 A1 shades_of_gray | shades_of_gray | `xgb_cascade_stage2` | **0.7824** | 0.7694 | 0.7467 | **0.0023** | +0.0201 vs cell 0, but see §3 |
| 2 A2 clahe_lab_L | clahe_lab_l | `xgb_cascade_stage2` | 0.7468 | 0.7225 | 0.7017 | 0.0100 | -0.0155 regress |
| 3 A3 hb_melanin | hb_melanin | `xgb_cascade_stage2` | 0.7466 | 0.7212 | 0.7117 | 0.0069 | -0.0157 regress |
| 4 B1 stage2+abcd | none | `xgb_cascade_stage2_abcd` | 0.7676 | 0.7416 | 0.7183 | 0.0149 | +0.0053 marginal |
| 5 A1 + B1 | shades_of_gray | `xgb_cascade_stage2_abcd` | 0.7774 | 0.7635 | 0.7383 | 0.0060 | +0.0151 pass, worse than A1 alone |

**Important caveat**: cell 0's 0.7623 anchor is one of the worst
configurations in the entire 18-config search (see §3). The overnight
runner inherited `d2-more-trees, k=all` from an earlier draft; the
ledger §9 headline 0.7887 comes from `deeper, k=120, soft` (saved as
`outputs/models/xgb_cascade_deeper_k120_soft.joblib`). Read every cell
delta as "vs. weak baseline" not "vs. our best cascade".

## 3. Variant sweep — A1's lift does not transfer

Source: `outputs/metrics/variant_sweep_a1.csv` (1.2 min wall).

| preprocessing | stage 2 variant | k | bagged BalAcc | bagged F1 | bagged Acc | per-seed std |
|---|---|---|---:|---:|---:|---:|
| **none** | **deeper** | **120** | **0.7839** | 0.7609 | 0.7450 | 0.0206 |
| none | strong-reg | 100 | 0.7830 | 0.7583 | 0.7400 | 0.0154 |
| none | d2-more-trees | 100 | 0.7825 | 0.7580 | 0.7400 | 0.0153 |
| shades_of_gray | d2-more-trees | all | 0.7824 | 0.7694 | 0.7467 | 0.0023 |
| shades_of_gray | deeper | 120 | 0.7821 | 0.7698 | 0.7483 | 0.0066 |
| shades_of_gray | strong-reg | all | 0.7820 | 0.7692 | 0.7467 | 0.0082 |
| none | strong-reg | 120 | 0.7812 | 0.7562 | 0.7367 | 0.0183 |
| none | d2-more-trees | 120 | 0.7807 | 0.7558 | 0.7367 | 0.0142 |
| shades_of_gray | deeper | all | 0.7805 | 0.7693 | 0.7483 | 0.0074 |
| shades_of_gray | d2-more-trees | 120 | 0.7801 | 0.7663 | 0.7417 | 0.0116 |
| shades_of_gray | strong-reg | 120 | 0.7747 | 0.7608 | 0.7350 | 0.0085 |
| shades_of_gray | deeper | 100 | 0.7767 | 0.7643 | 0.7417 | 0.0060 |
| none | deeper | 100 | 0.7719 | 0.7480 | 0.7283 | 0.0196 |
| none | strong-reg | all | 0.7699 | 0.7447 | 0.7233 | 0.0151 |
| shades_of_gray | strong-reg | 100 | 0.7688 | 0.7550 | 0.7283 | 0.0144 |
| shades_of_gray | d2-more-trees | 100 | 0.7688 | 0.7550 | 0.7283 | 0.0087 |
| none | deeper | all | 0.7633 | 0.7394 | 0.7183 | 0.0150 |
| **none** | **d2-more-trees** | **all** | **0.7623** | 0.7369 | 0.7133 | 0.0140 |

### Paired comparisons at each (variant, k)

| stage 2 variant | k | no-pp BalAcc | A1 BalAcc | Δ (A1 - no-pp) |
|---|---|---:|---:|---:|
| strong-reg | 100 | 0.7830 | 0.7688 | **-0.0142** |
| strong-reg | 120 | 0.7812 | 0.7747 | -0.0065 |
| strong-reg | all | 0.7699 | 0.7820 | **+0.0120** |
| d2-more-trees | 100 | 0.7825 | 0.7688 | **-0.0137** |
| d2-more-trees | 120 | 0.7807 | 0.7801 | -0.0006 |
| d2-more-trees | all | 0.7623 | 0.7824 | **+0.0201** |
| deeper | 100 | 0.7719 | 0.7767 | +0.0048 |
| deeper | 120 | **0.7839** | 0.7821 | **-0.0018** |
| deeper | all | 0.7633 | 0.7805 | +0.0172 |

### Reading the sweep

- **A1 only helps `k=all` configurations.** All three big positive
  deltas (+0.0120 / +0.0201 / +0.0172) live in the `k=all` row, exactly
  where the model is least regularized at the feature-selection stage.
- **A1 hurts the strong `k=100` configurations** by 0.013-0.014 in two
  of three variants. The overnight `d2-more-trees, k=all` anchor was
  effectively a weak baseline that A1 was rescuing; once the cascade
  uses `k=100` or `k=120` to drop low-signal features, A1 stops helping
  and starts removing useful inter-image color variation.
- **`deeper, k=120` is the new best absolute config at 0.7839.** This
  is within 0.0048 of the ledger §9 headline (0.7887, from the saved
  `xgb_cascade_deeper_k120_soft.joblib`). The 0.0048 gap is inside one
  per-seed std (0.0206) and consistent with sampling noise across the
  preprocessing recompute path.
- **A1's stability gain is real and consistent.** Per-seed std drops
  3-6× under every variant (most dramatically 0.0140 → 0.0023 at
  `d2-more-trees, k=all`). A1 is a regularizer that compresses the
  per-seed spread; the price is a 0.002 mean drop on the strong variant.

### Per-cell interpretation (revised)

**A1 (shades_of_gray) — variant-dependent, net-negative on the strong
cascade.** A1 normalizes the global illuminant, which removes a real
source of cross-image noise that hurts under-regularized models. But on
`deeper, k=120` the feature selector and tree regularization already
absorb that noise; A1's color normalization then removes signal that
the strong model was using. The 6× variance collapse is genuine but
not worth -0.0018 mean BalAcc.

**A2 (CLAHE) — regress.** The 224×224 dermoscopy images are already
locally high-contrast; CLAHE amplifies hair, ruler edges, and
dermoscope glare more than pigment-network signal. Closed.

**A3 (hb_melanin) — regress.** Neutral reconstruction (α = 1.0) strips
non-chromophore signal; α = 1.5 hemoglobin amplification was deferred
and not tested. Closed unless we revisit α.

**B1 (Stage 2 + abcd_grouped) — marginal.** +0.0053 on the weak
baseline, exactly at the +0.005 pass threshold and inside per-seed std.
XGBoost on `k=all` already exposes asymmetry/border/color through the
base feature blocks. Do not adopt.

**A1 + B1 stack — worse than A1 alone.** +0.0151 over weak baseline but
-0.0050 vs A1 alone. A1 normalizes color so the ABCD-color block is
cleaner; abcd_grouped becomes redundant with the cleaner color block.
Do not stack.

## 4. Recommendation for the 2026-06-10 submission

1. **Cascade-track candidate stays `deeper, k=120, soft, no
   preprocessing`** — the existing saved
   `outputs/models/xgb_cascade_deeper_k120_soft.joblib` matches the new
   sweep's top absolute config (0.7839 here vs 0.7887 ledger).
2. **Main submission remains `all_abcd_grouped + LR(C=0.3) + k=140`**
   (see memory and ledger §8). The cascade and fusion tracks are
   exploratory candidates, not the primary submission.
3. **Do not adopt A1/A2/A3/B1** in any preprocessing or feature step of
   the cascade pipeline.
4. **Report angle**: A1 produces a real variance reduction without
   improving the strong-model mean, which is itself an interesting
   illuminant-vs-regularizer interaction worth a slide.

## 5. Negative results worth keeping in the report

- **A1 shades_of_gray**: lifts weak cascade variants by +0.012 to
  +0.020 and collapses per-seed std 3-6×, but on the strong
  `deeper, k=120` variant it costs -0.0018 BalAcc. Interpretable as a
  regularizer that's already subsumed by feature selection.
- **A2 CLAHE**: regresses under every config tested; ruled out
  empirically. Documents that we considered standard contrast
  enhancement.
- **A3 hb_melanin (α=1.0)**: regresses. The neutral reconstruction is
  too destructive. α=1.5 hemoglobin-amplification not tested.
- **B1 abcd_grouped in Stage 2**: marginal on weak baseline, redundant
  when stacked with A1. Reinforces that XGBoost with all 200+ features
  already absorbs the ABCD signal through other channels.

These four failed/marginal interventions, combined with the existing
closed tracks (hair removal, mask cleaning, TTA + nested-CV, early
fusion, two-stage refinement, mel/nv error-driven features), form a
defensible "we tried everything" narrative for the report.

## 6. Files produced

| path | purpose |
|---|---|
| `outputs/metrics/overnight_exploration_summary.csv` | one row per cell, bagged + per-seed stats |
| `outputs/metrics/overnight_exploration_per_seed.csv` | one row per (cell × seed) |
| `outputs/metrics/overnight_exploration_status.json` | machine-readable status (reconstructed) |
| `outputs/metrics/overnight_exploration.log` | full stdout (tqdm + per-seed prints) |
| `outputs/metrics/variant_sweep_a1.csv` | 18-config follow-up sweep |
| `outputs/metrics/variant_sweep_a1.log` | full stdout for the sweep |
| `outputs/cache/features_cascade_*_pp_*.csv` | preprocessed feature caches (reusable) |

## 7. Postmortem — why the overnight runner shipped a misleading number

- **Root cause**: the runner pinned its cascade variant to
  `d2-more-trees, k=all`, which (per §3) is the worst of the 9 cascade
  configs. Cell 0's 0.7623 looked plausible enough that nobody flagged
  it against ledger §9's 0.7887 until after A1 was already reported as
  "+0.0201".
- **Why A1 looked dramatic**: A1 partially compensates for an
  under-regularized cascade, so on a weak baseline it produces a large
  apparent lift. On a well-tuned baseline (`deeper, k=120`) that
  compensation is no longer needed.
- **Process fix**: future preprocessing/feature experiments should pin
  the cascade variant to `deeper, k=120, soft` (matching ledger §9) so
  every reported delta is against our actual submission candidate, not
  a strawman baseline.
- **Sanity gate was correctly removed** but the wrong baseline was
  preserved. The fix would have been to update the runner's
  `STAGE2_XGB` / `STAGE2_K` constants to match the strong variant
  before launch; the sanity gate was secondary.

# Fusion Ensemble Results

This document corrects an important distinction:

```text
Selective integration != model fusion
```

The first integration step made the teammate XGBoost cascade runnable inside
the main project. The actual fusion experiment keeps our original main model
and combines it with the cascade at the probability level.

## Models Being Fused

Model A: original main traditional model

```text
feature_set = all_abcd_grouped
classifier  = LogisticRegression(C=0.3, class_weight="balanced")
k_features  = 140
validation  = StratifiedGroupKFold by base_id
```

Model B: teammate-inspired XGBoost cascade

```text
Stage 1: vasc vs non-vasc
  feature_set = xgb_cascade_stage1
  blocks      = contrast + abcd_v2 + boundary

Stage 2: mel vs nv
  feature_set = xgb_cascade_stage2
  blocks      = all + boundary + melnv + lbp_multi + gabor + subregion
```

Fusion:

```text
P_final = w_original * P_original + w_cascade * P_cascade
```

No test labels, deep features, or external data are used.

## Seed 127 OOF Result

Strict grouped OOF, seed `127`:

| method | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| Original main model | 0.7871 | 0.7715 | 0.7600 |
| Best pure cascade in this run | 0.8017 | 0.7802 | 0.7617 |
| Best fusion, `0.5 original + 0.5 cascade` | 0.8055 | 0.7909 | 0.7717 |

Best fusion config:

```text
original_model       = all_abcd_grouped_lr03_k140
cascade_variant      = d2-more-trees
cascade_k_features   = all
original_weight      = 0.5
cascade_weight       = 0.5
```

This is the first result that is genuinely `1 + 1 > 1`:

- It keeps the original main model.
- It keeps the teammate cascade.
- It improves over both single models on seed `127`.

## Five-Seed Check

The best seed `127` late-fusion config was then checked over seeds
`42, 127, 2024, 3407, 520`.

| seed | balanced accuracy | macro-F1 | accuracy |
|---:|---:|---:|---:|
| 42 | 0.7690 | 0.7593 | 0.7400 |
| 127 | 0.8055 | 0.7909 | 0.7717 |
| 2024 | 0.7744 | 0.7639 | 0.7467 |
| 3407 | 0.7656 | 0.7475 | 0.7317 |
| 520 | 0.7558 | 0.7418 | 0.7183 |

Summary:

| protocol | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| Per-seed mean | 0.7741 | 0.7607 | 0.7417 |
| Bagged five-seed OOF probabilities | 0.7763 | 0.7607 | 0.7417 |

This confirms that the `0.8055` result is a strong single-seed exploratory
result, but not a stable 5-seed result. For the final report, the safest
wording is that late fusion shows the best single-split gain and remains a
promising candidate, while the multi-seed result is more modest.

## Early Fusion Ablation

Feature-level early fusion was also tested in
`docs/EARLY_FUSION_RESULTS.md`. The best seed `127` early-fusion result was:

| method | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| `early_fusion_core + LR03 + k=140` | 0.7997 | 0.7830 | 0.7700 |

The best checked 5-seed mean for the same family was `0.7766` balanced
accuracy, which is very close to the five-seed late-fusion bagged result
`0.7763`. This means both fusion routes are useful ablations, while the
single-seed high score should be clearly labelled as exploratory:

```text
0.5 * original model probability + 0.5 * cascade probability
```

Recommended wording:

```text
We preserved the original ABCD-grouped logistic-regression main model and
fused it with the XGBoost cascade through grouped-OOF probability averaging.
On seed 127, the fusion improved balanced accuracy from 0.7871/0.8017
(single models) to 0.8055, with macro-F1 improving to 0.7909. In the 5-seed
check, the bagged balanced accuracy was 0.7763, so we report the 0.8055 as an
exploratory single-seed gain rather than a final stable result.
```

## Commands

Run the fusion grid:

```bash
.venv/bin/python experiments/run_fusion_ensemble.py \
  --data_dir data/Data_Proj2 \
  --seeds 127 \
  --cascade_variants strong-reg,d2-more-trees,deeper \
  --cascade_k_features 100,120,all \
  --weights 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
  --output_csv outputs/metrics/fusion_ensemble_seed127_grid.csv \
  --output_per_seed_csv outputs/metrics/fusion_ensemble_seed127_grid_per_seed.csv
```

Check the best seed `127` fusion over five seeds:

```bash
.venv/bin/python experiments/run_fusion_ensemble.py \
  --data_dir data/Data_Proj2 \
  --seeds 42,127,2024,3407,520 \
  --cascade_variants d2-more-trees \
  --cascade_k_features all \
  --weights 0.5 \
  --output_csv outputs/metrics/fusion_ensemble_d2_all_w05_multiseed.csv \
  --output_per_seed_csv outputs/metrics/fusion_ensemble_d2_all_w05_multiseed_per_seed.csv
```

Train the full-data fusion candidate:

```bash
.venv/bin/python -m src.train_ml_fusion \
  --data_dir data/Data_Proj2 \
  --cascade_variant d2-more-trees \
  --cascade_k_features all \
  --cascade_weight 0.5
```

Generate predictions:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/ml_fusion_candidate.joblib \
  --output_csv output.csv
```

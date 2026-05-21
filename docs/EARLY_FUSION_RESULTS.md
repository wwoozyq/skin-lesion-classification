# Early Fusion Results

This document records the feature-level fusion experiment added after the
probability-level fusion experiment.

## What Early Fusion Means

Early fusion combines information before model training:

```text
image + mask
  -> color / shape / texture / ABCD / boundary / mel-nv / LBP / Gabor / subregion features
  -> one concatenated feature vector
  -> one classifier
```

This is different from late fusion:

```text
model A probability + model B probability -> weighted average -> final class
```

Both are valid fusion strategies, but they answer different questions. Early
fusion asks whether all handcrafted features are better when learned by a
single classifier. Late fusion asks whether two models with different error
patterns can complement each other.

## Feature Sets

`early_fusion_core`:

```text
color + shape + texture
+ abcd_grouped
+ boundary
+ melnv
+ lbp_multi
+ gabor
+ subregion
```

Number of features: `414`.

`early_fusion_full`:

```text
color + shape + texture
+ contrast
+ abcd_v2
+ boundary
+ abcd_grouped
+ melnv
+ lbp_multi
+ gabor
+ subregion
```

Number of features: `475`.

## Seed 127 Search

Strict grouped OOF, seed `127`, no image-level leakage.

Best early-fusion result:

| feature set | classifier | k | postprocess | balanced accuracy | macro-F1 | accuracy |
|---|---|---:|---|---:|---:|---:|
| early_fusion_core | LR03 | 140 | class-weight argmax | 0.7997 | 0.7830 | 0.7700 |
| early_fusion_core | LR03 | 100 | class-weight argmax | 0.7981 | 0.7786 | 0.7633 |
| early_fusion_core | LR03 | 100 | plain argmax | 0.7836 | 0.7704 | 0.7550 |
| early_fusion_full | HGB | 180 | plain argmax | 0.7784 | 0.7789 | 0.7600 |

The best single-seed result is close to `0.80` balanced accuracy, but it uses
OOF-tuned class weights. Therefore it should be described as exploratory, not
as a frozen final result.

## Multi-Seed Check

The most promising early-fusion family was checked across seeds
`42, 127, 2024, 3407, 520`.

| config | postprocess | mean balanced accuracy | std | mean macro-F1 | mean accuracy |
|---|---|---:|---:|---:|---:|
| early_fusion_core + LR03 + k=100 | class-weight argmax | 0.7766 | 0.0144 | 0.7618 | 0.7450 |
| early_fusion_core + LR03 + k=140 | class-weight argmax | 0.7730 | 0.0160 | 0.7584 | 0.7440 |
| early_fusion_core + LR03 + k=100 | plain argmax | 0.7652 | 0.0145 | 0.7555 | 0.7373 |
| early_fusion_core + LR03 + k=140 | plain argmax | 0.7614 | 0.0114 | 0.7515 | 0.7390 |

This suggests that early fusion helps as an ablation and as evidence of broad
feature exploration, but the high seed `127` result is not yet stable enough
to replace the late-fusion candidate.

## Comparison With Other Traditional Models

| method | protocol | balanced accuracy | macro-F1 | accuracy |
|---|---|---:|---:|---:|
| Original `all_abcd_grouped + LR03` | seed `127` grouped OOF | 0.7871 | 0.7715 | 0.7600 |
| Pure XGBoost cascade | seed `127` grouped OOF | 0.8017 | 0.7802 | 0.7617 |
| Late fusion, `0.5 original + 0.5 cascade` | seed `127` grouped OOF | 0.8055 | 0.7909 | 0.7717 |
| Early fusion, best exploratory | seed `127` grouped OOF | 0.7997 | 0.7830 | 0.7700 |
| Early fusion, best 5-seed mean | grouped OOF | 0.7766 | 0.7618 | 0.7450 |

Current interpretation:

```text
Early fusion was tested and is a legitimate feature-level fusion baseline.
It nearly reaches 0.80 balanced accuracy on seed 127, but it is less stable
than the late-fusion candidate and does not clearly beat the current best.
For presentation, it is best used as an ablation showing that simply
concatenating all handcrafted features is not always better than preserving
complementary models and fusing probabilities.
```

## Commands

Seed `127` broad search:

```bash
.venv/bin/python experiments/run_early_fusion_search.py \
  --data_dir data/Data_Proj2 \
  --feature_sets early_fusion_core,early_fusion_full \
  --classifiers lr03,et,hgb,xgb_strong_reg,xgb_d2_more_trees \
  --k_features 100,140,180,220,all \
  --seeds 127 \
  --output_csv outputs/metrics/early_fusion_seed127.csv \
  --n_jobs 2
```

Light multi-seed check:

```bash
.venv/bin/python experiments/run_early_fusion_search.py \
  --data_dir data/Data_Proj2 \
  --feature_sets early_fusion_core \
  --classifiers lr03 \
  --k_features 100,140 \
  --seeds 42,127,2024,3407,520 \
  --output_csv outputs/metrics/early_fusion_lr03_multiseed.csv \
  --n_jobs 2
```

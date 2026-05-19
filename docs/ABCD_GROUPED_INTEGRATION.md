# ABCD Grouped Optimization Integration

This note records what was selectively absorbed from the teammate branch
`origin/abcd-grouped-optimization`.

## Why Not Merge the Branch Directly

The branch contains useful ideas, but it also commits generated files under
`outputs/` and rewrites core training code. Therefore, the safe integration
strategy is to keep the current modular pipeline and absorb only the reusable
parts as controlled experiments.

Integrated:

- extended ABCD-like color and geometry descriptors
- optional `mel` probability threshold support in prediction
- an experiment script for `mel` threshold and hierarchical classification
- an `lr03` classifier option, matching the best integrated result

Not directly merged:

- generated PNG/CSV/report files under `outputs/`
- large rewrite of `src/train_ml.py`
- direct modification of base `features_color.py` and `features_shape.py`

## New Feature Set

The new module is:

```text
src/features_abcd_grouped.py
```

It adds a separate feature set instead of changing the base color/shape
extractors:

```text
abcd_grouped
all_abcd_grouped
final_abcd_grouped
```

The best result came from `all_abcd_grouped`, which combines baseline color,
shape, texture, and the new grouped ABCD descriptors.

## Main Seed-127 Result

Previous best strict grouped-CV model:

```text
all_boundary + LogisticRegression(C=1.0) + top100
accuracy = 0.7233
macro-F1 = 0.7454
balanced accuracy = 0.7535
```

New best integrated candidate without threshold:

```text
all_abcd_grouped + LogisticRegression(C=0.3) + top140
accuracy = 0.7600
macro-F1 = 0.7715
balanced accuracy = 0.7871
```

Threshold ablation:

```text
all_abcd_grouped + LogisticRegression(C=0.3) + top140 + mel_threshold=0.45
accuracy = 0.7600
macro-F1 = 0.7730
balanced accuracy = 0.7910
```

The threshold version is slightly better on seed 127, but it is less clean as a
main result because the threshold is selected from the same OOF probability
diagnostics. The no-threshold argmax version is therefore the safer candidate
main model.

## Stability Check

Five grouped split seeds:

```text
42, 127, 2024, 3407, 520
```

No-threshold model:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_abcd_grouped` + `lr03` + top140 | 0.7283 | 0.7435 | 0.0271 | 0.7512 |

Fixed-threshold model:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_abcd_grouped` + `lr03` + top140 + `mel_threshold=0.45` | 0.7257 | 0.7426 | 0.0284 | 0.7520 |

Compared with the previous stable baseline:

| model | mean accuracy | mean macro-F1 | std macro-F1 | mean balanced accuracy |
|---|---:|---:|---:|---:|
| `all_boundary` + LR + top100 | 0.7077 | 0.7242 | 0.0143 | 0.7301 |

Conclusion:

The new feature set improves the mean strict grouped-CV performance, but it is
less stable across seeds. In the report, present this as a performance-oriented
candidate plus a stability caveat.

## Commands

Integration experiment:

```bash
.venv/bin/python experiments/run_abcd_grouped_integration.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_boundary,all_abcd_grouped,final_abcd_grouped \
  --classifiers lr,rf \
  --k_features 80,100,140 \
  --output_csv outputs/metrics/abcd_grouped_integration_lr_rf.csv
```

Train the safer candidate model:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw
```

Train the threshold ablation:

```bash
.venv/bin/python -m src.train_ml \
  --data_dir data/Data_Proj2 \
  --feature_set all_abcd_grouped \
  --classifier lr03 \
  --k_features 140 \
  --cv grouped \
  --mask_mode raw \
  --mel_threshold 0.45
```

Stability checks:

```bash
.venv/bin/python experiments/run_stability.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_abcd_grouped \
  --classifiers lr03 \
  --k_features 140 \
  --mask_modes raw \
  --seeds 42,127,2024,3407,520 \
  --output_csv outputs/metrics/stability_abcd_grouped_argmax.csv

.venv/bin/python experiments/run_stability.py \
  --data_dir data/Data_Proj2 \
  --feature_sets all_abcd_grouped \
  --classifiers lr03 \
  --k_features 140 \
  --mask_modes raw \
  --seeds 42,127,2024,3407,520 \
  --mel_threshold 0.45 \
  --output_csv outputs/metrics/stability_abcd_grouped_melthr045.csv
```

## Reporting Recommendation

Use this as a strong traditional-method improvement:

```text
Selective integration of teammate ABCD grouped features improved seed-127
strict grouped macro-F1 from 0.7454 to 0.7715. A mel-threshold ablation reached
0.7730, but the no-threshold model is preferred as the main candidate because it
has cleaner decision logic and slightly better multi-seed mean performance.
```

Also mention the limitation:

```text
The new feature set improves average performance but increases split-to-split
variance, so the final report should include both the selected split result and
the five-seed stability result.
```

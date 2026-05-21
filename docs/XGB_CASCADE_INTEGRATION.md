# XGBoost Cascade Integration

This document records the selectively integrated work from teammate branch
`origin/model`. The branch was not merged directly because it would remove
existing deep-learning scripts and project documentation. Instead, the useful
traditional ML ideas were ported into the main project.

## What Was Integrated

New traditional feature blocks:

- `lbp_multi`: multi-scale local binary pattern texture features.
- `gabor`: oriented Gabor texture response statistics.
- `subregion`: quadrant-level LAB color and area asymmetry features.

New feature aliases in `src/features.py`:

- `all_lbp_multi`
- `all_gabor`
- `all_subregion`
- `all_texture_plus`
- `all_boundary_melnv_texture_plus`
- `xgb_cascade_stage1`
- `xgb_cascade_stage2`

The feature extractor also accepts `+` expressions, for example:

```text
all+boundary+melnv+lbp_multi+gabor+subregion
```

New model support:

- `xgb`, `xgb_strong_reg`, and `xgb_d2_more_trees` in `src.train_ml`.
- Strict grouped-OOF cascade search in `experiments/run_xgb_cascade_search.py`.
- Full-data cascade candidate training in `src.train_ml_cascade`.
- Cascade bundle inference support in `run.py`.

## Cascade Design

Stage 1 detects vascular lesions:

```text
target      = vasc vs non-vasc
feature_set = xgb_cascade_stage1
blocks      = contrast + abcd_v2 + boundary
model       = XGBoost
```

Stage 2 separates melanoma and nevus:

```text
target      = mel vs nv
feature_set = xgb_cascade_stage2
blocks      = all + boundary + melnv + lbp_multi + gabor + subregion
model       = XGBoost
```

The final prediction uses soft probability fusion:

```text
P(vasc) = P1(vasc)
P(mel)  = (1 - P1(vasc)) * P2(mel | non-vasc)
P(nv)   = (1 - P1(vasc)) * P2(nv  | non-vasc)
```

This remains a traditional ML pipeline: handcrafted image features plus
XGBoost. No deep-learning features or external data are used.

## Reproduced Results

Reproduction was run in an isolated worktree at:

```text
/Users/wlm/Desktop/HW/skin-lesion-model-repro
```

Data:

```text
/Users/wlm/Desktop/HW/skin-lesion-classification/data/Data_Proj2
```

The validation protocol is strict `StratifiedGroupKFold`, grouped by original
lesion `base_id`, so augmented versions of the same lesion do not cross
train/validation folds.

### v7 Cascade

| setting | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| bagged cascade soft | 0.7771 | 0.7524 | 0.7333 |
| bagged flat 3-class | 0.7570 | 0.7497 | 0.7233 |

### v8 Final Sweep

Best single-seed result at seed `127`:

| setting | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| strong-reg, k=120, soft/hard@0.4 | 0.8006 | 0.7792 | 0.7600 |

After selective integration into this branch, a compact smoke run with the
same seed and the `strong-reg, k=120, soft` configuration produced:

| setting | balanced accuracy | macro-F1 | accuracy |
|---|---:|---:|---:|
| integrated smoke, seed `127` | 0.8017 | 0.7802 | 0.7617 |

Best 5-seed bagged result:

| setting | bagged balanced accuracy | bagged macro-F1 | bagged accuracy |
|---|---:|---:|---:|
| deeper, k=120, soft | 0.7887 | 0.7655 | 0.7500 |

Interpretation:

- The cascade direction is valuable because single-seed balanced accuracy can
  cross `0.80`.
- The stronger claim is not "accuracy reached 80%"; ordinary accuracy remains
  around `0.75-0.76`.
- The more reliable multi-seed result is about `0.789` balanced accuracy.
- This is a promising candidate branch, but the current stable main result
  should not be replaced until the integrated script reproduces the same
  result in this branch.

## Commands

Install dependencies:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

On this local macOS environment, XGBoost may need the OpenMP runtime already
bundled inside the scikit-learn wheel:

```bash
export DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs:$DYLD_LIBRARY_PATH
```

Run a compact cascade search:

```bash
.venv/bin/python experiments/run_xgb_cascade_search.py \
  --data_dir data/Data_Proj2 \
  --seeds 42,127,2024,3407,520 \
  --stage2_variants strong-reg,d2-more-trees,deeper \
  --k_features 100,120,all \
  --output_csv outputs/metrics/xgb_cascade_search.csv \
  --output_per_seed_csv outputs/metrics/xgb_cascade_search_per_seed.csv
```

Train the full-data cascade candidate:

```bash
.venv/bin/python -m src.train_ml_cascade \
  --data_dir data/Data_Proj2 \
  --stage2_variant deeper \
  --stage2_k_features 120
```

Generate predictions from the cascade candidate:

```bash
.venv/bin/python run.py \
  --input_dir data/Data_Proj2 \
  --model_path outputs/models/xgb_cascade_deeper_k120_soft.joblib \
  --output_csv output.csv
```

## Reporting Wording

Use this strict wording in presentation/report:

```text
We integrated an exploratory XGBoost cascade with additional handcrafted
texture/asymmetry features. Under strict grouped OOF validation, the integrated
single-seed smoke run reached 0.8017 balanced accuracy, while the more
conservative reproduced 5-seed bagged result was 0.7887 balanced accuracy.
Ordinary accuracy remained around 0.75-0.76, so we report this as an
improvement in class-balanced recognition rather than claiming 80% overall
accuracy.
```

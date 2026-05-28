# Lesion-Hard Error Analysis

Diagnostic study of the lesions the LR main model misclassifies on **all
three** of (orig, aug1, aug2) at seed `127`. These are the residual errors
that fusion, TTA, and nested-CV weight selection could not move (see
ledger sections 11 and 12).

## 1. Source

Main-line OOF predictions:

```text
outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv
```

A `base_id` is counted as **all-augs-wrong** when its 3 versions are all
misclassified. Result: 25 of the 200 lesions (12.5%).

By true class:

| true | n lesions |
|---|---:|
| mel | 13 |
| nv | 11 |
| vasc | 1 |

## 2. Visualization

Regenerate with:

```bash
.venv/bin/python experiments/visualize_lesion_hard.py
```

Outputs:

```text
outputs/figures/lesion_hard_overview.png   # 5x5 grid, one tile per lesion
outputs/figures/lesion_hard_detail.png     # one row per lesion, 3 versions
```

The overview is the working figure for the report.

## 3. Bucket Decomposition

The 25 lesions group into 5 clinically meaningful buckets. Counts add to
more than 25 because a few lesions fit two descriptions (for example a
hair-occluded mel that also looks benign):

| bucket | n | example ids | algorithmically fixable? |
|---|---:|---|---|
| `mel -> nv` benign-looking phenotype | 10 | 51, 55, 57, 73, 105, 116, 124, 175, 186, 200 | no |
| `nv -> mel` atypical-looking phenotype | 5 | 26, 33, 90, 120, 197 | no |
| Vascular confusion from red regions | 4 | 7, 71, 185, 189 | weak |
| Hair occlusion | 2 | 24, 154 | yes (preprocessing) |
| Other / single true-vasc miss | 4 | 23, 40, 76, 84 | mixed |

### 3.1 mel -> nv benign-looking phenotype (10 lesions)

These are labeled mel but visually appear as small, fairly uniform brown
nevi without the asymmetry, border irregularity, color variation, or
diameter that the ABCD-grouped feature set is designed to read. The model
correctly recovers the visual phenotype (nv) and is wrong on the
clinical label. In dermoscopy these correspond to early or in-situ
melanoma where ABCD on its own is not sufficient and dermoscopic
structures (atypical pigment network, regression, blue-white veil) are
needed for diagnosis.

### 3.2 nv -> mel atypical-looking phenotype (5 lesions)

The mirror case. These are labeled nv but exhibit heterogeneous color,
irregular borders, or dark central focus. A dermatologist would likely
call them dysplastic nevi. The model reads the same visual asymmetry and
predicts mel, which is the safer clinical call but counts as wrong here.

### 3.3 Vascular confusion (4 lesions)

Lesions with red or dark-red regions that trigger the cascade stage-1
vasc detector. Id 185 (mel) in particular has extensive vascular-looking
red areas and is essentially unrecoverable from color features alone.

### 3.4 Hair occlusion (2 lesions)

Id 24 and id 154 are heavily covered by dark terminal hairs. The
segmentation mask traces around or through the hair, so color and
texture features pick up hair contribution rather than lesion content.
This is a preprocessing failure, not a feature failure.

### 3.5 Other (4 lesions)

Id 84 is the only misclassified true-vasc lesion (the dataset has 30
vasc lesions, so each one matters). Id 23, 40, 76 sit at category
boundaries where multiple buckets apply.

## 4. Notable Negative Finding

We expected to find label errors in this set. We did not find any. The
mel-labeled lesions that look like nv are phenotypically consistent with
early melanoma; the nv-labeled lesions that look like mel are
phenotypically consistent with dysplastic nevi. Therefore relabel-based
gains are not on the table here.

## 5. Implication for the 0.78 BalAcc Ceiling

Of the 25 all-augs-wrong lesions, only ~2 are fixable with traditional
methods (hair removal preprocessing). The remaining ~23 are biological
mel-nv phenotype overlap or extreme vascular-color confusion, which
require either dermoscopic structural features or a deep model to break.

This grounds the bagged 0.78 balanced accuracy ceiling observed across
sections 9-12 of the ledger as a property of the data plus the
traditional feature space, not a property of weight tuning or candidate
selection.

## 6. Report Role

```text
Failure analysis for the traditional ML main line. Reframes the residual
20-25% error as biological phenotype overlap rather than algorithmic
weakness, and motivates the deep learning extension as the principled
route past the 0.78 ceiling.
```

## 7. Related Documents

- `docs/EXPERIMENT_LEDGER.md` section 5 (Error Visualization, aggregate)
- `docs/EXPERIMENT_LEDGER.md` section 12 (TTA + nested-CV, confirms ceiling)
- `docs/FUSION_ENSEMBLE_RESULTS.md` (sets up the candidate that produces these errors)
- `docs/DEEP_LEARNING_EXTENSION_RESULTS.md` (the route past the ceiling)

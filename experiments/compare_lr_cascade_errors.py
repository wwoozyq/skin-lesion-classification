"""Compute cascade OOF on seed 127 with the SAME folds as the LR main model,
then cross-tab error overlap LR vs cascade vs fusion."""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.features import build_feature_table
from src.utils import base_id_from_image_id

warnings.filterwarnings("ignore")

DATA_DIR = ROOT / "data" / "Data_Proj2"
SEED = 127
N_SPLITS = 5

STAGE1_FS = "xgb_cascade_stage1"
STAGE1_K = 100
STAGE1_XGB = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                  subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)

STAGE2_FS = "xgb_cascade_stage2"
STAGE2_K = 120
STAGE2_XGB = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)


def build_pipe(xgb_kwargs, n_classes, k, n_feat, seed):
    steps = [("scaler", StandardScaler())]
    if k != "all" and int(k) < n_feat:
        steps.append(("select", SelectKBest(f_classif, k=int(k))))
    obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
    em = "logloss" if n_classes == 2 else "mlogloss"
    steps.append(("clf", XGBClassifier(**xgb_kwargs, objective=obj, eval_metric=em,
                                       tree_method="hist", random_state=seed,
                                       n_jobs=-1, verbosity=0)))
    return Pipeline(steps)


def main():
    labels = load_labels(DATA_DIR).copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    labels["base_id"] = labels["image_id"].apply(base_id_from_image_id)
    y = labels["dx"].to_numpy()
    groups = labels["base_id"].to_numpy()

    print("Building feature tables...")
    t1 = build_feature_table(DATA_DIR, image_ids=labels["image_id"], feature_set=STAGE1_FS, mask_mode="raw")
    t2 = build_feature_table(DATA_DIR, image_ids=labels["image_id"], feature_set=STAGE2_FS, mask_mode="raw")
    t1["image_id"] = t1["image_id"].astype(str)
    t2["image_id"] = t2["image_id"].astype(str)
    X1 = labels.merge(t1, on="image_id").drop(columns=["image_id", "dx", "base_id"])
    X2 = labels.merge(t2, on="image_id").drop(columns=["image_id", "dx", "base_id"])
    print(f"  stage1 features: {X1.shape}, stage2 features: {X2.shape}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(cv.split(X1, y, groups=groups))

    p_vasc = np.zeros(len(y), dtype=np.float32)
    stage2_proba = np.zeros((len(y), 2), dtype=np.float32)
    fold_idx = np.full(len(y), -1, dtype=int)

    for f, (tr, va) in enumerate(splits):
        print(f"Fold {f}: {len(tr)} train / {len(va)} val")
        fold_idx[va] = f

        ys1 = (y == "vasc").astype(int)
        s1 = build_pipe(STAGE1_XGB, 2, STAGE1_K, X1.shape[1], SEED)
        s1.fit(X1.iloc[tr], ys1[tr], clf__sample_weight=compute_sample_weight("balanced", ys1[tr]))
        p_vasc[va] = s1.predict_proba(X1.iloc[va])[:, 1]

        mn_mask = y[tr] != "vasc"
        tr_mn = tr[mn_mask]
        ys2 = (y[tr_mn] == "nv").astype(int)
        s2 = build_pipe(STAGE2_XGB, 2, STAGE2_K, X2.shape[1], SEED)
        s2.fit(X2.iloc[tr_mn], ys2, clf__sample_weight=compute_sample_weight("balanced", ys2))
        stage2_proba[va] = s2.predict_proba(X2.iloc[va])

    label_idx = {l: i for i, l in enumerate(LABELS)}
    cascade_proba = np.zeros((len(y), 3), dtype=np.float32)
    cascade_proba[:, label_idx["vasc"]] = p_vasc
    cascade_proba[:, label_idx["mel"]] = (1.0 - p_vasc) * stage2_proba[:, 0]
    cascade_proba[:, label_idx["nv"]] = (1.0 - p_vasc) * stage2_proba[:, 1]
    cascade_pred = np.array(LABELS)[cascade_proba.argmax(axis=1)]

    out = pd.DataFrame({
        "image_id": labels["image_id"],
        "base_id": labels["base_id"],
        "dx": y,
        "fold": fold_idx,
        "cascade_pred": cascade_pred,
        "cascade_correct": cascade_pred == y,
        "cascade_p_mel": cascade_proba[:, label_idx["mel"]],
        "cascade_p_nv": cascade_proba[:, label_idx["nv"]],
        "cascade_p_vasc": cascade_proba[:, label_idx["vasc"]],
    })
    out_path = ROOT / "outputs/metrics/cascade_seed127_oof_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved cascade OOF: {out_path}")

    lr = pd.read_csv(ROOT / "outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv")
    lr["image_id"] = lr["image_id"].astype(str)
    out["image_id"] = out["image_id"].astype(str)
    merged = lr.merge(out[["image_id", "cascade_pred", "cascade_correct",
                           "cascade_p_mel", "cascade_p_nv", "cascade_p_vasc"]], on="image_id")
    merged["lr_correct"] = merged["correct"]
    merged["lr_pred"] = merged["pred"]

    n = len(merged)
    both_right = ((merged.lr_correct) & (merged.cascade_correct)).sum()
    both_wrong = ((~merged.lr_correct) & (~merged.cascade_correct)).sum()
    only_lr_right = ((merged.lr_correct) & (~merged.cascade_correct)).sum()
    only_cas_right = ((~merged.lr_correct) & (merged.cascade_correct)).sum()

    print(f"\n=== Error overlap on seed {SEED} (n={n}) ===")
    print(f"                 Cascade right    Cascade wrong")
    print(f"LR right         {both_right:>6}           {only_lr_right:>6}")
    print(f"LR wrong         {only_cas_right:>6}           {both_wrong:>6}")
    print(f"\nLR errors total:        {(~merged.lr_correct).sum()}")
    print(f"Cascade errors total:   {(~merged.cascade_correct).sum()}")
    print(f"Common errors:          {both_wrong}")
    print(f"Errors only LR makes:   {only_lr_right}")
    print(f"Errors only Cas makes:  {only_cas_right}")
    print(f"Error overlap rate:     {both_wrong / max(1,(~merged.lr_correct).sum()):.1%} of LR errors")
    print(f"                        {both_wrong / max(1,(~merged.cascade_correct).sum()):.1%} of cascade errors")

    print(f"\n=== Among 'cascade right but LR wrong' ({only_cas_right}) — what classes does cascade rescue? ===")
    rescued = merged[(~merged.lr_correct) & (merged.cascade_correct)]
    print(rescued.groupby(["dx", "lr_pred"]).size().to_string())

    print(f"\n=== Among 'LR right but cascade wrong' ({only_lr_right}) — where does cascade lose? ===")
    lost = merged[(merged.lr_correct) & (~merged.cascade_correct)]
    print(lost.groupby(["dx", "cascade_pred"]).size().to_string())

    lr_proba = pd.read_csv(ROOT / "outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv")
    lr_oof_path = ROOT / "outputs/metrics/ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv"
    if "p_mel" in lr_proba.columns:
        for cls in ["mel", "nv", "vasc"]:
            merged[f"lr_p_{cls}"] = lr_proba[f"p_{cls}"]
    else:
        print("\nNote: LR OOF doesn't include probabilities → can't compute fusion overlap directly.")
        print("Estimating fusion via 0.5*cascade_proba + 0.5*onehot_lr_pred is not equivalent.")
        return

    fusion_p = 0.5 * lr_proba[["p_mel", "p_nv", "p_vasc"]].values + 0.5 * cascade_proba[
        :, [label_idx["mel"], label_idx["nv"], label_idx["vasc"]]
    ]
    fusion_pred = np.array(["mel", "nv", "vasc"])[fusion_p.argmax(axis=1)]
    fusion_correct = fusion_pred == y
    print(f"\n=== Fusion (0.5 LR + 0.5 cascade) ===")
    print(f"Fusion errors: {(~fusion_correct).sum()}")
    print(f"  fusion right where both wrong:        {((~merged.lr_correct) & (~merged.cascade_correct) & fusion_correct).sum()}")
    print(f"  fusion wrong where both right:        {((merged.lr_correct) & (merged.cascade_correct) & ~fusion_correct).sum()}")
    print(f"  fusion saved by LR (cas wrong, lr right, fusion right):  {((merged.lr_correct) & (~merged.cascade_correct) & fusion_correct).sum()}")
    print(f"  fusion saved by cas (lr wrong, cas right, fusion right): {((~merged.lr_correct) & (merged.cascade_correct) & fusion_correct).sum()}")


if __name__ == "__main__":
    main()

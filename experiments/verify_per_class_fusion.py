"""Verify Path A: per-class weighted fusion of LR + XGBoost cascade.

Pipeline:
  1. Generate LR OOF probabilities (seed 127, same 5-fold StratifiedGroupKFold).
  2. Reuse cascade OOF probabilities saved in
     outputs/metrics/cascade_seed127_oof_predictions.csv.
  3. Grid-search per-class fusion weights on seed 127 (in-sample upper bound).
  4. Nested 5-fold weight selection (honest out-of-sample estimate).
  5. Compare against fixed 0.5/0.5 baseline and against the LR-only main model.

Usage:
  DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \
  .venv/bin/python experiments/verify_per_class_fusion.py
"""

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

LR_FEATURE_SET = "all_abcd_grouped"
LR_K = 140


def lr_oof_proba(seed):
    labels = load_labels(DATA_DIR).copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    labels["base_id"] = labels["image_id"].apply(base_id_from_image_id)
    y = labels["dx"].to_numpy()
    groups = labels["base_id"].to_numpy()

    print(f"[LR] Building feature table {LR_FEATURE_SET}...")
    fdf = build_feature_table(DATA_DIR, image_ids=labels["image_id"],
                              feature_set=LR_FEATURE_SET, mask_mode="raw")
    fdf["image_id"] = fdf["image_id"].astype(str)
    X = labels.merge(fdf, on="image_id").drop(columns=["image_id", "dx", "base_id"])
    print(f"[LR] features: {X.shape}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups=groups))

    proba = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    fold_idx = np.full(len(y), -1, dtype=int)
    for f, (tr, va) in enumerate(splits):
        fold_idx[va] = f
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(LR_K, X.shape[1]))),
            ("clf", LogisticRegression(C=0.3, class_weight="balanced",
                                       max_iter=4000, random_state=seed)),
        ])
        pipe.fit(X.iloc[tr], y[tr])
        # Re-order LR's class order to match LABELS
        cls_order = list(pipe.named_steps["clf"].classes_)
        idx_map = [cls_order.index(c) for c in LABELS]
        proba[va] = pipe.predict_proba(X.iloc[va])[:, idx_map]
    return labels, y, groups, fold_idx, proba, splits


def fuse(lr_p, cas_p, weights):
    """weights = dict label -> w_LR ; cascade weight = (1 - w_LR)."""
    w = np.array([weights[l] for l in LABELS])
    fused = lr_p * w + cas_p * (1.0 - w)
    fused = fused / fused.sum(axis=1, keepdims=True)
    return fused


def metrics(y, proba):
    pred = np.array(LABELS)[proba.argmax(axis=1)]
    return {
        "acc": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y, pred),
        "n_err": int((pred != y).sum()),
    }


def grid_search_weights(y, lr_p, cas_p, indices, step=0.1):
    """Grid over per-class weights on the given index set, return best (weights, bal_acc)."""
    grid = np.arange(0.0, 1.0001, step)
    best = None
    for w_mel, w_nv, w_vasc in product(grid, grid, grid):
        weights = {"mel": w_mel, "nv": w_nv, "vasc": w_vasc}
        fused = fuse(lr_p[indices], cas_p[indices], weights)
        ba = balanced_accuracy_score(y[indices], np.array(LABELS)[fused.argmax(axis=1)])
        if (best is None) or (ba > best[1]):
            best = (weights, ba)
    return best


def main():
    cas_csv = ROOT / "outputs/metrics/cascade_seed127_oof_predictions.csv"
    if not cas_csv.exists():
        raise FileNotFoundError(f"Run experiments/compare_lr_cascade_errors.py first; missing {cas_csv}")

    labels, y, groups, fold_idx, lr_p, splits = lr_oof_proba(SEED)
    cas_df = pd.read_csv(cas_csv)
    cas_df["image_id"] = cas_df["image_id"].astype(str)

    merged = labels[["image_id"]].merge(cas_df, on="image_id", how="left")
    cas_p = merged[["cascade_p_mel", "cascade_p_nv", "cascade_p_vasc"]].to_numpy()
    label_idx = {l: i for i, l in enumerate(LABELS)}
    cas_p_aligned = np.zeros_like(cas_p)
    cas_p_aligned[:, label_idx["mel"]] = merged["cascade_p_mel"].values
    cas_p_aligned[:, label_idx["nv"]] = merged["cascade_p_nv"].values
    cas_p_aligned[:, label_idx["vasc"]] = merged["cascade_p_vasc"].values

    print("\n=== Baselines on seed 127 OOF ===")
    print(f"  LR only          : {metrics(y, lr_p)}")
    print(f"  Cascade only     : {metrics(y, cas_p_aligned)}")
    fused_5050 = fuse(lr_p, cas_p_aligned, {"mel": 0.5, "nv": 0.5, "vasc": 0.5})
    print(f"  0.5/0.5 fusion   : {metrics(y, fused_5050)}")

    print("\n=== Grid: in-sample upper bound (overfits seed 127) ===")
    best_in, best_in_ba = grid_search_weights(y, lr_p, cas_p_aligned, np.arange(len(y)), step=0.1)
    fused_in = fuse(lr_p, cas_p_aligned, best_in)
    print(f"  Best in-sample w : {best_in}")
    print(f"  Metrics          : {metrics(y, fused_in)}  (BalAcc grid={best_in_ba:.4f})")

    print("\n=== Nested CV: pick weights on inner OOF, eval on outer fold ===")
    nested_pred = np.empty(len(y), dtype=object)
    chosen_weights = []
    for f, (tr, va) in enumerate(splits):
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + f)
        inner_oof_lr = np.zeros((len(tr), len(LABELS)))
        inner_oof_cas = np.zeros((len(tr), len(LABELS)))
        labels_tr = labels.iloc[tr].reset_index(drop=True)
        y_tr = y[tr]
        groups_tr = groups[tr]
        X_lr_tr = lr_p[tr]
        X_cas_tr = cas_p_aligned[tr]
        for itr, iva in inner_cv.split(np.zeros(len(tr)), y_tr, groups=groups_tr):
            inner_oof_lr[iva] = X_lr_tr[iva]
            inner_oof_cas[iva] = X_cas_tr[iva]
        best_w, _ = grid_search_weights(y_tr, X_lr_tr, X_cas_tr,
                                        np.arange(len(tr)), step=0.1)
        chosen_weights.append(best_w)
        fused_va = fuse(lr_p[va], cas_p_aligned[va], best_w)
        nested_pred[va] = np.array(LABELS)[fused_va.argmax(axis=1)]
        print(f"  Fold {f}: weights = {best_w}")

    nested_correct = (nested_pred == y)
    print(f"\n  Nested fusion seed 127:")
    print(f"    Acc      = {nested_correct.mean():.4f}")
    print(f"    Macro-F1 = {f1_score(y, nested_pred, average='macro'):.4f}")
    print(f"    Bal-Acc  = {balanced_accuracy_score(y, nested_pred):.4f}")
    print(f"    Errors   = {(~nested_correct).sum()}")

    print("\n=== Mean weights across folds (sanity check) ===")
    mean_w = {l: np.mean([w[l] for w in chosen_weights]) for l in LABELS}
    std_w = {l: np.std([w[l] for w in chosen_weights]) for l in LABELS}
    for l in LABELS:
        print(f"  w_{l}: mean={mean_w[l]:.3f} std={std_w[l]:.3f}")

    out_path = ROOT / "outputs/metrics/per_class_fusion_seed127.csv"
    pd.DataFrame({
        "image_id": labels["image_id"],
        "dx": y,
        "lr_pred": np.array(LABELS)[lr_p.argmax(axis=1)],
        "cas_pred": np.array(LABELS)[cas_p_aligned.argmax(axis=1)],
        "fused_5050_pred": np.array(LABELS)[fused_5050.argmax(axis=1)],
        "fused_nested_pred": nested_pred,
        "lr_p_mel": lr_p[:, label_idx["mel"]],
        "lr_p_nv": lr_p[:, label_idx["nv"]],
        "lr_p_vasc": lr_p[:, label_idx["vasc"]],
    }).to_csv(out_path, index=False)
    print(f"\nSaved per-class fusion OOF: {out_path}")


if __name__ == "__main__":
    main()

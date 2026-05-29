"""Path A': structural 2-axis fusion of LR + XGBoost cascade.

Decomposition:
  axis 1 (vasc gating): P_vasc_fused = w1 * P_lr_vasc + (1-w1) * P_cas_vasc
  axis 2 (mel|non-vasc): P_mel_cond_fused = w2 * P_lr_mel_cond + (1-w2) * P_cas_mel_cond
  recompose:            P_mel = (1 - P_vasc) * P_mel_cond
                        P_nv  = (1 - P_vasc) * (1 - P_mel_cond)

Reuses LR + cascade OOF probabilities cached on disk; no model retraining.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
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
EPS = 1e-9


def lr_oof_proba(seed):
    labels = load_labels(DATA_DIR).copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    labels["base_id"] = labels["image_id"].apply(base_id_from_image_id)
    y = labels["dx"].to_numpy()
    groups = labels["base_id"].to_numpy()

    print(f"[LR] Building feature table all_abcd_grouped...")
    fdf = build_feature_table(DATA_DIR, image_ids=labels["image_id"],
                              feature_set="all_abcd_grouped", mask_mode="raw")
    fdf["image_id"] = fdf["image_id"].astype(str)
    X = labels.merge(fdf, on="image_id").drop(columns=["image_id", "dx", "base_id"])

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups=groups))

    proba = np.zeros((len(y), len(LABELS)), dtype=np.float64)
    for tr, va in splits:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=min(140, X.shape[1]))),
            ("clf", LogisticRegression(C=0.3, class_weight="balanced",
                                       max_iter=4000, random_state=seed)),
        ])
        pipe.fit(X.iloc[tr], y[tr])
        cls_order = list(pipe.named_steps["clf"].classes_)
        idx_map = [cls_order.index(c) for c in LABELS]
        proba[va] = pipe.predict_proba(X.iloc[va])[:, idx_map]
    return labels, y, groups, proba, splits


def to_axes(p_mel, p_nv, p_vasc):
    """Return (p_vasc, p_mel_cond) where p_mel_cond = P(mel | not vasc)."""
    p_vasc = p_vasc.copy()
    denom = p_mel + p_nv + EPS
    p_mel_cond = p_mel / denom
    return p_vasc, p_mel_cond


def from_axes(p_vasc, p_mel_cond):
    p_vasc = np.clip(p_vasc, 0.0, 1.0)
    p_mel_cond = np.clip(p_mel_cond, 0.0, 1.0)
    p_mel = (1.0 - p_vasc) * p_mel_cond
    p_nv = (1.0 - p_vasc) * (1.0 - p_mel_cond)
    out = np.stack([p_mel, p_nv, p_vasc], axis=1)  # ordering must match LABELS below
    return out


def structural_fuse(lr_p, cas_p, w1, w2, label_idx):
    lr_v, lr_mc = to_axes(lr_p[:, label_idx["mel"]], lr_p[:, label_idx["nv"]], lr_p[:, label_idx["vasc"]])
    cas_v, cas_mc = to_axes(cas_p[:, label_idx["mel"]], cas_p[:, label_idx["nv"]], cas_p[:, label_idx["vasc"]])
    fused_v = w1 * lr_v + (1.0 - w1) * cas_v
    fused_mc = w2 * lr_mc + (1.0 - w2) * cas_mc
    return from_axes(fused_v, fused_mc)


def metrics(y, proba_mel_nv_vasc):
    pred = np.array(LABELS)[proba_mel_nv_vasc.argmax(axis=1)]
    return {
        "acc": round(accuracy_score(y, pred), 4),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "bal_acc": round(balanced_accuracy_score(y, pred), 4),
        "n_err": int((pred != y).sum()),
    }


def grid_search_2d(y, lr_p, cas_p, label_idx, step=0.05):
    grid = np.arange(0.0, 1.0001, step)
    best = None
    for w1 in grid:
        for w2 in grid:
            fused = structural_fuse(lr_p, cas_p, w1, w2, label_idx)
            ba = balanced_accuracy_score(y, np.array(LABELS)[fused.argmax(axis=1)])
            if (best is None) or (ba > best[2]):
                best = (w1, w2, ba)
    return best


def main():
    cas_csv = ROOT / "outputs/metrics/cascade_seed127_oof_predictions.csv"
    if not cas_csv.exists():
        raise FileNotFoundError(f"missing {cas_csv}")

    labels, y, groups, lr_p, splits = lr_oof_proba(SEED)
    cas_df = pd.read_csv(cas_csv)
    cas_df["image_id"] = cas_df["image_id"].astype(str)
    merged = labels[["image_id"]].merge(cas_df, on="image_id", how="left")
    label_idx = {l: i for i, l in enumerate(LABELS)}
    cas_p = np.zeros((len(y), 3))
    cas_p[:, label_idx["mel"]] = merged["cascade_p_mel"].values
    cas_p[:, label_idx["nv"]] = merged["cascade_p_nv"].values
    cas_p[:, label_idx["vasc"]] = merged["cascade_p_vasc"].values

    print("\n=== Baselines (seed 127 OOF, n=600) ===")
    print(f"  LR only          : {metrics(y, lr_p)}")
    print(f"  Cascade only     : {metrics(y, cas_p)}")
    fused_5050 = structural_fuse(lr_p, cas_p, 0.5, 0.5, label_idx)
    print(f"  Structural 0.5/0.5: {metrics(y, fused_5050)}")
    flat_5050 = 0.5 * lr_p + 0.5 * cas_p
    flat_5050 = flat_5050 / flat_5050.sum(axis=1, keepdims=True)
    print(f"  Flat 0.5/0.5     : {metrics(y, flat_5050)}")

    print("\n=== Grid: in-sample upper bound (overfits seed 127) ===")
    w1_in, w2_in, ba_in = grid_search_2d(y, lr_p, cas_p, label_idx, step=0.05)
    fused_in = structural_fuse(lr_p, cas_p, w1_in, w2_in, label_idx)
    print(f"  Best (w1, w2)    : ({w1_in:.2f}, {w2_in:.2f})  BalAcc grid={ba_in:.4f}")
    print(f"  Metrics          : {metrics(y, fused_in)}")

    print("\n=== Nested CV: weights from inner folds, eval on outer fold ===")
    nested_pred = np.empty(len(y), dtype=object)
    nested_proba = np.zeros((len(y), 3))
    chosen = []
    for f, (tr, va) in enumerate(splits):
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + f)
        # We use the OUTER train indices' OOF probas to pick weights; OOF probas
        # are already out-of-fold from the outer CV, so picking weights on them
        # is honest given the outer fold is held out.
        w1_f, w2_f, _ = grid_search_2d(y[tr], lr_p[tr], cas_p[tr], label_idx, step=0.05)
        chosen.append((w1_f, w2_f))
        fused_va = structural_fuse(lr_p[va], cas_p[va], w1_f, w2_f, label_idx)
        nested_proba[va] = fused_va
        nested_pred[va] = np.array(LABELS)[fused_va.argmax(axis=1)]
        print(f"  Fold {f}: w1={w1_f:.2f}  w2={w2_f:.2f}")

    print(f"\n  Nested fusion seed 127:")
    print(f"    Acc      = {accuracy_score(y, nested_pred):.4f}")
    print(f"    Macro-F1 = {f1_score(y, nested_pred, average='macro'):.4f}")
    print(f"    Bal-Acc  = {balanced_accuracy_score(y, nested_pred):.4f}")
    print(f"    Errors   = {(nested_pred != y).sum()}")

    print("\n=== Per-fold weight stability ===")
    w1s = [c[0] for c in chosen]
    w2s = [c[1] for c in chosen]
    print(f"  w1 (vasc gate)        : mean={np.mean(w1s):.3f}  std={np.std(w1s):.3f}  values={w1s}")
    print(f"  w2 (mel|nonvasc edge) : mean={np.mean(w2s):.3f}  std={np.std(w2s):.3f}  values={w2s}")

    print("\n=== Confusion delta vs flat 0.5/0.5 ===")
    flat_pred = np.array(LABELS)[flat_5050.argmax(axis=1)]
    structural_pred = np.array(LABELS)[fused_5050.argmax(axis=1)]
    nested_only = (nested_pred != y) & (flat_pred == y)
    flat_only = (flat_pred != y) & (nested_pred == y)
    both_wrong = (nested_pred != y) & (flat_pred != y)
    print(f"  Both wrong (hard floor)            : {both_wrong.sum()}")
    print(f"  Flat right, nested wrong (regress) : {nested_only.sum()}")
    print(f"  Nested right, flat wrong (gain)    : {flat_only.sum()}")


if __name__ == "__main__":
    main()

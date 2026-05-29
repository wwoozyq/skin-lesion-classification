"""Cascade-variant × {baseline, A1 shades_of_gray} sweep.

Resolves the discrepancy between ledger §9's headline 0.7887 (an unspecified
cascade config) and ``run_overnight_exploration.py``'s cell 0 (0.7623 with
``d2-more-trees, k=all, soft``). Sweeps all 3 Stage 2 variants × 3 k_features
× 1 cascade_mode under no-preprocessing AND under A1 to pick the best
absolute config.
"""

import argparse
import json
import sys
import time
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
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE1_K = "100"
STAGE1_XGB = {
    "n_estimators": 150, "max_depth": 2, "learning_rate": 0.05,
    "subsample": 0.80, "colsample_bytree": 0.60, "reg_lambda": 5.0,
}

STAGE2_FEATURE_SET = "xgb_cascade_stage2"
STAGE2_VARIANTS = {
    "strong-reg": {
        "n_estimators": 150, "max_depth": 2, "learning_rate": 0.05,
        "subsample": 0.70, "colsample_bytree": 0.50, "reg_lambda": 10.0,
    },
    "d2-more-trees": {
        "n_estimators": 300, "max_depth": 2, "learning_rate": 0.03,
        "subsample": 0.80, "colsample_bytree": 0.60, "reg_lambda": 5.0,
    },
    "deeper": {
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
        "subsample": 0.80, "colsample_bytree": 0.60, "reg_lambda": 5.0,
    },
}
K_FEATURES_GRID = ["100", "120", "all"]
SEEDS = [42, 127, 2024, 3407, 520]


def _cache_path(cache_dir, feature_set, mask_mode, preprocessing):
    return Path(cache_dir) / f"features_cascade_{feature_set}_{mask_mode}_pp_{preprocessing}.csv"


def _load_features(data_dir, labels, feature_set, preprocessing, cache_dir):
    cache_path = _cache_path(cache_dir, feature_set, "raw", preprocessing)
    if cache_path.exists():
        table = pd.read_csv(cache_path)
    else:
        table = build_feature_table(
            data_dir, image_ids=labels["image_id"].astype(str),
            feature_set=feature_set, mask_mode="raw", preprocessing=preprocessing,
        )
        ensure_dir(cache_path.parent)
        table.to_csv(cache_path, index=False)
    table["image_id"] = table["image_id"].astype(str)
    merged = labels.merge(table, on="image_id", how="inner")
    return merged.drop(columns=["image_id", "dx"])


def _build_pipeline(xgb_kwargs, k_features, n_features, random_state):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    clf = XGBClassifier(
        **xgb_kwargs, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=random_state, n_jobs=-1, verbosity=0,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


def _make_splits(labels, seed):
    groups = labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=labels.index)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(placeholder, labels["dx"], groups=groups))


def _run_cascade(X_stage1, X_stage2, y_true, labels_df, variant, k_features, seeds, label_idx, label_array):
    mel_nv_indices = np.where(y_true != "vasc")[0]
    y_stage1 = (y_true == "vasc").astype(int)
    xgb_stage2 = STAGE2_VARIANTS[variant]

    per_seed_probs = []
    per_seed_metrics = []
    for seed in seeds:
        splits = _make_splits(labels_df, seed)
        stage1_pipe = _build_pipeline(STAGE1_XGB, STAGE1_K, X_stage1.shape[1], seed)
        stage1_oof = np.zeros((len(y_true), 2), dtype=np.float32)
        for tr, va in splits:
            sw = compute_sample_weight("balanced", y_stage1[tr])
            stage1_pipe.fit(X_stage1.iloc[tr], y_stage1[tr], clf__sample_weight=sw)
            stage1_oof[va] = stage1_pipe.predict_proba(X_stage1.iloc[va])
        p_vasc = stage1_oof[:, 1]

        stage2_oof = np.zeros((len(y_true), 2), dtype=np.float32)
        for tr, va in splits:
            tr_mn = tr[np.isin(tr, mel_nv_indices)]
            y_s2 = (y_true[tr_mn] == "nv").astype(int)
            pipe = _build_pipeline(xgb_stage2, k_features, X_stage2.shape[1], seed)
            sw = compute_sample_weight("balanced", y_s2)
            pipe.fit(X_stage2.iloc[tr_mn], y_s2, clf__sample_weight=sw)
            stage2_oof[va] = pipe.predict_proba(X_stage2.iloc[va])
        p_mel = stage2_oof[:, 0]
        p_nv = stage2_oof[:, 1]

        probs = np.zeros((len(y_true), len(LABELS)), dtype=np.float32)
        probs[:, label_idx["vasc"]] = p_vasc
        probs[:, label_idx["mel"]] = (1.0 - p_vasc) * p_mel
        probs[:, label_idx["nv"]] = (1.0 - p_vasc) * p_nv
        per_seed_probs.append(probs)
        pred = label_array[probs.argmax(axis=1)]
        per_seed_metrics.append(compute_metrics(y_true, pred, labels=LABELS))

    bagged_probs = np.mean(per_seed_probs, axis=0)
    bagged_pred = label_array[bagged_probs.argmax(axis=1)]
    return compute_metrics(y_true, bagged_pred, labels=LABELS), per_seed_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--output_csv", default="outputs/metrics/variant_sweep_a1.csv")
    parser.add_argument("--preprocessings", default="none,shades_of_gray")
    args = parser.parse_args()

    labels = load_labels(Path(args.data_dir)).reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y_true = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {label: idx for idx, label in enumerate(LABELS)}

    rows = []
    started = time.time()
    for preprocessing in [p.strip() for p in args.preprocessings.split(",")]:
        print(f"\n=== preprocessing={preprocessing} ===")
        X1 = _load_features(Path(args.data_dir), labels, STAGE1_FEATURE_SET, preprocessing, Path(args.cache_dir))
        X2 = _load_features(Path(args.data_dir), labels, STAGE2_FEATURE_SET, preprocessing, Path(args.cache_dir))
        for variant in STAGE2_VARIANTS:
            for k in K_FEATURES_GRID:
                t0 = time.time()
                bagged, per_seed = _run_cascade(X1, X2, y_true, labels, variant, k, SEEDS, label_idx, label_array)
                seed_bal = [m["balanced_accuracy"] for m in per_seed]
                row = {
                    "preprocessing": preprocessing,
                    "stage2_variant": variant,
                    "k_features": k,
                    "bagged_balanced_accuracy": bagged["balanced_accuracy"],
                    "bagged_macro_f1": bagged["macro_f1"],
                    "bagged_accuracy": bagged["accuracy"],
                    "per_seed_bal_acc_mean": float(np.mean(seed_bal)),
                    "per_seed_bal_acc_std": float(np.std(seed_bal)),
                    "wall_min": (time.time() - t0) / 60.0,
                }
                rows.append(row)
                print(f"  {variant:>13s} k={k:>3s}: bagged bal_acc={bagged['balanced_accuracy']:.4f} "
                      f"f1={bagged['macro_f1']:.4f} acc={bagged['accuracy']:.4f} ({row['wall_min']:.1f} min)")
                ensure_dir(Path(args.output_csv).parent)
                pd.DataFrame(rows).to_csv(args.output_csv, index=False)

    df = pd.DataFrame(rows).sort_values("bagged_balanced_accuracy", ascending=False)
    print("\nTop 10:")
    print(df.head(10).to_string(index=False))
    print(f"\nTotal wall: {(time.time() - started) / 60.0:.1f} min")
    print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()

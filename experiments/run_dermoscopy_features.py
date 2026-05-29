"""Dermoscopy structural features experiment runner.

Implements the locked plan in docs/DERMOSCOPY_FEATURES_PLAN.md
(written alongside, not committed): two experiments, four modes.

Experiments:
  A) main_lr     — LR(C=0.3, k=140) + all_abcd_grouped_dermoscopy
  B) cascade_s2  — XGB cascade, Stage 1 unchanged, Stage 2 = xgb_cascade_stage2_dermoscopy
                   variant locked to deeper, k=120, soft

Modes:
  seed127_sanity   single seed (127), both experiments, ~2 min
  full_5seed       5-seed bagged, both experiments, gate=seed127 not regressing
  loo_ablation     5-seed bagged with one dermoscopy feature group dropped at a time
  a1_ablation      5-seed bagged, both experiments, with shades_of_gray preprocessing
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
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


SEEDS = [42, 127, 2024, 3407, 520]
GATE_SEED = 127

MAIN_LR_K = 140
MAIN_LR_C = 0.3

STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE1_K = "100"
STAGE1_XGB = {
    "n_estimators": 150, "max_depth": 2, "learning_rate": 0.05,
    "subsample": 0.80, "colsample_bytree": 0.60, "reg_lambda": 5.0,
}

# Locked Stage 2 cascade variant (best from variant sweep §15)
STAGE2_FEATURE_SET = "xgb_cascade_stage2_dermoscopy"
STAGE2_K = "120"
STAGE2_XGB = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.80, "colsample_bytree": 0.60, "reg_lambda": 5.0,
}

# Group prefixes used for leave-one-out ablation
DERMOSCOPY_GROUPS = {
    "color_diversity": ("dermoscopy_lab_L_range_p5_95", "dermoscopy_lab_a_range_p5_95",
                        "dermoscopy_lab_b_range_p5_95", "dermoscopy_lab_octant_entropy",
                        "dermoscopy_lab_pca1_std", "dermoscopy_lab_max_octant_fraction"),
    "blue_white":      ("dermoscopy_bluewhite_area_ratio",
                        "dermoscopy_bluewhite_max_component_ratio",
                        "dermoscopy_bluewhite_n_components"),
    "regression":      ("dermoscopy_regression_white_area_ratio",
                        "dermoscopy_regression_white_max_component_ratio",
                        "dermoscopy_regression_white_n_components",
                        "dermoscopy_regression_bluegray_area_ratio"),
    "asymmetry":       ("dermoscopy_eccentricity",
                        "dermoscopy_asym_pca1_L", "dermoscopy_asym_pca1_a", "dermoscopy_asym_pca1_b",
                        "dermoscopy_asym_pca2_L", "dermoscopy_asym_pca2_a", "dermoscopy_asym_pca2_b"),
    "vascular":        ("dermoscopy_vascular_area_ratio",
                        "dermoscopy_vascular_max_a_within_lesion"),
}


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


def _make_splits(labels, seed):
    groups = labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=labels.index)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(placeholder, labels["dx"], groups=groups))


def _build_lr_pipeline(n_features, random_state):
    steps = [("scaler", StandardScaler())]
    k = min(MAIN_LR_K, n_features)
    if k < n_features:
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))
    steps.append(("clf", LogisticRegression(
        C=MAIN_LR_C, max_iter=2000, class_weight="balanced", random_state=random_state,
    )))
    return Pipeline(steps)


def _build_xgb_pipeline(xgb_kwargs, k_features, n_features, random_state):
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


def _run_main_lr(X, y_true, labels_df, seeds, label_array):
    per_seed_probs, per_seed_metrics = [], []
    for seed in seeds:
        splits = _make_splits(labels_df, seed)
        oof = np.zeros((len(y_true), len(LABELS)), dtype=np.float32)
        for tr, va in splits:
            pipe = _build_lr_pipeline(X.shape[1], seed)
            pipe.fit(X.iloc[tr], y_true[tr])
            classes = list(pipe.named_steps["clf"].classes_)
            proba = pipe.predict_proba(X.iloc[va])
            for col_idx, label in enumerate(classes):
                oof[va, list(LABELS).index(label)] = proba[:, col_idx]
        pred = label_array[oof.argmax(axis=1)]
        per_seed_probs.append(oof)
        per_seed_metrics.append(compute_metrics(y_true, pred, labels=LABELS))
    bagged = np.mean(per_seed_probs, axis=0)
    bagged_pred = label_array[bagged.argmax(axis=1)]
    return compute_metrics(y_true, bagged_pred, labels=LABELS), per_seed_metrics


def _run_cascade(X1, X2, y_true, labels_df, seeds, label_idx, label_array):
    mel_nv = np.where(y_true != "vasc")[0]
    y_stage1 = (y_true == "vasc").astype(int)
    per_seed_probs, per_seed_metrics = [], []
    for seed in seeds:
        splits = _make_splits(labels_df, seed)

        stage1_oof = np.zeros((len(y_true), 2), dtype=np.float32)
        s1_pipe = _build_xgb_pipeline(STAGE1_XGB, STAGE1_K, X1.shape[1], seed)
        for tr, va in splits:
            sw = compute_sample_weight("balanced", y_stage1[tr])
            s1_pipe.fit(X1.iloc[tr], y_stage1[tr], clf__sample_weight=sw)
            stage1_oof[va] = s1_pipe.predict_proba(X1.iloc[va])
        p_vasc = stage1_oof[:, 1]

        stage2_oof = np.zeros((len(y_true), 2), dtype=np.float32)
        for tr, va in splits:
            tr_mn = tr[np.isin(tr, mel_nv)]
            y_s2 = (y_true[tr_mn] == "nv").astype(int)
            s2_pipe = _build_xgb_pipeline(STAGE2_XGB, STAGE2_K, X2.shape[1], seed)
            sw = compute_sample_weight("balanced", y_s2)
            s2_pipe.fit(X2.iloc[tr_mn], y_s2, clf__sample_weight=sw)
            stage2_oof[va] = s2_pipe.predict_proba(X2.iloc[va])
        p_mel = stage2_oof[:, 0]
        p_nv = stage2_oof[:, 1]

        probs = np.zeros((len(y_true), len(LABELS)), dtype=np.float32)
        probs[:, label_idx["vasc"]] = p_vasc
        probs[:, label_idx["mel"]] = (1.0 - p_vasc) * p_mel
        probs[:, label_idx["nv"]] = (1.0 - p_vasc) * p_nv
        per_seed_probs.append(probs)
        pred = label_array[probs.argmax(axis=1)]
        per_seed_metrics.append(compute_metrics(y_true, pred, labels=LABELS))

    bagged = np.mean(per_seed_probs, axis=0)
    bagged_pred = label_array[bagged.argmax(axis=1)]
    return compute_metrics(y_true, bagged_pred, labels=LABELS), per_seed_metrics


def _summarize(name, bagged, per_seed, wall_min, extra=None):
    seed_bal = [m["balanced_accuracy"] for m in per_seed]
    seed_f1 = [m["macro_f1"] for m in per_seed]
    seed_acc = [m["accuracy"] for m in per_seed]
    row = {
        "experiment": name,
        "bagged_balanced_accuracy": bagged["balanced_accuracy"],
        "bagged_macro_f1": bagged["macro_f1"],
        "bagged_accuracy": bagged["accuracy"],
        "per_seed_bal_acc_mean": float(np.mean(seed_bal)),
        "per_seed_bal_acc_std": float(np.std(seed_bal)),
        "per_seed_macro_f1_mean": float(np.mean(seed_f1)),
        "per_seed_accuracy_mean": float(np.mean(seed_acc)),
        "n_seeds": len(per_seed),
        "wall_min": wall_min,
    }
    if extra:
        row.update(extra)
    return row


def _drop_columns(X, drop_keys):
    cols_to_drop = [c for c in X.columns if c in drop_keys]
    if not cols_to_drop:
        return X
    return X.drop(columns=cols_to_drop)


def run_seed127_sanity(data_dir, cache_dir, preprocessing):
    labels = load_labels(Path(data_dir)).reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {l: i for i, l in enumerate(LABELS)}

    rows = []
    # A) main LR
    t0 = time.time()
    X_main = _load_features(Path(data_dir), labels, "all_abcd_grouped_dermoscopy", preprocessing, Path(cache_dir))
    bagged, per_seed = _run_main_lr(X_main, y, labels, [GATE_SEED], label_array)
    rows.append(_summarize("main_lr_dermoscopy", bagged, per_seed, (time.time() - t0) / 60.0,
                           extra={"preprocessing": preprocessing, "n_features": X_main.shape[1]}))

    # B) cascade Stage 2 = dermoscopy alias
    t0 = time.time()
    X1 = _load_features(Path(data_dir), labels, STAGE1_FEATURE_SET, preprocessing, Path(cache_dir))
    X2 = _load_features(Path(data_dir), labels, STAGE2_FEATURE_SET, preprocessing, Path(cache_dir))
    bagged, per_seed = _run_cascade(X1, X2, y, labels, [GATE_SEED], label_idx, label_array)
    rows.append(_summarize("cascade_s2_dermoscopy", bagged, per_seed, (time.time() - t0) / 60.0,
                           extra={"preprocessing": preprocessing, "n_features_s1": X1.shape[1], "n_features_s2": X2.shape[1]}))
    return rows


def run_full_5seed(data_dir, cache_dir, preprocessing):
    labels = load_labels(Path(data_dir)).reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {l: i for i, l in enumerate(LABELS)}

    rows = []
    t0 = time.time()
    X_main = _load_features(Path(data_dir), labels, "all_abcd_grouped_dermoscopy", preprocessing, Path(cache_dir))
    bagged, per_seed = _run_main_lr(X_main, y, labels, SEEDS, label_array)
    rows.append(_summarize("main_lr_dermoscopy", bagged, per_seed, (time.time() - t0) / 60.0,
                           extra={"preprocessing": preprocessing, "n_features": X_main.shape[1]}))

    t0 = time.time()
    X1 = _load_features(Path(data_dir), labels, STAGE1_FEATURE_SET, preprocessing, Path(cache_dir))
    X2 = _load_features(Path(data_dir), labels, STAGE2_FEATURE_SET, preprocessing, Path(cache_dir))
    bagged, per_seed = _run_cascade(X1, X2, y, labels, SEEDS, label_idx, label_array)
    rows.append(_summarize("cascade_s2_dermoscopy", bagged, per_seed, (time.time() - t0) / 60.0,
                           extra={"preprocessing": preprocessing, "n_features_s1": X1.shape[1], "n_features_s2": X2.shape[1]}))
    return rows


def run_loo_ablation(data_dir, cache_dir, preprocessing, experiment):
    """Leave-one-dermoscopy-group-out ablation. Runs whichever experiment is requested."""
    labels = load_labels(Path(data_dir)).reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {l: i for i, l in enumerate(LABELS)}

    rows = []
    if experiment == "main":
        X_full = _load_features(Path(data_dir), labels, "all_abcd_grouped_dermoscopy", preprocessing, Path(cache_dir))
        for drop_name, drop_keys in DERMOSCOPY_GROUPS.items():
            t0 = time.time()
            X = _drop_columns(X_full, set(drop_keys))
            bagged, per_seed = _run_main_lr(X, y, labels, SEEDS, label_array)
            rows.append(_summarize(f"main_lr_drop_{drop_name}", bagged, per_seed, (time.time() - t0) / 60.0,
                                   extra={"preprocessing": preprocessing, "n_features": X.shape[1],
                                          "dropped_group": drop_name, "dropped_n": len(drop_keys)}))
    elif experiment == "cascade":
        X1 = _load_features(Path(data_dir), labels, STAGE1_FEATURE_SET, preprocessing, Path(cache_dir))
        X2_full = _load_features(Path(data_dir), labels, STAGE2_FEATURE_SET, preprocessing, Path(cache_dir))
        for drop_name, drop_keys in DERMOSCOPY_GROUPS.items():
            t0 = time.time()
            X2 = _drop_columns(X2_full, set(drop_keys))
            bagged, per_seed = _run_cascade(X1, X2, y, labels, SEEDS, label_idx, label_array)
            rows.append(_summarize(f"cascade_s2_drop_{drop_name}", bagged, per_seed, (time.time() - t0) / 60.0,
                                   extra={"preprocessing": preprocessing, "n_features_s2": X2.shape[1],
                                          "dropped_group": drop_name, "dropped_n": len(drop_keys)}))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--mode", required=True,
                        choices=["seed127_sanity", "full_5seed", "loo_main", "loo_cascade", "a1_ablation"])
    parser.add_argument("--output_csv", default="outputs/metrics/dermoscopy_results.csv")
    args = parser.parse_args()

    started = time.time()
    if args.mode == "seed127_sanity":
        rows = run_seed127_sanity(args.data_dir, args.cache_dir, preprocessing="none")
    elif args.mode == "full_5seed":
        rows = run_full_5seed(args.data_dir, args.cache_dir, preprocessing="none")
    elif args.mode == "a1_ablation":
        rows = run_full_5seed(args.data_dir, args.cache_dir, preprocessing="shades_of_gray")
        for r in rows:
            r["experiment"] = r["experiment"] + "_a1"
    elif args.mode == "loo_main":
        rows = run_loo_ablation(args.data_dir, args.cache_dir, preprocessing="none", experiment="main")
    elif args.mode == "loo_cascade":
        rows = run_loo_ablation(args.data_dir, args.cache_dir, preprocessing="none", experiment="cascade")

    df = pd.DataFrame(rows)
    ensure_dir(Path(args.output_csv).parent)
    if Path(args.output_csv).exists():
        existing = pd.read_csv(args.output_csv)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(args.output_csv, index=False)
    total = (time.time() - started) / 60.0

    print()
    for r in rows:
        print(f"  [{r['experiment']:40s}] "
              f"bagged_bal_acc={r['bagged_balanced_accuracy']:.4f} "
              f"bagged_f1={r['bagged_macro_f1']:.4f} "
              f"bagged_acc={r['bagged_accuracy']:.4f} "
              f"(per-seed std={r['per_seed_bal_acc_std']:.4f}, "
              f"n_seeds={r['n_seeds']}, {r['wall_min']:.1f}min)")
    print(f"\nMode={args.mode} wall={total:.1f}min  Saved/appended to {args.output_csv}")


if __name__ == "__main__":
    main()

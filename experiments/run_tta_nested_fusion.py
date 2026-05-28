"""TTA + nested-CV fusion weight experiment.

Investigates whether the 0.8055 single-seed fusion result can be made stable
across 5 seeds by:

    (1) test-time augmentation (TTA) on the inference side, averaging
        probabilities across 6 geometric variants of each validation image;
    (2) per-outer-fold fusion weight selection on inner-CV OOF probabilities,
        replacing the fixed `weight = 0.5` used by `run_fusion_ensemble.py`.

Runs 4 ablation cells in one pass (sharing per-fold trained models):

    A: identity inference, w = 0.5     (reproduces existing fusion baseline)
    B: TTA inference,      w = 0.5     (isolates TTA contribution)
    C: identity inference, w = w*      (isolates nested-CV contribution)
    D: TTA inference,      w = w*      (headline)

Outputs:
    outputs/metrics/tta_nested_fusion_per_seed.csv
    outputs/metrics/tta_nested_fusion_summary.csv
    outputs/metrics/tta_nested_fusion_oof_seed127.csv
    outputs/cache/features_tta_{set}_{mask_mode}_{transform}.csv  (per transform)

Note: xgboost on this Mac requires the OpenMP shim from sklearn's bundled libs:

    DYLD_LIBRARY_PATH=.venv/lib/python3.9/site-packages/sklearn/.dylibs \\
        .venv/bin/python experiments/run_tta_nested_fusion.py ...
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "xgboost is required. Install via "
        "`uv pip install --python .venv/bin/python xgboost`."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABEL_TO_ID, LABELS
from src.dataset import load_image, load_labels, load_mask
from src.evaluate import compute_metrics
from src.features import build_feature_table, extract_features_for_image
from src.preprocess import prepare_mask
from src.utils import base_id_from_image_id, ensure_dir

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")

# ---- Frozen model configs (must match run_fusion_ensemble.py to keep cell A
#      as a faithful sanity reproduction of the existing 0.7763 bagged result).
BASELINE_FEATURE_SET = "all_abcd_grouped"
BASELINE_K = 140
BASELINE_LR_C = 0.3
BASELINE_LR_RANDOM_STATE = 42  # hardcoded in _baseline_oof_proba

STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE1_K = 100
STAGE1_XGB = {
    "n_estimators": 150,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_lambda": 5.0,
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

TTA_TRANSFORMS = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")
CELLS = ("A", "B", "C", "D")
CELL_USE_TTA = {"A": False, "B": True, "C": False, "D": True}
CELL_TUNE_WEIGHT = {"A": False, "B": False, "C": True, "D": True}


# ---------------------------------------------------------------------------
# Geometric transforms (image + mask kept in sync).
# ---------------------------------------------------------------------------

def apply_transform(image, mask, transform):
    if transform == "identity":
        return image, mask
    if transform == "hflip":
        return np.ascontiguousarray(np.flip(image, axis=1)), np.ascontiguousarray(np.flip(mask, axis=1))
    if transform == "vflip":
        return np.ascontiguousarray(np.flip(image, axis=0)), np.ascontiguousarray(np.flip(mask, axis=0))
    if transform == "rot90":
        return np.ascontiguousarray(np.rot90(image, k=1, axes=(0, 1))), np.ascontiguousarray(np.rot90(mask, k=1, axes=(0, 1)))
    if transform == "rot180":
        return np.ascontiguousarray(np.rot90(image, k=2, axes=(0, 1))), np.ascontiguousarray(np.rot90(mask, k=2, axes=(0, 1)))
    if transform == "rot270":
        return np.ascontiguousarray(np.rot90(image, k=3, axes=(0, 1))), np.ascontiguousarray(np.rot90(mask, k=3, axes=(0, 1)))
    raise ValueError(f"Unknown transform {transform!r}")


# ---------------------------------------------------------------------------
# Feature loading: identity hits existing cache; TTA writes its own.
# ---------------------------------------------------------------------------

def _existing_cache_path(cache_dir, feature_set, mask_mode):
    candidates = [
        Path(cache_dir) / f"features_fusion_{feature_set}_{mask_mode}.csv",
        Path(cache_dir) / f"features_cascade_{feature_set}_{mask_mode}.csv",
        Path(cache_dir) / f"features_{feature_set}_{mask_mode}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _tta_cache_path(cache_dir, feature_set, mask_mode, transform):
    return Path(cache_dir) / f"features_tta_{feature_set}_{mask_mode}_{transform}.csv"


def _build_tta_feature_table(data_dir, labels, feature_set, mask_mode, transform):
    """Compute (image, mask) -> transform -> prepare_mask -> features for all rows."""
    rows = []
    image_ids = labels["image_id"].astype(str).tolist()
    for image_id in tqdm(image_ids, desc=f"  TTA {feature_set}/{transform}", leave=False):
        image = load_image(data_dir, image_id)
        mask = load_mask(data_dir, image_id)
        img_t, mask_t = apply_transform(image, mask, transform)
        mask_prepared = prepare_mask(mask_t, mask_mode=mask_mode)
        feats = extract_features_for_image(img_t, mask_prepared, feature_set=feature_set)
        row = {"image_id": image_id, **feats}
        rows.append(row)
    return pd.DataFrame(rows)


def _load_feature_table_for_transform(data_dir, labels, feature_set, mask_mode,
                                      transform, cache_dir, use_cache):
    """Return DataFrame with `image_id` and feature columns for one transform."""
    if transform == "identity":
        if use_cache:
            existing = _existing_cache_path(cache_dir, feature_set, mask_mode)
            if existing is not None:
                return pd.read_csv(existing)
        table = build_feature_table(
            data_dir,
            image_ids=labels["image_id"].astype(str),
            feature_set=feature_set,
            mask_mode=mask_mode,
        )
        if use_cache:
            ensure_dir(Path(cache_dir))
            out = Path(cache_dir) / f"features_{feature_set}_{mask_mode}.csv"
            table.to_csv(out, index=False)
        return table

    cache_path = _tta_cache_path(cache_dir, feature_set, mask_mode, transform)
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    table = _build_tta_feature_table(data_dir, labels, feature_set, mask_mode, transform)
    if use_cache:
        ensure_dir(cache_path.parent)
        table.to_csv(cache_path, index=False)
    return table


def _materialize_X(labels, table):
    table = table.copy()
    table["image_id"] = table["image_id"].astype(str)
    merged = labels.merge(table, on="image_id", how="inner")
    if len(merged) != len(labels):
        raise ValueError(
            f"Feature table missing samples: got {len(merged)}, expected {len(labels)}."
        )
    return merged.drop(columns=["image_id", "dx"]).reset_index(drop=True)


def prepare_all_features(data_dir, labels, mask_mode, cache_dir, use_cache,
                         transforms):
    """Returns dict[transform][feature_set] -> DataFrame aligned with `labels`."""
    feature_sets = (BASELINE_FEATURE_SET, STAGE1_FEATURE_SET, STAGE2_FEATURE_SET)
    out = {}
    for transform in transforms:
        out[transform] = {}
        for feature_set in feature_sets:
            table = _load_feature_table_for_transform(
                data_dir=data_dir,
                labels=labels,
                feature_set=feature_set,
                mask_mode=mask_mode,
                transform=transform,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            out[transform][feature_set] = _materialize_X(labels, table)
    return out


# ---------------------------------------------------------------------------
# Model construction + fitting.
# ---------------------------------------------------------------------------

def _build_baseline_pipeline(n_features):
    k_value = min(BASELINE_K, n_features)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_value)),
        ("clf", LogisticRegression(
            C=BASELINE_LR_C,
            max_iter=2000,
            class_weight="balanced",
            random_state=BASELINE_LR_RANDOM_STATE,
        )),
    ])


def _build_xgb(xgb_kwargs, n_classes, random_state, n_jobs):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    return XGBClassifier(
        **xgb_kwargs,
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def _build_xgb_pipeline(xgb_kwargs, n_classes, k_features, n_features, random_state, n_jobs):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", _build_xgb(xgb_kwargs, n_classes, random_state, n_jobs)))
    return Pipeline(steps)


def fit_baseline_on_train(X_train, y_train):
    pipe = _build_baseline_pipeline(X_train.shape[1])
    pipe.fit(X_train, y_train)
    return pipe


def fit_cascade_on_train(X_s1_train, X_s2_train, y_train, variant, k_features,
                        random_state, n_jobs):
    """Fit stage1 (vasc vs rest) and stage2 (mel vs nv) on a train fold."""
    y_stage1 = (y_train == "vasc").astype(int)
    stage1 = _build_xgb_pipeline(
        STAGE1_XGB, n_classes=2, k_features=STAGE1_K,
        n_features=X_s1_train.shape[1], random_state=random_state, n_jobs=n_jobs,
    )
    stage1.fit(X_s1_train, y_stage1, clf__sample_weight=compute_sample_weight("balanced", y_stage1))

    mel_nv_mask = y_train != "vasc"
    X_s2_mn = X_s2_train[mel_nv_mask].reset_index(drop=True)
    y_s2 = (y_train[mel_nv_mask] == "nv").astype(int)
    stage2 = _build_xgb_pipeline(
        STAGE2_VARIANTS[variant], n_classes=2, k_features=k_features,
        n_features=X_s2_mn.shape[1], random_state=random_state, n_jobs=n_jobs,
    )
    stage2.fit(X_s2_mn, y_s2, clf__sample_weight=compute_sample_weight("balanced", y_s2))
    return stage1, stage2


# ---------------------------------------------------------------------------
# Prediction (with optional TTA averaging).
# ---------------------------------------------------------------------------

def _reorder_proba(proba, classes, target_labels=LABELS):
    classes = np.asarray(classes)
    ordered = np.zeros((len(proba), len(target_labels)), dtype=np.float32)
    for out_idx, label in enumerate(target_labels):
        col = int(np.where(classes == label)[0][0])
        ordered[:, out_idx] = proba[:, col]
    return ordered


def _baseline_predict_proba(pipe, X_val):
    return _reorder_proba(pipe.predict_proba(X_val), pipe.classes_)


def _cascade_predict_proba(stage1, stage2, X_s1_val, X_s2_val):
    p_vasc = stage1.predict_proba(X_s1_val)[:, 1]
    p_s2 = stage2.predict_proba(X_s2_val)
    out = np.zeros((len(X_s1_val), len(LABELS)), dtype=np.float32)
    out[:, LABEL_TO_ID["vasc"]] = p_vasc
    out[:, LABEL_TO_ID["mel"]] = (1.0 - p_vasc) * p_s2[:, 0]
    out[:, LABEL_TO_ID["nv"]] = (1.0 - p_vasc) * p_s2[:, 1]
    return out


def baseline_predict_with_tta(pipe, features_by_transform, val_idx, transforms):
    probs = [
        _baseline_predict_proba(pipe, features_by_transform[t][BASELINE_FEATURE_SET].iloc[val_idx])
        for t in transforms
    ]
    return np.mean(probs, axis=0)


def cascade_predict_with_tta(stage1, stage2, features_by_transform, val_idx, transforms):
    probs = [
        _cascade_predict_proba(
            stage1, stage2,
            features_by_transform[t][STAGE1_FEATURE_SET].iloc[val_idx],
            features_by_transform[t][STAGE2_FEATURE_SET].iloc[val_idx],
        )
        for t in transforms
    ]
    return np.mean(probs, axis=0)


# ---------------------------------------------------------------------------
# Inner CV for fusion weight selection.
# ---------------------------------------------------------------------------

def select_fusion_weight(X_main, X_s1, X_s2, y, base_ids_train, train_idx,
                         inner_k, weight_grid, weight_metric, variant, k_features,
                         random_state, n_jobs):
    """Run K_inner CV inside `train_idx`, sweep weight on inner-OOF, return w*."""
    inner_X_main = X_main.iloc[train_idx].reset_index(drop=True)
    inner_X_s1 = X_s1.iloc[train_idx].reset_index(drop=True)
    inner_X_s2 = X_s2.iloc[train_idx].reset_index(drop=True)
    inner_y = y[train_idx]
    inner_groups = base_ids_train

    splitter = StratifiedGroupKFold(n_splits=inner_k, shuffle=True,
                                    random_state=random_state + 1)
    inner_base = np.zeros((len(inner_y), len(LABELS)), dtype=np.float32)
    inner_cas = np.zeros((len(inner_y), len(LABELS)), dtype=np.float32)

    for tr_i, va_i in splitter.split(inner_X_main, inner_y, groups=inner_groups):
        bp = fit_baseline_on_train(inner_X_main.iloc[tr_i], inner_y[tr_i])
        inner_base[va_i] = _baseline_predict_proba(bp, inner_X_main.iloc[va_i])
        s1, s2 = fit_cascade_on_train(
            inner_X_s1.iloc[tr_i], inner_X_s2.iloc[tr_i], inner_y[tr_i],
            variant=variant, k_features=k_features,
            random_state=random_state, n_jobs=n_jobs,
        )
        inner_cas[va_i] = _cascade_predict_proba(
            s1, s2, inner_X_s1.iloc[va_i], inner_X_s2.iloc[va_i],
        )

    labels_arr = np.asarray(LABELS)
    best_w, best_score = 0.5, -np.inf
    for w in weight_grid:
        fused = (1.0 - w) * inner_base + w * inner_cas
        pred = labels_arr[fused.argmax(axis=1)]
        if weight_metric == "macro_f1":
            score = f1_score(inner_y, pred, average="macro")
        elif weight_metric == "balanced_accuracy":
            score = balanced_accuracy_score(inner_y, pred)
        else:
            raise ValueError(f"Unknown weight_metric {weight_metric!r}")
        if score > best_score:
            best_score, best_w = score, w
    return float(best_w), float(best_score)


# ---------------------------------------------------------------------------
# Metrics helpers.
# ---------------------------------------------------------------------------

def metrics_from_proba(y_true, probs):
    pred = np.asarray(LABELS)[probs.argmax(axis=1)]
    return compute_metrics(y_true, pred, labels=LABELS)


# ---------------------------------------------------------------------------
# Main experiment driver.
# ---------------------------------------------------------------------------

def run_experiment(
    data_dir,
    seeds,
    n_splits,
    inner_k,
    mask_mode,
    cascade_variant,
    cascade_k_features,
    weight_grid,
    weight_metric,
    transforms,
    run_cells,
    cache_dir,
    use_cache,
    output_per_seed_csv,
    output_summary_csv,
    output_oof_csv,
    oof_seed,
    n_jobs,
):
    t_start = time.time()
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required.")
    labels = labels.copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()
    base_ids = labels["image_id"].map(base_id_from_image_id)
    n = len(labels)

    print(f"Samples: {n}")
    print(f"Cells:   {','.join(run_cells)}")
    print(f"Seeds:   {seeds}")
    print(f"Cascade: variant={cascade_variant} k={cascade_k_features}")
    print(f"Weight:  grid={weight_grid[0]}..{weight_grid[-1]} ({len(weight_grid)} pts), metric={weight_metric}")
    print(f"TTA:     transforms={list(transforms)}")
    print()
    print(f"Preparing features (TTA × {len(transforms)} transforms × 3 feature sets)...")
    features_by_transform = prepare_all_features(
        data_dir, labels, mask_mode, cache_dir, use_cache, transforms,
    )
    for t in transforms:
        sizes = {fs: features_by_transform[t][fs].shape[1] for fs in features_by_transform[t]}
        print(f"  {t:>9s}: {sizes}")

    X_main = features_by_transform["identity"][BASELINE_FEATURE_SET]
    X_s1 = features_by_transform["identity"][STAGE1_FEATURE_SET]
    X_s2 = features_by_transform["identity"][STAGE2_FEATURE_SET]

    # cell -> seed -> (n, 3) fused proba
    cell_seed_probs = {c: {} for c in run_cells}
    per_seed_rows = []
    weight_log_rows = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        outer_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        outer_splits = list(outer_cv.split(X_main, y, groups=base_ids))

        # Per-seed accumulator: cell -> array (n, 3)
        fused_acc = {c: np.zeros((n, len(LABELS)), dtype=np.float32) for c in run_cells}

        for fold_i, (train_idx, val_idx) in enumerate(outer_splits):
            t_fold = time.time()
            base_ids_train = base_ids.iloc[train_idx].reset_index(drop=True)

            # Train outer baseline + cascade on identity features.
            bp = fit_baseline_on_train(X_main.iloc[train_idx], y[train_idx])
            s1, s2 = fit_cascade_on_train(
                X_s1.iloc[train_idx], X_s2.iloc[train_idx], y[train_idx],
                variant=cascade_variant, k_features=cascade_k_features,
                random_state=seed, n_jobs=n_jobs,
            )

            # Inner CV for w* (only when any cell needs it).
            needs_tune = any(CELL_TUNE_WEIGHT[c] for c in run_cells)
            if needs_tune:
                w_star, w_score = select_fusion_weight(
                    X_main=X_main, X_s1=X_s1, X_s2=X_s2, y=y,
                    base_ids_train=base_ids_train, train_idx=train_idx,
                    inner_k=inner_k, weight_grid=weight_grid,
                    weight_metric=weight_metric,
                    variant=cascade_variant, k_features=cascade_k_features,
                    random_state=seed, n_jobs=n_jobs,
                )
            else:
                w_star, w_score = 0.5, float("nan")
            weight_log_rows.append({
                "seed": seed, "fold": fold_i,
                "w_star": w_star, "w_inner_score": w_score,
            })

            # Identity predictions (shared by cells A and C).
            base_val_id = _baseline_predict_proba(bp, X_main.iloc[val_idx])
            cas_val_id = _cascade_predict_proba(
                s1, s2, X_s1.iloc[val_idx], X_s2.iloc[val_idx],
            )

            # TTA predictions (shared by cells B and D).
            need_tta = any(CELL_USE_TTA[c] for c in run_cells)
            if need_tta:
                base_val_tta = baseline_predict_with_tta(bp, features_by_transform, val_idx, transforms)
                cas_val_tta = cascade_predict_with_tta(s1, s2, features_by_transform, val_idx, transforms)
            else:
                base_val_tta = base_val_id
                cas_val_tta = cas_val_id

            for cell in run_cells:
                use_tta = CELL_USE_TTA[cell]
                tune_w = CELL_TUNE_WEIGHT[cell]
                base_val = base_val_tta if use_tta else base_val_id
                cas_val = cas_val_tta if use_tta else cas_val_id
                w = w_star if tune_w else 0.5
                fused = (1.0 - w) * base_val + w * cas_val
                fused_acc[cell][val_idx] = fused
                fm = metrics_from_proba(y[val_idx], fused)
                per_seed_rows.append({
                    "cell": cell, "seed": seed, "fold": fold_i,
                    "selected_weight": w,
                    "balanced_accuracy": fm["balanced_accuracy"],
                    "macro_f1": fm["macro_f1"],
                    "accuracy": fm["accuracy"],
                })

            print(
                f"  fold {fold_i} w*={w_star:.2f} time={time.time() - t_fold:.1f}s"
            )

        for cell in run_cells:
            cell_seed_probs[cell][seed] = fused_acc[cell]
            sm = metrics_from_proba(y, fused_acc[cell])
            print(
                f"  cell {cell} seed {seed:>4d}: "
                f"bal_acc={sm['balanced_accuracy']:.4f} "
                f"macro_f1={sm['macro_f1']:.4f} "
                f"acc={sm['accuracy']:.4f}"
            )

    # Summary across seeds: per-seed mean/std + bagged (mean of proba across seeds).
    summary_rows = []
    print("\n=== Summary ===")
    for cell in run_cells:
        seed_probs = list(cell_seed_probs[cell].values())
        per_seed = [metrics_from_proba(y, p) for p in seed_probs]
        bagged_proba = np.mean(seed_probs, axis=0)
        bagged = metrics_from_proba(y, bagged_proba)

        per_seed_weights = [
            r["selected_weight"] for r in per_seed_rows if r["cell"] == cell
        ]
        summary_rows.append({
            "cell": cell,
            "use_tta": CELL_USE_TTA[cell],
            "tune_weight": CELL_TUNE_WEIGHT[cell],
            "bagged_balanced_accuracy": bagged["balanced_accuracy"],
            "bagged_macro_f1": bagged["macro_f1"],
            "bagged_accuracy": bagged["accuracy"],
            "per_seed_bal_acc_mean": float(np.mean([m["balanced_accuracy"] for m in per_seed])),
            "per_seed_bal_acc_std": float(np.std([m["balanced_accuracy"] for m in per_seed])),
            "per_seed_macro_f1_mean": float(np.mean([m["macro_f1"] for m in per_seed])),
            "per_seed_macro_f1_std": float(np.std([m["macro_f1"] for m in per_seed])),
            "per_seed_accuracy_mean": float(np.mean([m["accuracy"] for m in per_seed])),
            "per_seed_accuracy_std": float(np.std([m["accuracy"] for m in per_seed])),
            "selected_weight_mean": float(np.mean(per_seed_weights)),
            "selected_weight_std": float(np.std(per_seed_weights)),
        })
        print(
            f"  cell {cell}: bagged bal_acc={bagged['balanced_accuracy']:.4f} "
            f"macro_f1={bagged['macro_f1']:.4f} acc={bagged['accuracy']:.4f}   "
            f"w*={summary_rows[-1]['selected_weight_mean']:.2f}"
            f"±{summary_rows[-1]['selected_weight_std']:.2f}"
        )

    ensure_dir(Path(output_per_seed_csv).parent)
    ensure_dir(Path(output_summary_csv).parent)
    pd.DataFrame(per_seed_rows).to_csv(output_per_seed_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(output_summary_csv, index=False)
    print(f"\nWrote {output_per_seed_csv}")
    print(f"Wrote {output_summary_csv}")

    # Cell-D, configured seed OOF for downstream error analysis.
    if "D" in run_cells and oof_seed in cell_seed_probs["D"]:
        d_probs = cell_seed_probs["D"][oof_seed]
        d_pred = np.asarray(LABELS)[d_probs.argmax(axis=1)]
        oof_df = pd.DataFrame({
            "image_id": labels["image_id"],
            "base_id": base_ids,
            "dx": y,
            "pred": d_pred,
            "p_mel": d_probs[:, LABEL_TO_ID["mel"]],
            "p_nv": d_probs[:, LABEL_TO_ID["nv"]],
            "p_vasc": d_probs[:, LABEL_TO_ID["vasc"]],
            "correct": d_pred == y,
        })
        ensure_dir(Path(output_oof_csv).parent)
        oof_df.to_csv(output_oof_csv, index=False)
        print(f"Wrote {output_oof_csv} (cell D, seed {oof_seed})")

    print(f"\nWall time: {(time.time() - t_start) / 60:.1f} min")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weight_grid(value):
    return [float(x) for x in _parse_csv(value)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--inner_k", type=int, default=3)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--cascade_variant", default="d2-more-trees",
                        choices=list(STAGE2_VARIANTS))
    parser.add_argument("--cascade_k_features", default="all")
    parser.add_argument("--weight_grid", default=",".join(
        f"{w:.2f}" for w in np.linspace(0.0, 1.0, 21)
    ))
    parser.add_argument("--weight_metric", default="macro_f1",
                        choices=["macro_f1", "balanced_accuracy"])
    parser.add_argument("--tta_transforms", default=",".join(TTA_TRANSFORMS))
    parser.add_argument("--run_cells", default=",".join(CELLS))
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--output_per_seed_csv",
                        default="outputs/metrics/tta_nested_fusion_per_seed.csv")
    parser.add_argument("--output_summary_csv",
                        default="outputs/metrics/tta_nested_fusion_summary.csv")
    parser.add_argument("--output_oof_csv",
                        default="outputs/metrics/tta_nested_fusion_oof_seed127.csv")
    parser.add_argument("--oof_seed", type=int, default=127)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    transforms = tuple(_parse_csv(args.tta_transforms))
    unknown_t = set(transforms) - set(TTA_TRANSFORMS)
    if unknown_t:
        raise ValueError(f"Unknown TTA transforms: {sorted(unknown_t)}")

    cells = tuple(_parse_csv(args.run_cells))
    unknown_c = set(cells) - set(CELLS)
    if unknown_c:
        raise ValueError(f"Unknown cells: {sorted(unknown_c)}")

    run_experiment(
        data_dir=Path(args.data_dir),
        seeds=[int(s) for s in _parse_csv(args.seeds)],
        n_splits=args.n_splits,
        inner_k=args.inner_k,
        mask_mode=args.mask_mode,
        cascade_variant=args.cascade_variant,
        cascade_k_features=args.cascade_k_features,
        weight_grid=_parse_weight_grid(args.weight_grid),
        weight_metric=args.weight_metric,
        transforms=transforms,
        run_cells=cells,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        output_per_seed_csv=Path(args.output_per_seed_csv),
        output_summary_csv=Path(args.output_summary_csv),
        output_oof_csv=Path(args.output_oof_csv),
        oof_seed=args.oof_seed,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "xgboost is required for fusion ensemble experiments. "
        "Install it with: uv pip install --python .venv/bin/python xgboost"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")

BASELINE_FEATURE_SET = "all_abcd_grouped"
BASELINE_K = "140"
BASELINE_MODEL = "lr03"

STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE1_K = "100"
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
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.70,
        "colsample_bytree": 0.50,
        "reg_lambda": 10.0,
    },
    "d2-more-trees": {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
    "deeper": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
}


def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_feature_name(feature_set):
    return feature_set.replace("+", "_AND_")


def _feature_cache_candidates(cache_dir, feature_set, mask_mode):
    safe = _safe_feature_name(feature_set)
    return [
        Path(cache_dir) / f"features_fusion_{safe}_{mask_mode}.csv",
        Path(cache_dir) / f"features_cascade_{safe}_{mask_mode}.csv",
        Path(cache_dir) / f"features_{safe}_{mask_mode}.csv",
    ]


def _load_feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    candidates = _feature_cache_candidates(cache_dir, feature_set, mask_mode)
    if use_cache:
        for path in candidates:
            if path.exists():
                return pd.read_csv(path)

    table = build_feature_table(
        data_dir,
        image_ids=labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    if use_cache:
        ensure_dir(candidates[0].parent)
        table.to_csv(candidates[0], index=False)
    return table


def _prepare_X(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    table = _load_feature_table(
        data_dir=data_dir,
        labels=labels,
        feature_set=feature_set,
        mask_mode=mask_mode,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    table["image_id"] = table["image_id"].astype(str)
    merged = labels.merge(table, on="image_id", how="inner")
    if len(merged) != len(labels):
        raise ValueError(f"Feature table for {feature_set} is missing samples.")
    return merged.drop(columns=["image_id", "dx"])


def _make_splits(labels, n_splits, seed):
    groups = labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(placeholder, labels["dx"], groups=groups))


def _reorder_proba(proba, classes, target_labels=LABELS):
    classes = np.asarray(classes)
    ordered = np.zeros((len(proba), len(target_labels)), dtype=np.float32)
    for out_idx, label in enumerate(target_labels):
        if label not in classes:
            raise ValueError(f"Class {label!r} missing from model classes {classes}.")
        ordered[:, out_idx] = proba[:, int(np.where(classes == label)[0][0])]
    return ordered


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


def _baseline_oof_proba(X, y, splits):
    k_value = min(int(BASELINE_K), X.shape[1])
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_value)),
        ("clf", LogisticRegression(
            C=0.3,
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    oof = np.zeros((len(X), len(LABELS)), dtype=np.float32)
    for train_idx, valid_idx in splits:
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = _reorder_proba(
            model.predict_proba(X.iloc[valid_idx]),
            model.classes_,
        )
    return oof


def _cascade_oof_proba(X_stage1, X_stage2, y, splits, variant, k_features, random_state, n_jobs):
    label_idx = {label: idx for idx, label in enumerate(LABELS)}
    y_stage1 = (y == "vasc").astype(int)

    stage1_pipe = _build_xgb_pipeline(
        STAGE1_XGB,
        n_classes=2,
        k_features=STAGE1_K,
        n_features=X_stage1.shape[1],
        random_state=random_state,
        n_jobs=n_jobs,
    )
    p_vasc = np.zeros(len(y), dtype=np.float32)
    stage2_binary = np.zeros((len(y), 2), dtype=np.float32)
    mel_nv_indices = np.where(y != "vasc")[0]

    for train_idx, valid_idx in splits:
        stage1_weight = compute_sample_weight("balanced", y_stage1[train_idx])
        stage1_pipe.fit(
            X_stage1.iloc[train_idx],
            y_stage1[train_idx],
            clf__sample_weight=stage1_weight,
        )
        p_vasc[valid_idx] = stage1_pipe.predict_proba(X_stage1.iloc[valid_idx])[:, 1]

        train_mel_nv = train_idx[np.isin(train_idx, mel_nv_indices)]
        y_stage2 = (y[train_mel_nv] == "nv").astype(int)
        stage2_pipe = _build_xgb_pipeline(
            STAGE2_VARIANTS[variant],
            n_classes=2,
            k_features=k_features,
            n_features=X_stage2.shape[1],
            random_state=random_state,
            n_jobs=n_jobs,
        )
        stage2_weight = compute_sample_weight("balanced", y_stage2)
        stage2_pipe.fit(
            X_stage2.iloc[train_mel_nv],
            y_stage2,
            clf__sample_weight=stage2_weight,
        )
        stage2_binary[valid_idx] = stage2_pipe.predict_proba(X_stage2.iloc[valid_idx])

    cascade = np.zeros((len(y), len(LABELS)), dtype=np.float32)
    cascade[:, label_idx["vasc"]] = p_vasc
    cascade[:, label_idx["mel"]] = (1.0 - p_vasc) * stage2_binary[:, 0]
    cascade[:, label_idx["nv"]] = (1.0 - p_vasc) * stage2_binary[:, 1]
    return cascade


def _metrics_from_proba(y_true, probs):
    pred = np.asarray(LABELS)[probs.argmax(axis=1)]
    return compute_metrics(y_true, pred, labels=LABELS)


def run_fusion(
    data_dir,
    seeds,
    n_splits,
    mask_mode,
    cascade_variants,
    cascade_k_features,
    weights,
    output_csv,
    output_per_seed_csv,
    cache_dir,
    use_cache,
    n_jobs,
):
    started = time.time()
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for fusion ensemble.")
    labels = labels.copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()

    print(f"Samples: {len(labels)}")
    print(f"Original model A: {BASELINE_FEATURE_SET} + {BASELINE_MODEL}, k={BASELINE_K}")
    print(f"Cascade model B: {STAGE1_FEATURE_SET} -> {STAGE2_FEATURE_SET}")
    print(f"Seeds: {seeds}")
    print(f"Cascade variants: {cascade_variants}")
    print(f"Cascade k: {cascade_k_features}")
    print(f"Cascade weights: {weights}")

    X_baseline = _prepare_X(
        data_dir, labels, BASELINE_FEATURE_SET, mask_mode, cache_dir, use_cache
    )
    X_stage1 = _prepare_X(
        data_dir, labels, STAGE1_FEATURE_SET, mask_mode, cache_dir, use_cache
    )
    X_stage2 = _prepare_X(
        data_dir, labels, STAGE2_FEATURE_SET, mask_mode, cache_dir, use_cache
    )
    print(f"Baseline features: {X_baseline.shape[1]}")
    print(f"Cascade Stage 1 features: {X_stage1.shape[1]}")
    print(f"Cascade Stage 2 features: {X_stage2.shape[1]}")

    config_seed_probs = {}
    per_seed_rows = []

    for seed in seeds:
        print(f"\nSeed {seed}")
        splits = _make_splits(labels, n_splits=n_splits, seed=seed)
        baseline_probs = _baseline_oof_proba(X_baseline, y, splits)
        baseline_metrics = _metrics_from_proba(y, baseline_probs)
        print(
            "  original baseline "
            f"bal_acc={baseline_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={baseline_metrics['macro_f1']:.4f} "
            f"acc={baseline_metrics['accuracy']:.4f}"
        )

        for variant in cascade_variants:
            for k in cascade_k_features:
                cascade_probs = _cascade_oof_proba(
                    X_stage1=X_stage1,
                    X_stage2=X_stage2,
                    y=y,
                    splits=splits,
                    variant=variant,
                    k_features=k,
                    random_state=seed,
                    n_jobs=n_jobs,
                )
                cascade_metrics = _metrics_from_proba(y, cascade_probs)
                print(
                    f"  cascade {variant:>10s} k={k:<3s} "
                    f"bal_acc={cascade_metrics['balanced_accuracy']:.4f} "
                    f"macro_f1={cascade_metrics['macro_f1']:.4f} "
                    f"acc={cascade_metrics['accuracy']:.4f}"
                )

                for weight in weights:
                    fused_probs = (1.0 - weight) * baseline_probs + weight * cascade_probs
                    metrics = _metrics_from_proba(y, fused_probs)
                    key = (variant, k, weight)
                    config_seed_probs.setdefault(key, []).append(fused_probs)
                    per_seed_rows.append({
                        "cascade_variant": variant,
                        "cascade_k_features": k,
                        "cascade_weight": weight,
                        "seed": seed,
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "accuracy": metrics["accuracy"],
                    })

    rows = []
    for (variant, k, weight), probs_list in config_seed_probs.items():
        per_seed_bal = []
        per_seed_macro = []
        per_seed_acc = []
        for probs in probs_list:
            metrics = _metrics_from_proba(y, probs)
            per_seed_bal.append(metrics["balanced_accuracy"])
            per_seed_macro.append(metrics["macro_f1"])
            per_seed_acc.append(metrics["accuracy"])

        bagged_probs = np.mean(probs_list, axis=0)
        bagged_metrics = _metrics_from_proba(y, bagged_probs)
        rows.append({
            "original_model": f"{BASELINE_FEATURE_SET}_{BASELINE_MODEL}_k{BASELINE_K}",
            "cascade_variant": variant,
            "cascade_k_features": k,
            "cascade_weight": weight,
            "original_weight": 1.0 - weight,
            "bagged_balanced_accuracy": bagged_metrics["balanced_accuracy"],
            "bagged_macro_f1": bagged_metrics["macro_f1"],
            "bagged_accuracy": bagged_metrics["accuracy"],
            "per_seed_bal_acc_mean": float(np.mean(per_seed_bal)),
            "per_seed_bal_acc_std": float(np.std(per_seed_bal)),
            "per_seed_macro_f1_mean": float(np.mean(per_seed_macro)),
            "per_seed_accuracy_mean": float(np.mean(per_seed_acc)),
        })

    result = pd.DataFrame(rows).sort_values("bagged_balanced_accuracy", ascending=False)
    per_seed_result = pd.DataFrame(per_seed_rows).sort_values(
        ["seed", "balanced_accuracy"],
        ascending=[True, False],
    )
    ensure_dir(Path(output_csv).parent)
    ensure_dir(Path(output_per_seed_csv).parent)
    result.to_csv(output_csv, index=False)
    per_seed_result.to_csv(output_per_seed_csv, index=False)

    print("\nTop fusion configs:")
    print(result.head(15).to_string(index=False))
    print(f"\nSaved fusion summary to {output_csv}")
    print(f"Saved per-seed fusion rows to {output_per_seed_csv}")
    print(f"Wall time: {(time.time() - started) / 60:.1f} min")
    return result, per_seed_result


def main():
    parser = argparse.ArgumentParser(
        description="Fuse the original LR/ABCD-grouped baseline with the XGBoost cascade."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seeds", default="127")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--cascade_variants", default="strong-reg")
    parser.add_argument("--cascade_k_features", default="120")
    parser.add_argument("--weights", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--output_csv", default="outputs/metrics/fusion_ensemble.csv")
    parser.add_argument(
        "--output_per_seed_csv",
        default="outputs/metrics/fusion_ensemble_per_seed.csv",
    )
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    cascade_variants = _parse_csv(args.cascade_variants)
    unknown = sorted(set(cascade_variants) - set(STAGE2_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown cascade variants: {unknown}")

    run_fusion(
        data_dir=Path(args.data_dir),
        seeds=[int(seed) for seed in _parse_csv(args.seeds)],
        n_splits=args.n_splits,
        mask_mode=args.mask_mode,
        cascade_variants=cascade_variants,
        cascade_k_features=_parse_csv(args.cascade_k_features),
        weights=[float(weight) for weight in _parse_csv(args.weights)],
        output_csv=Path(args.output_csv),
        output_per_seed_csv=Path(args.output_per_seed_csv),
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

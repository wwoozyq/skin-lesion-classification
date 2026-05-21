import argparse
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

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "xgboost is required for this cascade search. "
        "Install it with: uv pip install --python .venv/bin/python xgboost"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE2_FEATURE_SET = "xgb_cascade_stage2"
STAGE1_K = "100"
STAGE1_XGB = {
    "n_estimators": 150,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_lambda": 5.0,
}

STAGE2_XGB_VARIANTS = {
    "v7-default": {
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
    "d3-more-trees": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
    "d2-more-trees": {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
    "less-reg": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_lambda": 2.0,
    },
    "strong-reg": {
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.70,
        "colsample_bytree": 0.50,
        "reg_lambda": 10.0,
    },
    "slow-learner": {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.02,
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

CASCADE_MODES = {
    "soft": ("soft", 0.5),
    "hard@0.4": ("hard", 0.4),
}


def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _feature_cache_path(cache_dir, feature_set, mask_mode):
    safe_name = feature_set.replace("+", "_AND_")
    return Path(cache_dir) / f"features_cascade_{safe_name}_{mask_mode}.csv"


def _load_feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    cache_path = _feature_cache_path(cache_dir, feature_set, mask_mode)
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    table = build_feature_table(
        data_dir,
        image_ids=labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    if use_cache:
        ensure_dir(cache_path.parent)
        table.to_csv(cache_path, index=False)
    return table


def _build_pipeline(xgb_kwargs, n_classes, k_features, n_features, random_state, n_jobs):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))

    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    clf = XGBClassifier(
        **xgb_kwargs,
        objective=objective,
        eval_metric=eval_metric,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


def _make_splits(labels, n_splits, seed):
    groups = labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(placeholder, labels["dx"], groups=groups))


def _fit_oof_binary(pipe, X, y_binary, splits):
    oof = np.zeros((len(X), 2), dtype=np.float32)
    for train_idx, valid_idx in splits:
        sample_weight = compute_sample_weight("balanced", y_binary[train_idx])
        pipe.fit(X.iloc[train_idx], y_binary[train_idx], clf__sample_weight=sample_weight)
        oof[valid_idx] = pipe.predict_proba(X.iloc[valid_idx])
    return oof


def run_search(
    data_dir,
    seeds,
    n_splits,
    mask_mode,
    stage2_variants,
    k_features,
    cascade_modes,
    output_csv,
    output_per_seed_csv,
    cache_dir,
    use_cache,
    n_jobs,
):
    started = time.time()
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for cascade search.")

    labels = labels.copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y_true = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {label: idx for idx, label in enumerate(LABELS)}

    print(f"Samples: {len(labels)}")
    print(f"Seeds: {seeds}")
    print(f"Stage 1: {STAGE1_FEATURE_SET}, k={STAGE1_K}, XGB={STAGE1_XGB}")
    print(f"Stage 2: {STAGE2_FEATURE_SET}")
    print(f"Stage 2 variants: {stage2_variants}")
    print(f"k_features: {k_features}")
    print(f"modes: {cascade_modes}")

    feature_tables = {}
    for feature_set in [STAGE1_FEATURE_SET, STAGE2_FEATURE_SET]:
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
        feature_tables[feature_set] = merged.drop(columns=["image_id", "dx"])
        print(f"{feature_set}: {feature_tables[feature_set].shape[1]} features")

    X_stage1 = feature_tables[STAGE1_FEATURE_SET]
    X_stage2 = feature_tables[STAGE2_FEATURE_SET]
    y_stage1 = (y_true == "vasc").astype(int)
    mel_nv_indices = np.where(y_true != "vasc")[0]

    per_seed_probs = {}
    stage1_rows = []

    for seed in seeds:
        print(f"\nSeed {seed}")
        splits = _make_splits(labels, n_splits=n_splits, seed=seed)

        stage1_pipe = _build_pipeline(
            STAGE1_XGB,
            n_classes=2,
            k_features=STAGE1_K,
            n_features=X_stage1.shape[1],
            random_state=seed,
            n_jobs=n_jobs,
        )
        stage1_oof = _fit_oof_binary(stage1_pipe, X_stage1, y_stage1, splits)
        p_vasc = stage1_oof[:, 1]
        stage1_pred = np.where(p_vasc > 0.5, "vasc", "non_vasc")
        stage1_true = np.where(y_true == "vasc", "vasc", "non_vasc")
        stage1_acc = float((stage1_pred == stage1_true).mean())
        stage1_rows.append({"seed": seed, "stage1_vasc_accuracy": stage1_acc})
        print(f"  Stage 1 vasc OOF accuracy: {stage1_acc:.4f}")

        for variant_name in stage2_variants:
            xgb_kwargs = STAGE2_XGB_VARIANTS[variant_name]
            for k in k_features:
                stage2_oof = np.zeros((len(labels), 2), dtype=np.float32)
                for train_idx, valid_idx in splits:
                    train_mel_nv = train_idx[np.isin(train_idx, mel_nv_indices)]
                    y_stage2 = (y_true[train_mel_nv] == "nv").astype(int)
                    pipe = _build_pipeline(
                        xgb_kwargs,
                        n_classes=2,
                        k_features=k,
                        n_features=X_stage2.shape[1],
                        random_state=seed,
                        n_jobs=n_jobs,
                    )
                    sample_weight = compute_sample_weight("balanced", y_stage2)
                    pipe.fit(
                        X_stage2.iloc[train_mel_nv],
                        y_stage2,
                        clf__sample_weight=sample_weight,
                    )
                    stage2_oof[valid_idx] = pipe.predict_proba(X_stage2.iloc[valid_idx])

                p_mel = stage2_oof[:, 0]
                p_nv = stage2_oof[:, 1]

                for mode_name in cascade_modes:
                    mode, threshold = CASCADE_MODES[mode_name]
                    probs = np.zeros((len(labels), len(LABELS)), dtype=np.float32)
                    if mode == "soft":
                        probs[:, label_idx["vasc"]] = p_vasc
                        probs[:, label_idx["mel"]] = (1.0 - p_vasc) * p_mel
                        probs[:, label_idx["nv"]] = (1.0 - p_vasc) * p_nv
                    else:
                        is_vasc = p_vasc > threshold
                        probs[:, label_idx["vasc"]] = np.where(is_vasc, 1.0, 0.0)
                        probs[:, label_idx["mel"]] = np.where(is_vasc, 0.0, p_mel)
                        probs[:, label_idx["nv"]] = np.where(is_vasc, 0.0, p_nv)

                    key = (variant_name, k, mode_name)
                    per_seed_probs.setdefault(key, []).append((seed, probs))

                soft_probs = per_seed_probs[(variant_name, k, "soft")][-1][1]
                soft_pred = label_array[soft_probs.argmax(axis=1)]
                soft_metrics = compute_metrics(y_true, soft_pred, labels=LABELS)
                print(
                    f"  {variant_name:>13s} k={k:<3s} "
                    f"soft_bal_acc={soft_metrics['balanced_accuracy']:.4f}"
                )

    rows = []
    per_seed_rows = []
    for (variant_name, k, mode_name), seed_probs in per_seed_probs.items():
        probs_only = []
        per_seed_balanced = []
        for seed, probs in seed_probs:
            probs_only.append(probs)
            pred = label_array[probs.argmax(axis=1)]
            metrics = compute_metrics(y_true, pred, labels=LABELS)
            per_seed_balanced.append(metrics["balanced_accuracy"])
            per_seed_rows.append({
                "stage2_xgb_variant": variant_name,
                "stage2_k_features": k,
                "cascade_mode": mode_name,
                "seed": seed,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            })

        bagged_probs = np.mean(probs_only, axis=0)
        bagged_pred = label_array[bagged_probs.argmax(axis=1)]
        bagged_metrics = compute_metrics(y_true, bagged_pred, labels=LABELS)
        variant_kwargs = STAGE2_XGB_VARIANTS[variant_name]
        rows.append({
            "stage2_xgb_variant": variant_name,
            "stage2_k_features": k,
            "cascade_mode": mode_name,
            "stage2_n_estimators": variant_kwargs["n_estimators"],
            "stage2_max_depth": variant_kwargs["max_depth"],
            "stage2_learning_rate": variant_kwargs["learning_rate"],
            "stage2_reg_lambda": variant_kwargs["reg_lambda"],
            "bagged_balanced_accuracy": bagged_metrics["balanced_accuracy"],
            "bagged_macro_f1": bagged_metrics["macro_f1"],
            "bagged_accuracy": bagged_metrics["accuracy"],
            "per_seed_bal_acc_mean": float(np.mean(per_seed_balanced)),
            "per_seed_bal_acc_std": float(np.std(per_seed_balanced)),
            "per_seed_bal_acc_min": float(np.min(per_seed_balanced)),
            "per_seed_bal_acc_max": float(np.max(per_seed_balanced)),
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

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_path = Path(output_csv).with_name(Path(output_csv).stem + "_stage1.csv")
    stage1_df.to_csv(stage1_path, index=False)

    print("\nTop cascade configs:")
    print(result.head(12).to_string(index=False))
    print(f"\nSaved bagged results to {output_csv}")
    print(f"Saved per-seed results to {output_per_seed_csv}")
    print(f"Saved Stage 1 diagnostics to {stage1_path}")
    print(f"Wall time: {(time.time() - started) / 60:.1f} min")

    return result, per_seed_result


def main():
    parser = argparse.ArgumentParser(
        description="Strict grouped-OOF XGBoost cascade search for traditional ML."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument(
        "--stage2_variants",
        default="strong-reg,d2-more-trees,deeper",
        help=f"Comma-separated subset of: {','.join(STAGE2_XGB_VARIANTS)}",
    )
    parser.add_argument("--k_features", default="100,120,all")
    parser.add_argument("--cascade_modes", default="soft,hard@0.4")
    parser.add_argument("--output_csv", default="outputs/metrics/xgb_cascade_search.csv")
    parser.add_argument(
        "--output_per_seed_csv",
        default="outputs/metrics/xgb_cascade_search_per_seed.csv",
    )
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    stage2_variants = _parse_csv(args.stage2_variants)
    unknown_variants = sorted(set(stage2_variants) - set(STAGE2_XGB_VARIANTS))
    if unknown_variants:
        raise ValueError(f"Unknown Stage 2 variants: {unknown_variants}")

    cascade_modes = _parse_csv(args.cascade_modes)
    unknown_modes = sorted(set(cascade_modes) - set(CASCADE_MODES))
    if unknown_modes:
        raise ValueError(f"Unknown cascade modes: {unknown_modes}")

    run_search(
        data_dir=Path(args.data_dir),
        seeds=[int(seed) for seed in _parse_csv(args.seeds)],
        n_splits=args.n_splits,
        mask_mode=args.mask_mode,
        stage2_variants=stage2_variants,
        k_features=_parse_csv(args.k_features),
        cascade_modes=cascade_modes,
        output_csv=Path(args.output_csv),
        output_per_seed_csv=Path(args.output_per_seed_csv),
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

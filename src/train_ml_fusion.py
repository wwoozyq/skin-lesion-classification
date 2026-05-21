import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - optional dependency.
    raise ImportError(
        "xgboost is required for fusion training. "
        "Install it with: uv pip install --python .venv/bin/python xgboost"
    ) from exc

from .config import DEFAULT_MODEL_DIR, LABELS
from .dataset import load_labels
from .features import build_feature_table
from .utils import ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")

ORIGINAL_FEATURE_SET = "all_abcd_grouped"
ORIGINAL_K = "140"

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
    "d2-more-trees": {
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.80,
        "colsample_bytree": 0.60,
        "reg_lambda": 5.0,
    },
    "strong-reg": {
        "n_estimators": 150,
        "max_depth": 2,
        "learning_rate": 0.05,
        "subsample": 0.70,
        "colsample_bytree": 0.50,
        "reg_lambda": 10.0,
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


def _cache_path(cache_dir, feature_set, mask_mode):
    safe_name = feature_set.replace("+", "_AND_")
    return Path(cache_dir) / f"features_fusion_{safe_name}_{mask_mode}.csv"


def _cache_candidates(cache_dir, feature_set, mask_mode):
    safe_name = feature_set.replace("+", "_AND_")
    return [
        Path(cache_dir) / f"features_fusion_{safe_name}_{mask_mode}.csv",
        Path(cache_dir) / f"features_cascade_{safe_name}_{mask_mode}.csv",
        Path(cache_dir) / f"features_{safe_name}_{mask_mode}.csv",
    ]


def _build_training_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    feature_df = None
    if use_cache:
        for path in _cache_candidates(cache_dir, feature_set, mask_mode):
            if path.exists():
                feature_df = pd.read_csv(path)
                break
    if feature_df is None:
        feature_df = build_feature_table(
            data_dir,
            image_ids=labels["image_id"].astype(str),
            feature_set=feature_set,
            mask_mode=mask_mode,
        )
        if use_cache:
            cache_path = _cache_path(cache_dir, feature_set, mask_mode)
            ensure_dir(cache_path.parent)
            feature_df.to_csv(cache_path, index=False)

    feature_df["image_id"] = feature_df["image_id"].astype(str)
    merged = labels.merge(feature_df, on="image_id", how="inner")
    if len(merged) != len(labels):
        raise ValueError(f"Feature table for {feature_set} is missing samples.")
    return merged.drop(columns=["image_id", "dx"])


def _build_xgb_pipeline(xgb_kwargs, n_classes, k_features, n_features, random_state, n_jobs):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))

    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    steps.append((
        "clf",
        XGBClassifier(
            **xgb_kwargs,
            objective=objective,
            eval_metric=eval_metric,
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        ),
    ))
    return Pipeline(steps)


def train_fusion(
    data_dir,
    mask_mode="raw",
    cascade_variant="d2-more-trees",
    cascade_k_features="all",
    cascade_weight=0.5,
    model_path=None,
    random_state=42,
    n_jobs=-1,
    cache_dir=Path("outputs/cache"),
    use_cache=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for fusion training.")

    labels = labels.copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y = labels["dx"].to_numpy()

    if cascade_variant not in STAGE2_VARIANTS:
        raise ValueError(f"Unknown cascade_variant: {cascade_variant}")
    if not 0.0 <= cascade_weight <= 1.0:
        raise ValueError("cascade_weight must be in [0, 1].")

    X_original = _build_training_table(
        data_dir,
        labels,
        ORIGINAL_FEATURE_SET,
        mask_mode,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    original_model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(int(ORIGINAL_K), X_original.shape[1]))),
        ("clf", LogisticRegression(
            C=0.3,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])
    original_model.fit(X_original, y)

    X_stage1 = _build_training_table(
        data_dir,
        labels,
        STAGE1_FEATURE_SET,
        mask_mode,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    y_stage1 = (y == "vasc").astype(int)
    stage1_model = _build_xgb_pipeline(
        STAGE1_XGB,
        n_classes=2,
        k_features=STAGE1_K,
        n_features=X_stage1.shape[1],
        random_state=random_state,
        n_jobs=n_jobs,
    )
    stage1_model.fit(
        X_stage1,
        y_stage1,
        clf__sample_weight=compute_sample_weight("balanced", y_stage1),
    )

    X_stage2_full = _build_training_table(
        data_dir,
        labels,
        STAGE2_FEATURE_SET,
        mask_mode,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )
    mel_nv_mask = y != "vasc"
    X_stage2 = X_stage2_full.loc[mel_nv_mask].reset_index(drop=True)
    y_stage2 = (y[mel_nv_mask] == "nv").astype(int)
    stage2_model = _build_xgb_pipeline(
        STAGE2_VARIANTS[cascade_variant],
        n_classes=2,
        k_features=cascade_k_features,
        n_features=X_stage2.shape[1],
        random_state=random_state,
        n_jobs=n_jobs,
    )
    stage2_model.fit(
        X_stage2,
        y_stage2,
        clf__sample_weight=compute_sample_weight("balanced", y_stage2),
    )

    original_weight = 1.0 - cascade_weight
    bundle = {
        "model_type": "fusion_ensemble",
        "labels": list(LABELS),
        "mask_mode": mask_mode,
        "original_weight": original_weight,
        "cascade_weight": cascade_weight,
        "original": {
            "model": original_model,
            "feature_set": ORIGINAL_FEATURE_SET,
            "feature_columns": list(X_original.columns),
            "k_features": ORIGINAL_K,
            "classifier": "lr03",
        },
        "cascade": {
            "model_type": "xgb_cascade",
            "labels": list(LABELS),
            "mask_mode": mask_mode,
            "cascade_mode": "soft",
            "stage1": {
                "model": stage1_model,
                "feature_set": STAGE1_FEATURE_SET,
                "feature_columns": list(X_stage1.columns),
                "k_features": STAGE1_K,
                "xgb_kwargs": STAGE1_XGB,
            },
            "stage2": {
                "model": stage2_model,
                "feature_set": STAGE2_FEATURE_SET,
                "feature_columns": list(X_stage2_full.columns),
                "k_features": cascade_k_features,
                "xgb_kwargs": STAGE2_VARIANTS[cascade_variant],
                "variant": cascade_variant,
            },
        },
        "metadata": {
            "trained_on_n_samples": int(len(labels)),
            "trained_on_mel_nv_samples": int(mel_nv_mask.sum()),
            "random_state": int(random_state),
            "selection_note": (
                "Default config follows seed127 OOF fusion search: "
                "original all_abcd_grouped LR03 plus d2-more-trees cascade, "
                "cascade_weight=0.5."
            ),
        },
    }

    weight_tag = str(cascade_weight).replace(".", "p")
    model_path = Path(model_path) if model_path else (
        DEFAULT_MODEL_DIR
        / f"fusion_lr03_abcdgrouped_xgbcascade_{cascade_variant}_k{cascade_k_features}_w{weight_tag}.joblib"
    )
    ensure_dir(model_path.parent)
    joblib.dump(bundle, model_path)
    joblib.dump(bundle, DEFAULT_MODEL_DIR / "ml_fusion_candidate.joblib")
    print(f"Saved fusion candidate to {model_path}")
    print(f"Also updated {DEFAULT_MODEL_DIR / 'ml_fusion_candidate.joblib'}")
    return bundle


def main():
    parser = argparse.ArgumentParser(description="Train full-data original+XGB-cascade fusion candidate.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--cascade_variant", default="d2-more-trees", choices=sorted(STAGE2_VARIANTS))
    parser.add_argument("--cascade_k_features", default="all")
    parser.add_argument("--cascade_weight", type=float, default=0.5)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    train_fusion(
        data_dir=Path(args.data_dir),
        mask_mode=args.mask_mode,
        cascade_variant=args.cascade_variant,
        cascade_k_features=args.cascade_k_features,
        cascade_weight=args.cascade_weight,
        model_path=args.model_path,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()

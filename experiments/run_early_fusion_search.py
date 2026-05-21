import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")


def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value):
    return value.replace("+", "_AND_")


def _feature_cache_candidates(cache_dir, feature_set, mask_mode):
    safe = _safe_name(feature_set)
    return [
        Path(cache_dir) / f"features_early_{safe}_{mask_mode}.csv",
        Path(cache_dir) / f"features_fusion_{safe}_{mask_mode}.csv",
        Path(cache_dir) / f"features_cascade_{safe}_{mask_mode}.csv",
        Path(cache_dir) / f"features_{safe}_{mask_mode}.csv",
    ]


def _try_build_from_component_caches(cache_dir, feature_set, mask_mode):
    components_by_feature_set = {
        "early_fusion_core": ["all_abcd_grouped", "xgb_cascade_stage2"],
        "early_fusion_full": ["final_abcd_grouped", "xgb_cascade_stage2"],
    }
    components = components_by_feature_set.get(feature_set)
    if components is None:
        return None

    tables = []
    for component in components:
        candidates = _feature_cache_candidates(cache_dir, component, mask_mode)
        existing = next((path for path in candidates if path.exists()), None)
        if existing is None:
            return None
        table = pd.read_csv(existing)
        table["image_id"] = table["image_id"].astype(str)
        tables.append(table)

    merged = tables[0][["image_id"]].copy()
    seen = {"image_id"}
    new_columns = {}
    for table in tables:
        table = table.set_index("image_id")
        for col in table.columns:
            if col not in seen:
                new_columns[col] = merged["image_id"].map(table[col]).to_numpy()
                seen.add(col)
    if new_columns:
        merged = pd.concat([merged, pd.DataFrame(new_columns)], axis=1)
    return merged


def _load_feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    if use_cache:
        for cache_path in _feature_cache_candidates(cache_dir, feature_set, mask_mode):
            if cache_path.exists():
                return pd.read_csv(cache_path)
        component_table = _try_build_from_component_caches(cache_dir, feature_set, mask_mode)
        if component_table is not None:
            cache_path = _feature_cache_candidates(cache_dir, feature_set, mask_mode)[0]
            ensure_dir(cache_path.parent)
            component_table.to_csv(cache_path, index=False)
            return component_table

    feature_df = build_feature_table(
        data_dir,
        image_ids=labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    if use_cache:
        cache_path = _feature_cache_candidates(cache_dir, feature_set, mask_mode)[0]
        ensure_dir(cache_path.parent)
        feature_df.to_csv(cache_path, index=False)
    return feature_df


def _make_splits(image_ids, y, n_splits, seed):
    groups = image_ids.map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(image_ids.to_frame(name="image_id"), y, groups=groups))
    for fold, (train_idx, valid_idx) in enumerate(splits):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[valid_idx])
        if overlap:
            raise RuntimeError(f"Group leakage detected in fold {fold}: {sorted(overlap)[:5]}")
    return splits


def _classifier_grid(classifier_names, preset, n_jobs):
    if "all" in classifier_names:
        classifier_names = ["lr03", "et", "hgb", "rf_strong", "xgb_strong_reg", "xgb_d2_more_trees"]

    grid = []
    compact = preset == "compact"

    if "lr03" in classifier_names:
        grid.append((
            "lr03",
            "C=0.3,class_weight=balanced",
            LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", random_state=42),
        ))
    if "lr1" in classifier_names:
        grid.append((
            "lr1",
            "C=1.0,class_weight=balanced",
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42),
        ))
    if "et" in classifier_names:
        for max_features in (["sqrt"] if compact else ["sqrt", "log2"]):
            grid.append((
                "et",
                f"n=900,depth=None,max_features={max_features}",
                ExtraTreesClassifier(
                    n_estimators=900,
                    max_depth=None,
                    max_features=max_features,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=n_jobs,
                ),
            ))
    if "rf_strong" in classifier_names:
        grid.append((
            "rf_strong",
            "n=700,depth=None,max_features=sqrt",
            RandomForestClassifier(
                n_estimators=700,
                max_depth=None,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=n_jobs,
            ),
        ))
    if "hgb" in classifier_names:
        params = [(0.06, 15, 220, 0.1)] if compact else [(0.03, 15, 260, 0.1), (0.06, 15, 220, 0.1)]
        for learning_rate, max_leaf_nodes, max_iter, l2 in params:
            grid.append((
                "hgb",
                f"lr={learning_rate},leaves={max_leaf_nodes},iter={max_iter},l2={l2}",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    max_iter=max_iter,
                    l2_regularization=l2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ))
    if XGBClassifier is not None:
        xgb_common = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": n_jobs,
            "verbosity": 0,
        }
        if "xgb_strong_reg" in classifier_names:
            grid.append((
                "xgb_strong_reg",
                "n=150,depth=2,lr=0.05,sub=0.70,col=0.50,lambda=10",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=2,
                    learning_rate=0.05,
                    subsample=0.70,
                    colsample_bytree=0.50,
                    reg_lambda=10.0,
                    **xgb_common,
                ),
            ))
        if "xgb_d2_more_trees" in classifier_names:
            grid.append((
                "xgb_d2_more_trees",
                "n=300,depth=2,lr=0.03,sub=0.80,col=0.60,lambda=5",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=2,
                    learning_rate=0.03,
                    subsample=0.80,
                    colsample_bytree=0.60,
                    reg_lambda=5.0,
                    **xgb_common,
                ),
            ))
        if "xgb_deeper" in classifier_names:
            grid.append((
                "xgb_deeper",
                "n=200,depth=4,lr=0.05,sub=0.80,col=0.60,lambda=5",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.80,
                    colsample_bytree=0.60,
                    reg_lambda=5.0,
                    **xgb_common,
                ),
            ))
    return grid


def _pipeline(classifier, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def _aligned_proba(model, X_valid):
    if not hasattr(model, "predict_proba"):
        return None
    raw = model.predict_proba(X_valid)
    aligned = np.zeros((len(X_valid), len(LABELS)), dtype=float)
    classes = np.asarray(model.classes_)
    for idx, label in enumerate(LABELS):
        if label in classes:
            aligned[:, idx] = raw[:, int(np.where(classes == label)[0][0])]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


def _is_xgb_classifier(classifier_name):
    return classifier_name.startswith("xgb_")


def _metrics_row(y, pred):
    metrics = compute_metrics(y, pred, labels=LABELS)
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    row = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }
    for label in LABELS:
        row[f"{label}_precision"] = report[label]["precision"]
        row[f"{label}_recall"] = report[label]["recall"]
        row[f"{label}_f1"] = report[label]["f1-score"]
    return row


def _weighted_pred(proba, weights):
    weighted = proba * np.asarray(weights, dtype=float)
    return pd.Series(np.asarray(LABELS)[weighted.argmax(axis=1)])


def _tune_class_weights(y, proba):
    best = None
    for mel_w in [0.85, 1.0, 1.1, 1.25]:
        for nv_w in [0.85, 1.0, 1.1]:
            for vasc_w in [0.9, 1.0, 1.15, 1.3]:
                weights = [mel_w, nv_w, vasc_w]
                pred = _weighted_pred(proba, weights)
                row = _metrics_row(y, pred)
                row.update({
                    "postprocess": "class_weight_argmax",
                    "class_weights": ",".join(f"{label}:{weight:g}" for label, weight in zip(LABELS, weights)),
                })
                if best is None or (
                    row["balanced_accuracy"],
                    row["macro_f1"],
                    row["accuracy"],
                ) > (
                    best["balanced_accuracy"],
                    best["macro_f1"],
                    best["accuracy"],
                ):
                    best = row
    return best


def run_early_fusion_search(
    data_dir,
    feature_sets,
    classifiers,
    k_features,
    seeds,
    mask_mode,
    output_csv,
    cache_dir,
    use_cache,
    n_splits,
    preset,
    n_jobs,
    tune_weights,
):
    started = time.time()
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for early fusion search.")
    labels = labels.copy()
    labels["image_id"] = labels["image_id"].astype(str)

    rows = []
    classifier_grid = _classifier_grid(classifiers, preset=preset, n_jobs=n_jobs)
    if not classifier_grid:
        raise ValueError("No classifiers selected or available.")

    for feature_set in feature_sets:
        feature_df = _load_feature_table(
            data_dir=data_dir,
            labels=labels,
            feature_set=feature_set,
            mask_mode=mask_mode,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        feature_df["image_id"] = feature_df["image_id"].astype(str)
        data = feature_df.merge(labels, on="image_id", how="inner")
        if len(data) != len(labels):
            raise ValueError(f"Feature table for {feature_set} is missing samples.")
        X = data.drop(columns=["image_id", "dx"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y = data["dx"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        image_ids = data["image_id"].reset_index(drop=True)
        print(f"\nFeature set {feature_set}: {X.shape[1]} features")

        for seed in seeds:
            splits = _make_splits(image_ids, y, n_splits=n_splits, seed=seed)
            for classifier_name, classifier_params, classifier in classifier_grid:
                for k in k_features:
                    model = _pipeline(clone(classifier), k, X.shape[1])
                    pred = pd.Series(index=y.index, dtype=object)
                    proba = np.zeros((len(y), len(LABELS)), dtype=float)
                    has_proba = True

                    for train_idx, valid_idx in splits:
                        if _is_xgb_classifier(classifier_name):
                            label_to_id = {label: idx for idx, label in enumerate(LABELS)}
                            y_train = y.iloc[train_idx].map(label_to_id).to_numpy()
                            model.fit(X.iloc[train_idx], y_train)
                            pred_ids = model.predict(X.iloc[valid_idx]).astype(int)
                            pred.iloc[valid_idx] = np.asarray(LABELS)[pred_ids]
                            fold_proba = model.predict_proba(X.iloc[valid_idx])
                            if fold_proba.shape[1] != len(LABELS):
                                has_proba = False
                            else:
                                proba[valid_idx] = fold_proba
                        else:
                            model.fit(X.iloc[train_idx], y.iloc[train_idx])
                            pred.iloc[valid_idx] = model.predict(X.iloc[valid_idx])
                            fold_proba = _aligned_proba(model, X.iloc[valid_idx])
                            if fold_proba is None:
                                has_proba = False
                            else:
                                proba[valid_idx] = fold_proba

                    row = {
                        "feature_set": feature_set,
                        "mask_mode": mask_mode,
                        "classifier": classifier_name,
                        "classifier_params": classifier_params,
                        "k_features": k,
                        "n_features": X.shape[1],
                        "seed": seed,
                        "postprocess": "argmax",
                        "class_weights": "",
                        **_metrics_row(y, pred),
                    }
                    rows.append(row)
                    print(
                        f"  seed={seed:<5d} {classifier_name:>17s} k={str(k):>4s} "
                        f"bal={row['balanced_accuracy']:.4f} "
                        f"macro={row['macro_f1']:.4f} acc={row['accuracy']:.4f}",
                        flush=True,
                    )

                    if tune_weights and has_proba:
                        tuned = _tune_class_weights(y, proba)
                        tuned.update({
                            "feature_set": feature_set,
                            "mask_mode": mask_mode,
                            "classifier": classifier_name,
                            "classifier_params": classifier_params,
                            "k_features": k,
                            "n_features": X.shape[1],
                            "seed": seed,
                        })
                        rows.append(tuned)

    result = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "accuracy"],
        ascending=False,
    )
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print("\nTop early-fusion results:")
    print(result.head(20).to_string(index=False))
    print(f"\nSaved results to {output_csv}")
    print(f"Wall time: {(time.time() - started) / 60:.1f} min")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Feature-level early fusion search under strict grouped OOF."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_sets", default="early_fusion_core,early_fusion_full")
    parser.add_argument("--classifiers", default="lr03,et,hgb,xgb_strong_reg,xgb_d2_more_trees")
    parser.add_argument("--k_features", default="100,140,180,220,all")
    parser.add_argument("--seeds", default="127")
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/early_fusion_search.csv")
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--preset", default="compact", choices=["compact", "full"])
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_weight_tuning", action="store_true")
    args = parser.parse_args()

    run_early_fusion_search(
        data_dir=Path(args.data_dir),
        feature_sets=_parse_csv(args.feature_sets),
        classifiers=_parse_csv(args.classifiers),
        k_features=_parse_csv(args.k_features),
        seeds=[int(seed) for seed in _parse_csv(args.seeds)],
        mask_mode=args.mask_mode,
        output_csv=Path(args.output_csv),
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        n_splits=args.n_splits,
        preset=args.preset,
        n_jobs=args.n_jobs,
        tune_weights=not args.no_weight_tuning,
    )


if __name__ == "__main__":
    main()

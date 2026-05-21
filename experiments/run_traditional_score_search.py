import argparse
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache=True):
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"features_{feature_set}_{mask_mode}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)

    feature_df = build_feature_table(
        data_dir,
        labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def _splits(image_ids, y, n_splits, random_state):
    groups = image_ids.map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(image_ids.to_frame(name="image_id"), y, groups=groups))
    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_groups = set(groups.iloc[train_idx])
        valid_groups = set(groups.iloc[valid_idx])
        overlap = train_groups & valid_groups
        if overlap:
            raise RuntimeError(f"Group leakage detected in fold {fold}: {sorted(overlap)[:5]}")
    return splits


def _external_status():
    rows = []
    for name in ["xgboost", "lightgbm", "catboost"]:
        try:
            module = __import__(name)
            rows.append({"package": name, "status": "installed", "version": getattr(module, "__version__", "")})
        except Exception as exc:
            rows.append({"package": name, "status": f"missing:{type(exc).__name__}", "version": ""})
    return pd.DataFrame(rows)


def _classifier_grid(names, preset):
    if "all" in names:
        names = ["lr", "svm", "rf", "rf_strong", "et", "gb", "hgb", "knn"]

    compact = preset == "compact"
    grid = []
    if "lr" in names:
        values = [0.3, 1.0] if compact else [0.1, 0.3, 1.0, 3.0]
        for c_value in values:
            grid.append((
                "lr",
                f"C={c_value}",
                LogisticRegression(C=c_value, max_iter=3000, class_weight="balanced", random_state=42),
            ))
    if "svm" in names:
        values = [30, 100] if compact else [10, 30, 100, 300]
        for c_value in values:
            grid.append((
                "svm",
                f"C={c_value},prob",
                SVC(
                    kernel="rbf",
                    C=c_value,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ))
    if "rf" in names:
        depths = [15] if compact else [10, 15, None]
        for depth in depths:
            grid.append((
                "rf",
                f"n=500,depth={depth},sqrt",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=depth,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ))
    if "rf_strong" in names:
        grid.append((
            "rf_strong",
            "n=900,depth=None,log2",
            RandomForestClassifier(
                n_estimators=900,
                max_depth=None,
                max_features="log2",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ))
    if "et" in names:
        depths = [None] if compact else [12, None]
        for depth in depths:
            grid.append((
                "et",
                f"n=900,depth={depth},sqrt",
                ExtraTreesClassifier(
                    n_estimators=900,
                    max_depth=depth,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ))
    if "gb" in names:
        params = [(0.05, 2, 200)] if compact else [(0.03, 2, 300), (0.05, 2, 200), (0.05, 3, 160)]
        for lr, depth, n_estimators in params:
            grid.append((
                "gb",
                f"lr={lr},depth={depth},n={n_estimators}",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=42,
                ),
            ))
    if "hgb" in names:
        params = [(0.06, 15, 200, 0.1)] if compact else [(0.03, 15, 260, 0.1), (0.06, 15, 200, 0.1), (0.08, 31, 160, 0.0)]
        for lr, leaves, max_iter, l2 in params:
            grid.append((
                "hgb",
                f"lr={lr},leaves={leaves},iter={max_iter},l2={l2}",
                HistGradientBoostingClassifier(
                    learning_rate=lr,
                    max_iter=max_iter,
                    max_leaf_nodes=leaves,
                    l2_regularization=l2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ))
    if "knn" in names:
        values = [7] if compact else [5, 7, 11]
        for neighbors in values:
            grid.append((
                "knn",
                f"k={neighbors},distance",
                KNeighborsClassifier(n_neighbors=neighbors, weights="distance"),
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
    class_to_idx = {label: idx for idx, label in enumerate(model.classes_)}
    for col, label in enumerate(LABELS):
        if label in class_to_idx:
            aligned[:, col] = raw[:, class_to_idx[label]]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


def _evaluate_oof(X, y, splits, classifier, k_features):
    pred = pd.Series(index=y.index, dtype=object)
    proba = np.zeros((len(y), len(LABELS)), dtype=float)
    has_proba = True

    for train_idx, valid_idx in splits:
        model = _pipeline(clone(classifier), k_features, X.shape[1])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[valid_idx] = model.predict(X.iloc[valid_idx])
        fold_proba = _aligned_proba(model, X.iloc[valid_idx])
        if fold_proba is None:
            has_proba = False
        else:
            proba[valid_idx] = fold_proba

    metrics = compute_metrics(y, pred, labels=LABELS)
    return metrics, pred, proba if has_proba else None


def _per_class_metrics(y, pred):
    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    row = {}
    for label in LABELS:
        row[f"{label}_precision"] = report[label]["precision"]
        row[f"{label}_recall"] = report[label]["recall"]
        row[f"{label}_f1"] = report[label]["f1-score"]
    return row


def _pred_from_weighted_proba(proba, weights):
    weighted = proba * np.asarray(weights, dtype=float)
    return pd.Series(np.asarray(LABELS)[weighted.argmax(axis=1)])


def _tune_class_weights(y, proba, weight_values):
    best = None
    for weights in itertools.product(weight_values, repeat=len(LABELS)):
        pred = _pred_from_weighted_proba(proba, weights)
        metrics = compute_metrics(y, pred, labels=LABELS)
        row = {
            "postprocess": "class_weight_argmax",
            "class_weights": ",".join(f"{label}:{weight:g}" for label, weight in zip(LABELS, weights)),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            **_per_class_metrics(y, pred),
        }
        if best is None or (row["accuracy"], row["macro_f1"]) > (best["accuracy"], best["macro_f1"]):
            best = row
    return best


def _ensemble_rows(y, proba_store, base_rows, sizes):
    rows = []
    ranked = [row for row in base_rows if row["run_id"] in proba_store]
    ranked = sorted(ranked, key=lambda row: (row["accuracy"], row["macro_f1"]), reverse=True)
    for size in sizes:
        selected = ranked[: min(size, len(ranked))]
        if len(selected) < 2:
            continue
        proba = np.mean([proba_store[row["run_id"]] for row in selected], axis=0)
        pred = _pred_from_weighted_proba(proba, [1.0, 1.0, 1.0])
        metrics = compute_metrics(y, pred, labels=LABELS)
        rows.append({
            "run_id": f"ensemble_top{len(selected)}",
            "feature_set": "ensemble",
            "mask_mode": "mixed",
            "classifier": "soft_vote",
            "classifier_params": ";".join(row["run_id"] for row in selected),
            "k_features": "mixed",
            "n_features": "",
            "postprocess": "soft_vote",
            "class_weights": "",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            **_per_class_metrics(y, pred),
        })
    return rows


def _write_markdown(result, external_status, output_md):
    top = result.sort_values(["accuracy", "macro_f1"], ascending=False).head(15)
    lines = [
        "# Traditional ML Score Search",
        "",
        "All results use grouped CV by original lesion id. The search avoids augmentation leakage.",
        "",
        "## Optional External Packages",
        "",
        "| package | status | version |",
        "|---|---|---|",
    ]
    for _, row in external_status.iterrows():
        lines.append(f"| {row['package']} | {row['status']} | {row['version']} |")
    lines.extend([
        "",
        "## Top Results",
        "",
        "| rank | run | feature set | mask | classifier | k | postprocess | accuracy | macro-F1 | balanced acc |",
        "|---:|---|---|---|---|---:|---|---:|---:|---:|",
    ])
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['run_id']}` | `{row['feature_set']}` | `{row['mask_mode']}` | "
            f"`{row['classifier']}` | `{row['k_features']}` | `{row.get('postprocess', '')}` | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['balanced_accuracy']:.4f} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def _write_partial(rows, output_dir):
    if not rows:
        return
    partial = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    partial.to_csv(output_dir / "traditional_score_search_partial.csv", index=False)


def run_search(
    data_dir,
    feature_sets,
    classifiers,
    k_features,
    mask_modes,
    output_dir,
    n_splits=5,
    random_state=127,
    preset="compact",
    max_combos=None,
    use_cache=True,
    tune_weights=True,
    run_ensembles=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for score search.")
    labels["image_id"] = labels["image_id"].astype(str)
    ensure_dir(output_dir)
    external_status = _external_status()
    external_status.to_csv(output_dir / "external_package_status.csv", index=False)

    rows = []
    proba_store = {}
    cache_dir = Path("outputs/cache")
    classifier_grid = _classifier_grid(classifiers, preset=preset)
    combo_count = 0
    for mask_mode in mask_modes:
        for feature_set in feature_sets:
            feature_df = _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache)
            data = feature_df.merge(labels, on="image_id", how="inner")
            X = data.drop(columns=["image_id", "dx"])
            y = data["dx"].reset_index(drop=True)
            X = X.reset_index(drop=True)
            image_ids = data["image_id"].reset_index(drop=True)
            splits = _splits(image_ids, y, n_splits=n_splits, random_state=random_state)

            for classifier_name, classifier_params, classifier in classifier_grid:
                for k_value in k_features:
                    if max_combos is not None and combo_count >= max_combos:
                        break
                    combo_count += 1
                    run_id = f"{feature_set}__{mask_mode}__{classifier_name}__{combo_count:04d}"
                    metrics, pred, proba = _evaluate_oof(X, y, splits, classifier, k_value)
                    row = {
                        "run_id": run_id,
                        "feature_set": feature_set,
                        "mask_mode": mask_mode,
                        "classifier": classifier_name,
                        "classifier_params": classifier_params,
                        "k_features": k_value,
                        "n_features": X.shape[1],
                        "postprocess": "argmax",
                        "class_weights": "",
                        "n_splits": n_splits,
                        "random_state": random_state,
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        **_per_class_metrics(y, pred),
                    }
                    rows.append(row)
                    if proba is not None:
                        proba_store[run_id] = proba
                        if tune_weights:
                            tuned = _tune_class_weights(y, proba, [0.75, 0.9, 1.0, 1.1, 1.25])
                            tuned.update({
                                **{key: row[key] for key in [
                                    "feature_set", "mask_mode", "classifier", "classifier_params",
                                    "k_features", "n_features", "n_splits", "random_state",
                                ]},
                                "run_id": run_id + "__weights",
                            })
                            rows.append(tuned)
                    _write_partial(rows, output_dir)
                    print(
                        f"{run_id:55s} k={str(k_value):>4s} "
                        f"acc={metrics['accuracy']:.4f} macro={metrics['macro_f1']:.4f}",
                        flush=True,
                    )
                if max_combos is not None and combo_count >= max_combos:
                    break
            if max_combos is not None and combo_count >= max_combos:
                break
        if max_combos is not None and combo_count >= max_combos:
            break

    if run_ensembles:
        rows.extend(_ensemble_rows(labels["dx"].reset_index(drop=True), proba_store, rows, [3, 5, 10]))

    result = pd.DataFrame(rows).sort_values(["accuracy", "macro_f1"], ascending=False)
    result.to_csv(output_dir / "traditional_score_search_results.csv", index=False)
    _write_markdown(result, external_status, output_dir / "traditional_score_search_summary.md")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_sets", default="all_abcd_grouped,final_abcd_grouped,all_boundary,final_melnv,all_boundary_melnv")
    parser.add_argument("--classifiers", default="lr,svm,rf,rf_strong,et,gb,hgb")
    parser.add_argument("--k_features", default="80,100,120,140,180,all")
    parser.add_argument("--mask_modes", default="raw,clean")
    parser.add_argument("--output_dir", default="outputs/metrics/traditional_score_search")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--preset", default="compact", choices=["compact", "full"])
    parser.add_argument("--max_combos", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--no_weight_tuning", action="store_true")
    parser.add_argument("--no_ensembles", action="store_true")
    args = parser.parse_args()

    result = run_search(
        data_dir=Path(args.data_dir),
        feature_sets=_parse_list(args.feature_sets),
        classifiers=_parse_list(args.classifiers),
        k_features=_parse_list(args.k_features),
        mask_modes=_parse_list(args.mask_modes),
        output_dir=Path(args.output_dir),
        n_splits=args.n_splits,
        random_state=args.random_state,
        preset=args.preset,
        max_combos=args.max_combos,
        use_cache=not args.no_cache,
        tune_weights=not args.no_weight_tuning,
        run_ensembles=not args.no_ensembles,
    )
    print("\nTop results:")
    print(result.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

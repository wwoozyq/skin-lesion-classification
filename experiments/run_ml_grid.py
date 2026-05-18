import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
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


def _parse_list(value):
    if value is None:
        return []
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


def _splits(data, n_splits, random_state):
    y = data["dx"]
    groups = data["image_id"].map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(data.drop(columns=["dx"]), y, groups=groups))


def _classifier_grid(names):
    grid = []
    if "svm" in names:
        for c_value in [10, 30, 100]:
            grid.append((
                "svm",
                f"C={c_value},gamma=scale",
                SVC(kernel="rbf", C=c_value, gamma="scale", class_weight="balanced", random_state=42),
            ))
    if "rf" in names:
        for max_depth in [10, 15]:
            grid.append((
                "rf",
                f"n=300,depth={max_depth}",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=max_depth,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ))
    if "lr" in names:
        for c_value in [0.3, 1.0]:
            grid.append((
                "lr",
                f"C={c_value}",
                LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced", random_state=42),
            ))
    if "knn" in names:
        for n_neighbors in [5, 9]:
            grid.append((
                "knn",
                f"k={n_neighbors},distance",
                KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
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


def run_grid(
    data_dir,
    feature_sets,
    classifiers,
    k_features,
    mask_modes,
    output_csv,
    n_splits=5,
    random_state=127,
    use_cache=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for grid search.")
    labels["image_id"] = labels["image_id"].astype(str)

    rows = []
    cache_dir = Path("outputs/cache")
    for mask_mode in mask_modes:
        for feature_set in feature_sets:
            feature_df = _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache)
            data = feature_df.merge(labels, on="image_id", how="inner")
            X = data.drop(columns=["image_id", "dx"])
            y = data["dx"]
            splits = _splits(data[["image_id", "dx"]], n_splits=n_splits, random_state=random_state)

            for classifier_name, classifier_params, classifier in _classifier_grid(classifiers):
                for k_value in k_features:
                    model = _pipeline(classifier, k_value, X.shape[1])
                    pred = pd.Series(index=data.index, dtype=object)
                    for train_idx, valid_idx in splits:
                        model.fit(X.iloc[train_idx], y.iloc[train_idx])
                        pred.iloc[valid_idx] = model.predict(X.iloc[valid_idx])
                    metrics = compute_metrics(y, pred, labels=LABELS)
                    row = {
                        "feature_set": feature_set,
                        "mask_mode": mask_mode,
                        "classifier": classifier_name,
                        "classifier_params": classifier_params,
                        "k_features": k_value,
                        "n_features": X.shape[1],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                    }
                    rows.append(row)
                    print(
                        f"{feature_set:18s} {mask_mode:5s} {classifier_name:3s} "
                        f"{classifier_params:18s} k={k_value:>4s} "
                        f"macro_f1={metrics['macro_f1']:.4f}"
                    )

    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_sets", default="all,all_contrast,all_abcd_v2,all_boundary,final")
    parser.add_argument("--classifiers", default="svm,rf,lr,knn")
    parser.add_argument("--k_features", default="all,60,100")
    parser.add_argument("--mask_modes", default="raw")
    parser.add_argument("--output_csv", default="outputs/metrics/ml_grid_search.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    result = run_grid(
        data_dir=Path(args.data_dir),
        feature_sets=_parse_list(args.feature_sets),
        classifiers=_parse_list(args.classifiers),
        k_features=_parse_list(args.k_features),
        mask_modes=_parse_list(args.mask_modes),
        output_csv=Path(args.output_csv),
        n_splits=args.n_splits,
        random_state=args.random_state,
        use_cache=not args.no_cache,
    )
    print("\nTop results:")
    print(result.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

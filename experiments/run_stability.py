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


def _classifier(name):
    if name == "svm":
        return SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced", random_state=42)
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    raise ValueError(f"Unknown classifier={name}")


def _pipeline(classifier_name, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", _classifier(classifier_name)))
    return Pipeline(steps)


def _evaluate_seed(X, y, image_ids, classifier_name, k_features, n_splits, seed):
    groups = image_ids.map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = pd.Series(index=y.index, dtype=object)

    for train_idx, valid_idx in cv.split(X, y, groups=groups):
        model = _pipeline(classifier_name, k_features, X.shape[1])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[valid_idx] = model.predict(X.iloc[valid_idx])

    return compute_metrics(y, pred, labels=LABELS)


def run_stability(
    data_dir,
    feature_sets,
    classifiers,
    k_features,
    mask_modes,
    seeds,
    output_csv,
    n_splits=5,
    use_cache=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for stability evaluation.")
    labels["image_id"] = labels["image_id"].astype(str)

    rows = []
    cache_dir = Path("outputs/cache")
    for mask_mode in mask_modes:
        for feature_set in feature_sets:
            feature_df = _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache)
            data = feature_df.merge(labels, on="image_id", how="inner")
            X = data.drop(columns=["image_id", "dx"])
            y = data["dx"]
            image_ids = data["image_id"]

            for classifier_name in classifiers:
                for k_value in k_features:
                    for seed in seeds:
                        metrics = _evaluate_seed(
                            X=X,
                            y=y,
                            image_ids=image_ids,
                            classifier_name=classifier_name,
                            k_features=k_value,
                            n_splits=n_splits,
                            seed=int(seed),
                        )
                        row = {
                            "feature_set": feature_set,
                            "mask_mode": mask_mode,
                            "classifier": classifier_name,
                            "k_features": k_value,
                            "n_features": X.shape[1],
                            "seed": int(seed),
                            "accuracy": metrics["accuracy"],
                            "macro_f1": metrics["macro_f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                        }
                        rows.append(row)
                        print(
                            f"{feature_set:22s} {mask_mode:5s} {classifier_name:3s} "
                            f"k={k_value:>4s} seed={int(seed):>4d} "
                            f"macro_f1={metrics['macro_f1']:.4f}"
                        )

    result = pd.DataFrame(rows)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)

    summary = (
        result
        .groupby(["feature_set", "mask_mode", "classifier", "k_features", "n_features"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
        .sort_values(["macro_f1_mean", "balanced_accuracy_mean"], ascending=False)
    )
    summary_path = Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_sets", default="all_boundary,all_boundary_melnv")
    parser.add_argument("--classifiers", default="lr")
    parser.add_argument("--k_features", default="100")
    parser.add_argument("--mask_modes", default="raw")
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--output_csv", default="outputs/metrics/stability.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    _, summary = run_stability(
        data_dir=Path(args.data_dir),
        feature_sets=_parse_list(args.feature_sets),
        classifiers=_parse_list(args.classifiers),
        k_features=_parse_list(args.k_features),
        mask_modes=_parse_list(args.mask_modes),
        seeds=_parse_list(args.seeds),
        output_csv=Path(args.output_csv),
        n_splits=args.n_splits,
        use_cache=not args.no_cache,
    )
    print("\nStability summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

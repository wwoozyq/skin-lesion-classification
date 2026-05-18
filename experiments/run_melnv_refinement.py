import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
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


PIGMENT_LABELS = {"mel", "nv"}


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
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=42)
    if name == "svm":
        return SVC(kernel="rbf", C=100, gamma="scale", class_weight="balanced", random_state=42)
    raise ValueError(f"Unknown classifier={name}")


def _pipeline(classifier_name, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", _classifier(classifier_name)))
    return Pipeline(steps)


def _evaluate_refinement(
    main_X,
    ref_X,
    y,
    image_ids,
    main_classifier,
    main_k,
    ref_classifier,
    ref_k,
    seed,
    n_splits,
):
    groups = image_ids.map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    main_pred = pd.Series(index=y.index, dtype=object)
    refined_pred = pd.Series(index=y.index, dtype=object)

    for train_idx, valid_idx in cv.split(main_X, y, groups=groups):
        main_model = _pipeline(main_classifier, main_k, main_X.shape[1])
        main_model.fit(main_X.iloc[train_idx], y.iloc[train_idx])
        fold_main_pred = pd.Series(main_model.predict(main_X.iloc[valid_idx]), index=valid_idx)
        fold_refined_pred = fold_main_pred.copy()

        train_y = y.iloc[train_idx]
        pigment_train = train_y.isin(PIGMENT_LABELS).to_numpy()
        ref_model = _pipeline(ref_classifier, ref_k, ref_X.shape[1])
        ref_model.fit(ref_X.iloc[train_idx].iloc[pigment_train], train_y.iloc[pigment_train])

        pigment_predicted = fold_main_pred.isin(PIGMENT_LABELS)
        if pigment_predicted.any():
            replace_idx = fold_main_pred.index[pigment_predicted]
            fold_refined_pred.loc[replace_idx] = ref_model.predict(ref_X.iloc[replace_idx])

        main_pred.iloc[valid_idx] = fold_main_pred
        refined_pred.iloc[valid_idx] = fold_refined_pred

    return compute_metrics(y, main_pred, labels=LABELS), compute_metrics(y, refined_pred, labels=LABELS)


def run_refinement(
    data_dir,
    main_feature_set,
    main_classifier,
    main_k,
    ref_feature_sets,
    ref_classifiers,
    ref_k_features,
    mask_mode,
    seeds,
    output_csv,
    n_splits=5,
    use_cache=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for mel/nv refinement.")
    labels["image_id"] = labels["image_id"].astype(str)

    cache_dir = Path("outputs/cache")
    main_df = _feature_table(data_dir, labels, main_feature_set, mask_mode, cache_dir, use_cache)
    main_data = main_df.merge(labels, on="image_id", how="inner")
    main_X = main_data.drop(columns=["image_id", "dx"])
    y = main_data["dx"]
    image_ids = main_data["image_id"]

    rows = []
    for ref_feature_set in ref_feature_sets:
        ref_df = _feature_table(data_dir, labels, ref_feature_set, mask_mode, cache_dir, use_cache)
        ref_data = ref_df.merge(labels[["image_id", "dx"]], on="image_id", how="inner")
        ref_X = ref_data.drop(columns=["image_id", "dx"])

        for ref_classifier in ref_classifiers:
            for ref_k in ref_k_features:
                for seed in seeds:
                    main_metrics, refined_metrics = _evaluate_refinement(
                        main_X=main_X,
                        ref_X=ref_X,
                        y=y,
                        image_ids=image_ids,
                        main_classifier=main_classifier,
                        main_k=main_k,
                        ref_classifier=ref_classifier,
                        ref_k=ref_k,
                        seed=int(seed),
                        n_splits=n_splits,
                    )
                    row = {
                        "main_feature_set": main_feature_set,
                        "main_classifier": main_classifier,
                        "main_k_features": main_k,
                        "ref_feature_set": ref_feature_set,
                        "ref_classifier": ref_classifier,
                        "ref_k_features": ref_k,
                        "mask_mode": mask_mode,
                        "seed": int(seed),
                        "main_accuracy": main_metrics["accuracy"],
                        "main_macro_f1": main_metrics["macro_f1"],
                        "main_balanced_accuracy": main_metrics["balanced_accuracy"],
                        "refined_accuracy": refined_metrics["accuracy"],
                        "refined_macro_f1": refined_metrics["macro_f1"],
                        "refined_balanced_accuracy": refined_metrics["balanced_accuracy"],
                        "macro_f1_delta": refined_metrics["macro_f1"] - main_metrics["macro_f1"],
                    }
                    rows.append(row)
                    print(
                        f"{ref_feature_set:22s} {ref_classifier:3s} k={ref_k:>4s} "
                        f"seed={int(seed):>4d} main={main_metrics['macro_f1']:.4f} "
                        f"refined={refined_metrics['macro_f1']:.4f} "
                        f"delta={row['macro_f1_delta']:+.4f}"
                    )

    result = pd.DataFrame(rows)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    summary = (
        result
        .groupby(["ref_feature_set", "ref_classifier", "ref_k_features", "mask_mode"])
        .agg(
            main_macro_f1_mean=("main_macro_f1", "mean"),
            refined_macro_f1_mean=("refined_macro_f1", "mean"),
            macro_f1_delta_mean=("macro_f1_delta", "mean"),
            macro_f1_delta_std=("macro_f1_delta", "std"),
            refined_accuracy_mean=("refined_accuracy", "mean"),
            refined_balanced_accuracy_mean=("refined_balanced_accuracy", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_delta_mean", ascending=False)
    )
    summary_path = Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--main_feature_set", default="all_boundary")
    parser.add_argument("--main_classifier", default="lr", choices=["lr", "svm"])
    parser.add_argument("--main_k_features", default="100")
    parser.add_argument("--ref_feature_sets", default="all_boundary,all_boundary_melnv,final_melnv")
    parser.add_argument("--ref_classifiers", default="lr,svm")
    parser.add_argument("--ref_k_features", default="80,100,160")
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--output_csv", default="outputs/metrics/melnv_refinement.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    _, summary = run_refinement(
        data_dir=Path(args.data_dir),
        main_feature_set=args.main_feature_set,
        main_classifier=args.main_classifier,
        main_k=args.main_k_features,
        ref_feature_sets=_parse_list(args.ref_feature_sets),
        ref_classifiers=_parse_list(args.ref_classifiers),
        ref_k_features=_parse_list(args.ref_k_features),
        mask_mode=args.mask_mode,
        seeds=_parse_list(args.seeds),
        output_csv=Path(args.output_csv),
        n_splits=args.n_splits,
        use_cache=not args.no_cache,
    )
    print("\nRefinement summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

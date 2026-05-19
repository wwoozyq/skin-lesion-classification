import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DEFAULT_METRICS_DIR, DEFAULT_MODEL_DIR, LABELS
from .dataset import load_labels
from .evaluate import compute_metrics
from .features import build_feature_table
from .utils import base_id_from_image_id, ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")

CLASSIFIERS = {
    "svm": SVC(kernel="rbf", C=100, class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
    "lr": LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=42),
    "lr03": LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced", random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
}

FEATURE_SETS = [
    "all",
    "color",
    "shape",
    "texture",
    "contrast",
    "all_contrast",
    "abcd_v2",
    "all_abcd_v2",
    "boundary",
    "all_boundary",
    "melnv",
    "all_melnv",
    "all_boundary_melnv",
    "all_abcd_v2_boundary",
    "abcd_grouped",
    "all_abcd_grouped",
    "final",
    "final_melnv",
    "final_abcd_grouped",
]


def _predict_with_mel_threshold(probabilities, class_labels, mel_threshold):
    class_labels = np.asarray(class_labels)
    if "mel" not in class_labels:
        raise ValueError("Cannot apply mel threshold: class labels do not contain 'mel'.")
    mel_idx = int(np.where(class_labels == "mel")[0][0])
    non_mel_indices = np.asarray([idx for idx, label in enumerate(class_labels) if label != "mel"])
    non_mel_best = class_labels[non_mel_indices[np.argmax(probabilities[:, non_mel_indices], axis=1)]]
    return np.where(probabilities[:, mel_idx] >= mel_threshold, "mel", non_mel_best)


def _make_cv_splits(X, y, image_ids, cv_method, n_splits, random_state):
    if cv_method == "grouped":
        groups = image_ids.map(base_id_from_image_id)
        cv = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(cv.split(X, y, groups=groups))
        split_rows = []
        for fold, (_, valid_idx) in enumerate(splits):
            for row_idx in valid_idx:
                split_rows.append({
                    "row_idx": int(row_idx),
                    "image_id": image_ids.iloc[row_idx],
                    "base_id": groups.iloc[row_idx],
                    "dx": y.iloc[row_idx],
                    "fold": fold,
                })
        return splits, pd.DataFrame(split_rows)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(X, y))
    split_rows = []
    for fold, (_, valid_idx) in enumerate(splits):
        for row_idx in valid_idx:
            split_rows.append({
                "row_idx": int(row_idx),
                "image_id": image_ids.iloc[row_idx],
                "base_id": base_id_from_image_id(image_ids.iloc[row_idx]),
                "dx": y.iloc[row_idx],
                "fold": fold,
            })
    return splits, pd.DataFrame(split_rows)


def train_ml(
    data_dir,
    feature_set="all",
    k_features="all",
    classifier="svm",
    cv_method="grouped",
    n_splits=5,
    random_state=127,
    mask_mode="raw",
    mel_threshold=None,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for training.")

    feature_df = build_feature_table(
        data_dir,
        labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    data = feature_df.merge(labels, on="image_id", how="inner")
    X = data.drop(columns=["image_id", "dx"])
    y = data["dx"]
    image_ids = data["image_id"]

    k = int(k_features) if k_features != "all" else "all"
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", CLASSIFIERS[classifier]),
    ])

    splits, split_df = _make_cv_splits(
        X=X,
        y=y,
        image_ids=image_ids,
        cv_method=cv_method,
        n_splits=n_splits,
        random_state=random_state,
    )
    if mel_threshold is None:
        pred = cross_val_predict(model, X, y, cv=splits)
    else:
        if not hasattr(model, "predict_proba"):
            raise ValueError("mel_threshold requires a classifier with predict_proba.")
        probabilities = cross_val_predict(model, X, y, cv=splits, method="predict_proba")
        pred = _predict_with_mel_threshold(probabilities, LABELS, mel_threshold)
    metrics = compute_metrics(y, pred, labels=LABELS)

    print("CV:", cv_method)
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print(metrics["classification_report"])

    model.fit(X, y)
    ensure_dir(DEFAULT_MODEL_DIR)
    ensure_dir(DEFAULT_METRICS_DIR)
    threshold_suffix = "" if mel_threshold is None else f"_melthr{str(mel_threshold).replace('.', '')}"
    run_name = f"ml_{feature_set}_{classifier}_{cv_method}_{mask_mode}_seed{random_state}{threshold_suffix}"
    model_path = DEFAULT_MODEL_DIR / f"{run_name}.joblib"
    model_bundle = {
        "model": model,
        "feature_set": feature_set,
        "feature_columns": list(X.columns),
        "mask_mode": mask_mode,
        "cv_method": cv_method,
        "classifier": classifier,
    }
    if mel_threshold is not None:
        model_bundle["mel_threshold"] = mel_threshold
    joblib.dump(model_bundle, model_path)
    joblib.dump(model_bundle, DEFAULT_MODEL_DIR / "ml_baseline.joblib")

    metric_df = pd.DataFrame([{
        "feature_set": feature_set,
        "classifier": classifier,
        "cv_method": cv_method,
        "n_splits": n_splits,
        "random_state": random_state,
        "k_features": k_features,
        "mask_mode": mask_mode,
        "mel_threshold": mel_threshold if mel_threshold is not None else "",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }])
    metric_df.to_csv(DEFAULT_METRICS_DIR / f"{run_name}_metrics.csv", index=False)
    metric_df.to_csv(DEFAULT_METRICS_DIR / "ml_baseline_metrics.csv", index=False)
    split_df.to_csv(DEFAULT_METRICS_DIR / f"{run_name}_splits.csv", index=False)
    split_df.to_csv(DEFAULT_METRICS_DIR / "ml_baseline_splits.csv", index=False)
    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=LABELS,
        columns=LABELS,
    )
    confusion_df.to_csv(DEFAULT_METRICS_DIR / f"{run_name}_confusion_matrix.csv")
    confusion_df.to_csv(DEFAULT_METRICS_DIR / "ml_baseline_confusion_matrix.csv")
    report_path = DEFAULT_METRICS_DIR / f"{run_name}_classification_report.txt"
    report_path.write_text(metrics["classification_report"])
    (DEFAULT_METRICS_DIR / "ml_baseline_classification_report.txt").write_text(metrics["classification_report"])
    prediction_df = split_df.copy()
    pred_by_row = pd.Series(pred, index=range(len(pred)))
    prediction_df["pred"] = prediction_df["row_idx"].map(pred_by_row)
    prediction_df["correct"] = prediction_df["dx"] == prediction_df["pred"]
    prediction_df = prediction_df.sort_values(["fold", "image_id"])
    prediction_df.to_csv(DEFAULT_METRICS_DIR / f"{run_name}_oof_predictions.csv", index=False)
    prediction_df.to_csv(DEFAULT_METRICS_DIR / "ml_baseline_oof_predictions.csv", index=False)

    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_set", default="all", choices=FEATURE_SETS)
    parser.add_argument("--k_features", default="all",
                        help="Number of top features to keep via f_classif, or 'all'.")
    parser.add_argument("--classifier", default="svm", choices=list(CLASSIFIERS.keys()),
                        help="Classifier to use: svm (default), rf, lr.")
    parser.add_argument("--cv", default="grouped", choices=["grouped", "stratified"],
                        help="Cross-validation protocol. Use grouped to keep augmentations together.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--mel_threshold", type=float, default=None,
                        help="Optional decision threshold for the mel class. Use only with predict_proba classifiers.")
    args = parser.parse_args()
    train_ml(
        Path(args.data_dir),
        args.feature_set,
        args.k_features,
        args.classifier,
        args.cv,
        args.n_splits,
        args.random_state,
        args.mask_mode,
        args.mel_threshold,
    )


if __name__ == "__main__":
    main()

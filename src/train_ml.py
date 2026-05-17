import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DEFAULT_METRICS_DIR, DEFAULT_MODEL_DIR, LABELS
from .dataset import load_labels
from .evaluate import compute_metrics
from .features import build_feature_table
from .utils import ensure_dir


CLASSIFIERS = {
    "svm": SVC(kernel="rbf", C=100, class_weight="balanced", random_state=42),
    "rf": RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1),
    "lr": LogisticRegression(C=1, max_iter=1000, class_weight="balanced", random_state=42),
}


def train_ml(data_dir, feature_set="all", k_features="all", classifier="svm"):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for training.")

    feature_df = build_feature_table(data_dir, labels["image_id"].astype(str), feature_set)
    data = feature_df.merge(labels, on="image_id", how="inner")
    X = data.drop(columns=["image_id", "dx"])
    y = data["dx"]

    k = int(k_features) if k_features != "all" else "all"
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k)),
        ("clf", CLASSIFIERS[classifier]),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv)
    metrics = compute_metrics(y, pred, labels=LABELS)

    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print(metrics["classification_report"])

    model.fit(X, y)
    ensure_dir(DEFAULT_MODEL_DIR)
    ensure_dir(DEFAULT_METRICS_DIR)
    model_path = DEFAULT_MODEL_DIR / "ml_baseline.joblib"
    joblib.dump({
        "model": model,
        "feature_set": feature_set,
        "feature_columns": list(X.columns),
    }, model_path)

    pd.DataFrame([{
        "feature_set": feature_set,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }]).to_csv(DEFAULT_METRICS_DIR / "ml_baseline_metrics.csv", index=False)

    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_set", default="all", choices=["all", "color", "shape", "texture"])
    parser.add_argument("--k_features", default="all",
                        help="Number of top features to keep via f_classif, or 'all'.")
    parser.add_argument("--classifier", default="svm", choices=list(CLASSIFIERS.keys()),
                        help="Classifier to use: svm (default), rf, lr.")
    args = parser.parse_args()
    train_ml(Path(args.data_dir), args.feature_set, args.k_features, args.classifier)


if __name__ == "__main__":
    main()

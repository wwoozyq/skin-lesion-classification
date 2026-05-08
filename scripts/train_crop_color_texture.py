import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.evaluate import compute_metrics
from src.features_color import extract_color_features
from src.features_texture import extract_texture_features


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_map(image_dir):
    paths = {}
    for path in Path(image_dir).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths[path.stem] = path
    return paths


def extract_crop_features(image_path, feature_set):
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"))
    mask = np.ones(arr.shape[:2], dtype=bool)

    features = {}
    if feature_set in {"color", "color_texture"}:
        features.update(extract_color_features(arr, mask))
    if feature_set in {"texture", "color_texture"}:
        features.update(extract_texture_features(arr, mask))
    return features


def build_crop_feature_table(image_dir, label_csv, feature_set):
    labels = pd.read_csv(label_csv)
    labels["image_id"] = labels["image_id"].astype(str)
    images = image_map(image_dir)

    rows = []
    for image_id in labels["image_id"]:
        if image_id not in images:
            continue
        row = {"image_id": image_id}
        row.update(extract_crop_features(images[image_id], feature_set))
        rows.append(row)

    if not rows:
        raise ValueError("No labeled images were found in image_dir.")

    feature_df = pd.DataFrame(rows)
    return feature_df.merge(labels, on="image_id", how="inner")


def train_crop_ml(image_dir, label_csv, feature_set, output_dir):
    data = build_crop_feature_table(image_dir, label_csv, feature_set)
    X = data.drop(columns=["image_id", "dx"])
    y = data["dx"]

    min_class_count = int(y.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Each class needs at least 2 samples for cross validation.")
    n_splits = min(5, min_class_count)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv)
    metrics = compute_metrics(y, pred, labels=LABELS)

    print("Samples:", len(data))
    print("Feature set:", feature_set)
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print(metrics["classification_report"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "dataset": str(image_dir),
        "feature_set": feature_set,
        "samples": len(data),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }]).to_csv(output_dir / "crop_color_texture_metrics.csv", index=False)

    model.fit(X, y)
    joblib.dump({
        "model": model,
        "feature_set": feature_set,
        "feature_columns": list(X.columns),
    }, output_dir / "crop_color_texture_model.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--feature_set", default="color_texture", choices=["color", "texture", "color_texture"])
    parser.add_argument("--output_dir", default="outputs/external")
    args = parser.parse_args()

    train_crop_ml(
        image_dir=Path(args.image_dir),
        label_csv=Path(args.label_csv),
        feature_set=args.feature_set,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

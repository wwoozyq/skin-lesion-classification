import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_image, load_labels, load_mask
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


def extract_features(image, mask, feature_set):
    features = {}
    if feature_set in {"color", "color_texture"}:
        features.update(extract_color_features(image, mask))
    if feature_set in {"texture", "color_texture"}:
        features.update(extract_texture_features(image, mask))
    return features


def build_course_features(data_dir, feature_set):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("Training data requires label.csv.")

    rows = []
    for image_id in labels["image_id"].astype(str):
        image = load_image(data_dir, image_id)
        mask = load_mask(data_dir, image_id)
        row = {"image_id": image_id}
        row.update(extract_features(image, mask, feature_set))
        rows.append(row)

    return pd.DataFrame(rows).merge(labels, on="image_id", how="inner")


def build_external_crop_features(image_dir, label_csv, feature_set):
    labels = pd.read_csv(label_csv)
    labels["image_id"] = labels["image_id"].astype(str)
    images = image_map(image_dir)

    rows = []
    for image_id in labels["image_id"]:
        if image_id not in images:
            continue
        with Image.open(images[image_id]) as image:
            arr = np.asarray(image.convert("RGB"))
        mask = np.ones(arr.shape[:2], dtype=bool)
        row = {"image_id": image_id}
        row.update(extract_features(arr, mask, feature_set))
        rows.append(row)

    if not rows:
        raise ValueError("No external labeled images were found.")

    return pd.DataFrame(rows).merge(labels, on="image_id", how="inner")


def evaluate_generalization(train_data_dir, external_image_dir, external_label_csv, feature_set, output_dir):
    train_data = build_course_features(train_data_dir, feature_set)
    external_data = build_external_crop_features(external_image_dir, external_label_csv, feature_set)

    X_train = train_data.drop(columns=["image_id", "dx"])
    y_train = train_data["dx"]
    X_external = external_data.drop(columns=["image_id", "dx"])
    y_external = external_data["dx"]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_external[X_train.columns])
    metrics = compute_metrics(y_external, pred, labels=LABELS)

    print("Train samples:", len(train_data))
    print("External samples:", len(external_data))
    print("Feature set:", feature_set)
    print("External accuracy:", metrics["accuracy"])
    print("External macro-F1:", metrics["macro_f1"])
    print("External balanced accuracy:", metrics["balanced_accuracy"])
    print(metrics["classification_report"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "train_dataset": str(train_data_dir),
        "external_dataset": str(external_image_dir),
        "feature_set": feature_set,
        "train_samples": len(train_data),
        "external_samples": len(external_data),
        "external_accuracy": metrics["accuracy"],
        "external_macro_f1": metrics["macro_f1"],
        "external_balanced_accuracy": metrics["balanced_accuracy"],
    }]).to_csv(output_dir / "external_generalization_metrics.csv", index=False)

    pd.DataFrame({
        "image_id": external_data["image_id"],
        "y_true": y_external,
        "y_pred": pred,
    }).to_csv(output_dir / "external_generalization_predictions.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--external_image_dir", required=True)
    parser.add_argument("--external_label_csv", required=True)
    parser.add_argument("--feature_set", default="color_texture", choices=["color", "texture", "color_texture"])
    parser.add_argument("--output_dir", default="outputs/external")
    args = parser.parse_args()

    evaluate_generalization(
        train_data_dir=Path(args.train_data_dir),
        external_image_dir=Path(args.external_image_dir),
        external_label_csv=Path(args.external_label_csv),
        feature_set=args.feature_set,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()

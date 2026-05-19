import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import build_feature_table


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")


def predict_with_mel_threshold(model, X, mel_threshold):
    if not hasattr(model, "predict_proba"):
        raise ValueError("The saved model has mel_threshold but does not support predict_proba.")
    probabilities = model.predict_proba(X)
    class_labels = np.asarray(model.classes_)
    if "mel" not in class_labels:
        raise ValueError("The saved model has no 'mel' class.")
    mel_idx = int(np.where(class_labels == "mel")[0][0])
    non_mel_indices = np.asarray([idx for idx, label in enumerate(class_labels) if label != "mel"])
    non_mel_best = class_labels[non_mel_indices[np.argmax(probabilities[:, non_mel_indices], axis=1)]]
    return np.where(probabilities[:, mel_idx] >= mel_threshold, "mel", non_mel_best)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing image/ and mask/.")
    parser.add_argument("--output_csv", default="output.csv")
    parser.add_argument("--model_path", default="outputs/models/ml_baseline.joblib")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_set = bundle["feature_set"]
    feature_columns = bundle["feature_columns"]
    mask_mode = bundle.get("mask_mode", "raw")
    mel_threshold = bundle.get("mel_threshold")

    features = build_feature_table(Path(args.input_dir), feature_set=feature_set, mask_mode=mask_mode)
    X = features[feature_columns]
    if mel_threshold is None:
        pred = model.predict(X)
    else:
        pred = predict_with_mel_threshold(model, X, mel_threshold)

    output = pd.DataFrame({
        "image_id": features["image_id"],
        "dx": pred,
    })
    output.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import LABELS
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


def reorder_proba(probabilities, class_labels, target_labels):
    class_labels = np.asarray(class_labels)
    ordered = np.zeros((len(probabilities), len(target_labels)), dtype=np.float32)
    for target_idx, label in enumerate(target_labels):
        if label not in class_labels:
            raise ValueError(f"Class {label!r} missing from model classes {class_labels}.")
        ordered[:, target_idx] = probabilities[:, int(np.where(class_labels == label)[0][0])]
    return ordered


def xgb_cascade_probabilities(bundle, input_dir):
    labels = np.asarray(bundle.get("labels", LABELS))
    mask_mode = bundle.get("mask_mode", "raw")
    stage1 = bundle["stage1"]
    stage2 = bundle["stage2"]

    stage1_features = build_feature_table(
        Path(input_dir),
        feature_set=stage1["feature_set"],
        mask_mode=mask_mode,
    )
    stage2_features = build_feature_table(
        Path(input_dir),
        feature_set=stage2["feature_set"],
        mask_mode=mask_mode,
    )
    X_stage1 = stage1_features[stage1["feature_columns"]]
    X_stage2 = stage2_features[stage2["feature_columns"]]

    p_vasc = stage1["model"].predict_proba(X_stage1)[:, 1]
    stage2_proba = stage2["model"].predict_proba(X_stage2)
    p_mel = stage2_proba[:, 0]
    p_nv = stage2_proba[:, 1]

    label_idx = {label: idx for idx, label in enumerate(labels)}
    probabilities = np.zeros((len(stage1_features), len(labels)), dtype=np.float32)
    probabilities[:, label_idx["vasc"]] = p_vasc
    probabilities[:, label_idx["mel"]] = (1.0 - p_vasc) * p_mel
    probabilities[:, label_idx["nv"]] = (1.0 - p_vasc) * p_nv
    return stage1_features["image_id"], labels, probabilities


def predict_with_xgb_cascade(bundle, input_dir):
    image_ids, labels, probabilities = xgb_cascade_probabilities(bundle, input_dir)
    pred = labels[probabilities.argmax(axis=1)]

    return pd.DataFrame({
        "image_id": image_ids,
        "dx": pred,
    })


def predict_with_fusion_ensemble(bundle, input_dir):
    labels = np.asarray(bundle.get("labels", LABELS))
    mask_mode = bundle.get("mask_mode", "raw")
    original = bundle["original"]

    original_features = build_feature_table(
        Path(input_dir),
        feature_set=original["feature_set"],
        mask_mode=mask_mode,
    )
    X_original = original_features[original["feature_columns"]]
    original_prob = reorder_proba(
        original["model"].predict_proba(X_original),
        original["model"].classes_,
        labels,
    )

    cascade_image_ids, cascade_labels, cascade_prob = xgb_cascade_probabilities(
        bundle["cascade"],
        input_dir,
    )
    if list(cascade_image_ids) != list(original_features["image_id"]):
        raise ValueError("Original and cascade feature tables produced different image order.")
    if list(cascade_labels) != list(labels):
        raise ValueError("Original and cascade labels are inconsistent.")

    fused_prob = (
        float(bundle["original_weight"]) * original_prob
        + float(bundle["cascade_weight"]) * cascade_prob
    )
    pred = labels[fused_prob.argmax(axis=1)]
    return pd.DataFrame({
        "image_id": original_features["image_id"],
        "dx": pred,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing image/ and mask/.")
    parser.add_argument("--output_csv", default="output.csv")
    parser.add_argument("--model_path", default="outputs/models/ml_baseline.joblib")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    if bundle.get("model_type") == "xgb_cascade":
        output = predict_with_xgb_cascade(bundle, args.input_dir)
        output.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
        return
    if bundle.get("model_type") == "fusion_ensemble":
        output = predict_with_fusion_ensemble(bundle, args.input_dir)
        output.to_csv(args.output_csv, index=False)
        print(f"Saved predictions to {args.output_csv}")
        return

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

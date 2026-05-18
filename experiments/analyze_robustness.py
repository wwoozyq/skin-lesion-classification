import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir


def augmentation_type(image_id, base_id):
    image_id = str(image_id)
    base_id = str(base_id)
    if image_id == base_id:
        return "original"
    suffix = image_id[len(base_id):].lstrip("_-")
    if not suffix:
        return "augmented"
    match = re.match(r"([A-Za-z]+)", suffix)
    return match.group(1).lower() if match else "augmented"


def analyze(prediction_csv, output_dir):
    pred = pd.read_csv(prediction_csv)
    pred["image_id"] = pred["image_id"].astype(str)
    pred["base_id"] = pred["base_id"].astype(str)
    pred["aug_type"] = [
        augmentation_type(image_id, base_id)
        for image_id, base_id in zip(pred["image_id"], pred["base_id"])
    ]

    image_summary = pred.groupby("aug_type").agg(
        n_images=("image_id", "count"),
        accuracy=("correct", "mean"),
    ).reset_index()
    image_summary.insert(0, "level", "image")

    group_rows = []
    for base_id, group in pred.groupby("base_id"):
        original = group[group["aug_type"] == "original"]
        original_pred = original["pred"].iloc[0] if len(original) else None
        group_rows.append({
            "base_id": base_id,
            "dx": group["dx"].iloc[0],
            "n_images": len(group),
            "n_unique_pred": group["pred"].nunique(),
            "consistent": group["pred"].nunique() == 1,
            "all_correct": group["correct"].all(),
            "any_correct": group["correct"].any(),
            "majority_pred": group["pred"].mode().iloc[0],
            "original_pred": original_pred,
            "all_match_original": bool((group["pred"] == original_pred).all()) if original_pred else False,
        })
    group_detail = pd.DataFrame(group_rows)
    group_summary = pd.DataFrame([{
        "level": "group",
        "n_groups": len(group_detail),
        "prediction_consistency": group_detail["consistent"].mean(),
        "all_images_correct": group_detail["all_correct"].mean(),
        "any_image_correct": group_detail["any_correct"].mean(),
        "all_match_original": group_detail["all_match_original"].mean(),
    }])

    ensure_dir(output_dir)
    image_summary.to_csv(output_dir / "robustness_by_aug_type.csv", index=False)
    group_detail.to_csv(output_dir / "robustness_group_detail.csv", index=False)
    group_summary.to_csv(output_dir / "robustness_group_summary.csv", index=False)

    print("Image-level robustness:")
    print(image_summary.to_string(index=False))
    print("\nGroup-level robustness:")
    print(group_summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/metrics/robustness")
    args = parser.parse_args()
    analyze(Path(args.prediction_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()

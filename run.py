import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.features import build_feature_table


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

    features = build_feature_table(Path(args.input_dir), feature_set=feature_set)
    X = features[feature_columns]
    pred = model.predict(X)

    output = pd.DataFrame({
        "image_id": features["image_id"],
        "dx": pred,
    })
    output.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()


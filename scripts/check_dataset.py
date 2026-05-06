import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = data_dir / "image"
    mask_dir = data_dir / "mask"
    label_path = data_dir / "label.csv"

    print(f"Data dir: {data_dir}")
    print(f"Images: {len(list(image_dir.glob('*')))}")
    print(f"Masks: {len(list(mask_dir.glob('*')))}")

    if label_path.exists():
        labels = pd.read_csv(label_path)
        print(f"Labels: {len(labels)}")
        print(labels["dx"].value_counts())
    else:
        print("label.csv not found. This is OK for hidden test data.")


if __name__ == "__main__":
    main()


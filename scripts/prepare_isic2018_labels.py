import argparse
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
}


def convert_ground_truth(ground_truth_csv, output_dir):
    ground_truth_csv = Path(ground_truth_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ground_truth_csv)
    required = {"image", *LABEL_MAP.keys()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in ground truth CSV: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        matched = [label for label in LABEL_MAP if row[label] == 1]
        if len(matched) != 1:
            continue
        external_label = matched[0]
        rows.append({
            "image_id": row["image"],
            "dx": LABEL_MAP[external_label],
            "external_label": external_label,
            "source": "ISIC2018_Task3_Training",
            "image_file": f"{row['image']}.jpg",
            "has_mask": False,
        })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No MEL/NV/VASC rows were found.")

    label_csv = manifest[["image_id", "dx"]]
    label_csv.to_csv(output_dir / "label.csv", index=False)
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    summary = manifest["dx"].value_counts().rename_axis("dx").reset_index(name="count")
    summary.to_csv(output_dir / "summary.csv", index=False)

    print(f"Input rows: {len(df)}")
    print(f"Kept rows: {len(manifest)}")
    print(summary.to_string(index=False))
    print(f"Saved: {output_dir / 'label.csv'}")
    print(f"Saved: {output_dir / 'manifest.csv'}")
    print(f"Saved: {output_dir / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    convert_ground_truth(args.ground_truth_csv, args.output_dir)


if __name__ == "__main__":
    main()

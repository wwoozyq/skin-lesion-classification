import argparse
from pathlib import Path

import pandas as pd


def _load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def check_grouped_splits(splits_csv):
    splits = _load_csv(splits_csv)
    required = {"image_id", "base_id", "dx", "fold"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"{splits_csv} is missing required columns: {sorted(missing)}")

    fold_counts = splits.groupby("fold").size().rename("n_images")
    class_counts = splits.groupby(["fold", "dx"]).size().unstack(fill_value=0)
    base_fold_counts = splits.groupby("base_id")["fold"].nunique()
    leaked = base_fold_counts[base_fold_counts > 1]
    if not leaked.empty:
        examples = ", ".join(map(str, leaked.index[:10]))
        raise AssertionError(f"Group leakage detected for base_id examples: {examples}")

    duplicate_rows = splits["image_id"].duplicated().sum()
    if duplicate_rows:
        raise AssertionError(f"Duplicate image_id rows found: {duplicate_rows}")

    return fold_counts, class_counts, splits["base_id"].nunique(), len(splits)


def check_oof_predictions(oof_csv, splits_csv=None):
    oof = _load_csv(oof_csv)
    required = {"image_id", "base_id", "dx", "fold", "pred", "correct"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"{oof_csv} is missing required columns: {sorted(missing)}")

    if oof["image_id"].duplicated().any():
        dupes = oof.loc[oof["image_id"].duplicated(), "image_id"].head(10).tolist()
        raise AssertionError(f"Duplicate OOF predictions for image_id examples: {dupes}")

    base_fold_counts = oof.groupby("base_id")["fold"].nunique()
    leaked = base_fold_counts[base_fold_counts > 1]
    if not leaked.empty:
        examples = ", ".join(map(str, leaked.index[:10]))
        raise AssertionError(f"OOF group leakage detected for base_id examples: {examples}")

    if splits_csv is not None:
        splits = _load_csv(splits_csv)
        if set(oof["image_id"].astype(str)) != set(splits["image_id"].astype(str)):
            raise AssertionError("OOF predictions and split file contain different image_id sets.")

    accuracy = oof["correct"].astype(bool).mean()
    return len(oof), accuracy


def main():
    parser = argparse.ArgumentParser(description="Validate grouped-CV split and OOF artifacts.")
    parser.add_argument("--splits_csv", required=True)
    parser.add_argument("--oof_csv")
    args = parser.parse_args()

    fold_counts, class_counts, n_groups, n_images = check_grouped_splits(args.splits_csv)
    print("PASS: no base_id appears in more than one fold.")
    print(f"Images: {n_images}")
    print(f"Groups: {n_groups}")
    print("\nFold image counts:")
    print(fold_counts.to_string())
    print("\nFold class counts:")
    print(class_counts.to_string())

    if args.oof_csv:
        n_oof, accuracy = check_oof_predictions(args.oof_csv, args.splits_csv)
        print(f"\nPASS: OOF predictions cover the split image set. n={n_oof}")
        print(f"OOF accuracy from 'correct' column: {accuracy:.4f}")


if __name__ == "__main__":
    main()

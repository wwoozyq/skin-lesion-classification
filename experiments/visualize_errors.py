import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "skin_lesion_mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import load_image, load_mask
from src.preprocess import prepare_mask
from src.utils import ensure_dir


def _plot_case(ax, image, mask, title):
    ax.imshow(image)
    ax.contour(mask, levels=[0.5], colors=["yellow"], linewidths=1.2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def visualize_errors(data_dir, prediction_csv, output_dir, mask_mode="raw", max_per_pair=6):
    pred = pd.read_csv(prediction_csv)
    pred["image_id"] = pred["image_id"].astype(str)
    errors = pred[pred["dx"] != pred["pred"]].copy()
    if errors.empty:
        print("No errors found.")
        return

    ensure_dir(output_dir)
    pair_rows = []
    for (true_label, pred_label), group in errors.groupby(["dx", "pred"]):
        sample = group.head(max_per_pair)
        fig, axes = plt.subplots(len(sample), 1, figsize=(4.5, 3.4 * len(sample)))
        if len(sample) == 1:
            axes = [axes]

        for ax, (_, row) in zip(axes, sample.iterrows()):
            image = load_image(data_dir, row["image_id"])
            mask = prepare_mask(load_mask(data_dir, row["image_id"]), mask_mode=mask_mode)
            title = f"{row['image_id']} | true={true_label}, pred={pred_label}, fold={row['fold']}"
            _plot_case(ax, image, mask, title)

        filename = f"errors_true_{true_label}_pred_{pred_label}.png"
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        pair_rows.append({
            "true_label": true_label,
            "pred_label": pred_label,
            "n_errors": len(group),
            "figure": str(path),
        })

    summary = pd.DataFrame(pair_rows).sort_values("n_errors", ascending=False)
    summary.to_csv(output_dir / "error_pairs_summary.csv", index=False)
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--prediction_csv", required=True)
    parser.add_argument("--output_dir", default="outputs/figures/errors")
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--max_per_pair", type=int, default=6)
    args = parser.parse_args()
    visualize_errors(
        data_dir=Path(args.data_dir),
        prediction_csv=Path(args.prediction_csv),
        output_dir=Path(args.output_dir),
        mask_mode=args.mask_mode,
        max_per_pair=args.max_per_pair,
    )


if __name__ == "__main__":
    main()

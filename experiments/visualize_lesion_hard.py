"""One-shot diagnostic: visualize the lesions the main LR model gets wrong on
all three of (orig, aug1, aug2). Output two figures:

  outputs/figures/lesion_hard_overview.png
      One tile per lesion (original image only, mask outline overlay),
      5 cols x ceil(N/5) rows. Title shows id, true label, predicted label.
      Use this for the quick scan.

  outputs/figures/lesion_hard_detail.png
      One row per lesion, 3 cols (orig, aug1, aug2). Use this when you need
      to see whether augmentation is destroying signal.

Reads the existing seed-127 baseline OOF csv; does not retrain anything.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.dataset import load_image, load_mask

DATA_DIR = REPO / "data" / "Data_Proj2"
OOF_CSV = REPO / "outputs" / "metrics" / "ml_all_abcd_grouped_lr03_grouped_raw_seed127_oof_predictions.csv"
OUT_DIR = REPO / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLOR = {"mel": "#d62728", "nv": "#1f77b4", "vasc": "#2ca02c"}


def find_all_wrong_base_ids(oof: pd.DataFrame) -> list[int]:
    err = oof[oof.dx != oof.pred]
    err_per_base = err.groupby("base_id").size()
    total_per_base = oof.groupby("base_id").size()
    return err_per_base[err_per_base == total_per_base.loc[err_per_base.index]].index.tolist()


def overlay_mask(ax, image: np.ndarray, mask: np.ndarray, edge_color="#ffff00"):
    ax.imshow(image)
    contours = measure.find_contours(mask.astype(float), 0.5)
    for c in contours:
        ax.plot(c[:, 1], c[:, 0], color=edge_color, linewidth=1.2)
    ax.set_xticks([])
    ax.set_yticks([])


def image_ids_for_base(base_id: int) -> list[str]:
    return [str(base_id), f"{base_id}_aug1", f"{base_id}_aug2"]


def title_for_lesion(base_id: int, true_label: str, pred_counts: dict) -> str:
    pred_summary = " ".join(f"{lab}:{n}" for lab, n in sorted(pred_counts.items()))
    return f"id {base_id}  true={true_label}  pred=[{pred_summary}]"


def draw_overview(base_ids: list[int], oof: pd.DataFrame, out_path: Path):
    n = len(base_ids)
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.4))
    axes = np.atleast_2d(axes)
    for i, base_id in enumerate(base_ids):
        ax = axes[i // ncols, i % ncols]
        try:
            image = load_image(DATA_DIR, str(base_id))
            mask = load_mask(DATA_DIR, str(base_id))
            overlay_mask(ax, image, mask)
        except FileNotFoundError as exc:
            ax.text(0.5, 0.5, f"missing\n{exc}", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        rows = oof[oof.base_id == base_id]
        true_label = rows.dx.iloc[0]
        pred_counts = rows.pred.value_counts().to_dict()
        ax.set_title(
            title_for_lesion(base_id, true_label, pred_counts),
            fontsize=10,
            color=CLASS_COLOR.get(true_label, "black"),
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle(
        f"Lesions misclassified on ALL 3 versions  (LR main model, seed 127, n={n})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def draw_detail(base_ids: list[int], oof: pd.DataFrame, out_path: Path):
    n = len(base_ids)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.0))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i, base_id in enumerate(base_ids):
        for j, image_id in enumerate(image_ids_for_base(base_id)):
            ax = axes[i, j]
            row = oof[oof.image_id == image_id]
            if row.empty:
                ax.text(0.5, 0.5, "(no oof row)", ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            try:
                image = load_image(DATA_DIR, image_id)
                mask = load_mask(DATA_DIR, image_id)
                overlay_mask(ax, image, mask)
            except FileNotFoundError as exc:
                ax.text(0.5, 0.5, str(exc), ha="center", va="center")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            true_label = row.dx.iloc[0]
            pred_label = row.pred.iloc[0]
            ok = row.correct.iloc[0]
            ax.set_title(
                f"{image_id}  true={true_label}  pred={pred_label}  {'OK' if ok else 'WRONG'}",
                fontsize=10,
                color=CLASS_COLOR.get(true_label, "black"),
            )
    fig.suptitle(
        f"Per-lesion detail: 3 versions of each all-wrong lesion (n={n})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    oof = pd.read_csv(OOF_CSV)
    oof["image_id"] = oof["image_id"].astype(str)
    base_ids = find_all_wrong_base_ids(oof)
    print(f"all-wrong lesions: {len(base_ids)}")
    print("ids:", base_ids)

    class_breakdown = (
        oof[oof.base_id.isin(base_ids)]
        .drop_duplicates("base_id")
        .dx.value_counts()
        .to_dict()
    )
    print("by true class:", class_breakdown)

    pred_breakdown = (
        oof[oof.base_id.isin(base_ids)]
        .groupby("base_id")
        .pred.agg(lambda s: s.mode().iloc[0])
        .value_counts()
        .to_dict()
    )
    print("majority predicted class:", pred_breakdown)

    overview_path = OUT_DIR / "lesion_hard_overview.png"
    detail_path = OUT_DIR / "lesion_hard_detail.png"
    draw_overview(base_ids, oof, overview_path)
    draw_detail(base_ids, oof, detail_path)
    print(f"saved: {overview_path}")
    print(f"saved: {detail_path}")


if __name__ == "__main__":
    main()

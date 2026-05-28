"""Hair-removal preprocessing evaluation.

Hypothesis from docs/LESION_HARD_ERROR_ANALYSIS.md: 2 of 25 all-augs-wrong
lesions (id 24, id 154) fail because of dark hair occluding the lesion;
DullRazor-style morphological hair removal could rescue them. This script
quantifies whether that translates into measurable lift on the LR main
line.

Pipeline:
  1. Build a parallel data dir at outputs/cache/hair_removed/ containing
     hair-removed copies of the 600 images, with mask/ and label.csv
     symlinked from the original data dir.
  2. Extract all_abcd_grouped features (mask_mode=raw) from the cleaned
     images using the existing src.features.build_feature_table.
  3. Run 5-seed bagged LR(C=0.3) + SelectKBest(k=140) + StandardScaler
     + StratifiedGroupKFold(5) by base_id, mirroring src.train_ml.
  4. Compare bagged BalAcc / macro-F1 / accuracy to the LR main baseline
     reported in project memory (0.7512 / 0.7435 / 0.7283).

Reads from data/Data_Proj2; writes only to outputs/cache and a single
metrics CSV. Does not modify any src/ file.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage import color, filters, morphology
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.dataset import (  # noqa: E402
    list_image_ids,
    load_image,
    load_labels,
    load_mask,
)
from src.features import build_feature_table  # noqa: E402
from src.preprocess import prepare_mask  # noqa: E402
from src.utils import base_id_from_image_id  # noqa: E402

DEFAULT_SEEDS = (42, 127, 2024, 3407, 520)
BASELINE_BAGGED = {  # from project memory, LR main 5-seed bagged
    "balanced_accuracy": 0.7512,
    "macro_f1": 0.7435,
    "accuracy": 0.7283,
}


def hair_remove(
    image: np.ndarray,
    line_len: int = 15,
    bh_thresh: float = 0.15,
    dilate_r: int = 2,
    med_r: int = 5,
) -> np.ndarray:
    """DullRazor-style hair removal using skimage."""
    gray = color.rgb2gray(image)
    bh_h = morphology.black_tophat(gray, morphology.rectangle(1, line_len))
    bh_v = morphology.black_tophat(gray, morphology.rectangle(line_len, 1))
    bh = np.maximum(bh_h, bh_v)
    hair_mask = bh > bh_thresh
    hair_mask = morphology.binary_dilation(hair_mask, morphology.disk(dilate_r))
    med = np.stack(
        [filters.median(image[:, :, c], morphology.disk(med_r)) for c in range(3)],
        axis=-1,
    )
    out = np.where(hair_mask[:, :, None], med, image)
    return out.astype(np.uint8)


def prepare_parallel_data_dir(src_dir: Path, dst_dir: Path, force: bool = False):
    image_ids = list_image_ids(src_dir)
    dst_image = dst_dir / "image"
    dst_mask = dst_dir / "mask"
    dst_image.mkdir(parents=True, exist_ok=True)
    dst_mask.mkdir(parents=True, exist_ok=True)

    src_label = src_dir / "label.csv"
    dst_label = dst_dir / "label.csv"
    if dst_label.exists() or dst_label.is_symlink():
        dst_label.unlink()
    os.symlink(src_label.resolve(), dst_label)

    for entry in (src_dir / "mask").iterdir():
        if not entry.is_file():
            continue
        link_target = dst_mask / entry.name
        if link_target.exists() or link_target.is_symlink():
            link_target.unlink()
        os.symlink(entry.resolve(), link_target)

    todo = []
    for image_id in image_ids:
        out_path = dst_image / f"{image_id}.jpg"
        if force or not out_path.exists():
            todo.append((image_id, out_path))

    if not todo:
        print(f"hair-removed images already cached at {dst_image}")
        return

    print(f"removing hair from {len(todo)} / {len(image_ids)} images")
    t0 = time.time()
    for image_id, out_path in tqdm(todo, desc="hair removal"):
        img = load_image(src_dir, image_id)
        cleaned = hair_remove(img)
        Image.fromarray(cleaned).save(out_path, quality=92)
    print(f"hair removal done in {time.time() - t0:.1f}s")


def build_pipeline(k: int = 140, C: float = 0.3, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("clf", LogisticRegression(
            C=C,
            max_iter=4000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def oof_bagged_eval(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    seeds: tuple[int, ...],
    n_outer: int = 5,
    k: int = 140,
    C: float = 0.3,
) -> dict:
    all_seed_proba = []
    per_seed_metrics = []
    for seed in seeds:
        proba = np.zeros((len(y), len(classes)))
        skf = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for tr, va in skf.split(X, y, groups=groups):
            pipe = build_pipeline(k=k, C=C, random_state=42)
            pipe.fit(X[tr], y[tr])
            fold_proba = pipe.predict_proba(X[va])
            # align classes order
            for cls_idx, cls in enumerate(pipe.classes_):
                proba[va, classes.index(cls)] = fold_proba[:, cls_idx]
        pred = np.array([classes[i] for i in proba.argmax(axis=1)])
        per_seed_metrics.append({
            "seed": seed,
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
        })
        all_seed_proba.append(proba)

    bagged_proba = np.mean(all_seed_proba, axis=0)
    bagged_pred = np.array([classes[i] for i in bagged_proba.argmax(axis=1)])
    bagged = {
        "balanced_accuracy": balanced_accuracy_score(y, bagged_pred),
        "macro_f1": f1_score(y, bagged_pred, average="macro"),
        "accuracy": accuracy_score(y, bagged_pred),
    }
    per_seed_df = pd.DataFrame(per_seed_metrics)
    return {"bagged": bagged, "per_seed": per_seed_df, "bagged_proba": bagged_proba}


def feature_matrix(features: pd.DataFrame, labels: pd.DataFrame):
    merged = features.merge(labels, on="image_id", how="inner")
    feature_cols = [c for c in features.columns if c != "image_id"]
    X = merged[feature_cols].to_numpy(dtype=float)
    y = merged["dx"].to_numpy()
    groups = np.array([base_id_from_image_id(s) for s in merged["image_id"]])
    return X, y, groups, merged["image_id"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/Data_Proj2")
    ap.add_argument(
        "--hair_removed_dir",
        default="outputs/cache/hair_removed",
        help="parallel data dir for cleaned images",
    )
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--k_features", type=int, default=140)
    ap.add_argument("--C", type=float, default=0.3)
    ap.add_argument("--mask_mode", default="raw")
    ap.add_argument("--feature_set", default="all_abcd_grouped")
    ap.add_argument("--force_clean", action="store_true",
                    help="rebuild hair-removed images even if cached")
    ap.add_argument("--skip_clean", action="store_true",
                    help="reuse cleaned dir, evaluate only")
    ap.add_argument("--summary_csv",
                    default="outputs/metrics/hair_removal_eval_summary.csv")
    args = ap.parse_args()

    src_dir = REPO / args.data_dir
    dst_dir = REPO / args.hair_removed_dir
    seeds = tuple(int(s) for s in args.seeds.split(","))

    print(f"src data:  {src_dir}")
    print(f"hr cache:  {dst_dir}")
    print(f"seeds:     {seeds}")
    print(f"pipeline:  LR(C={args.C}, balanced) + SelectKBest(k={args.k_features})"
          f" + StandardScaler, StratifiedGroupKFold(5)")
    print(f"features:  {args.feature_set}, mask_mode={args.mask_mode}")
    print()

    if not args.skip_clean:
        prepare_parallel_data_dir(src_dir, dst_dir, force=args.force_clean)

    image_ids = list_image_ids(src_dir)
    labels = load_labels(src_dir)
    classes = sorted(labels["dx"].unique().tolist())

    print()
    print("Extracting features from ORIGINAL images...")
    feats_orig = build_feature_table(
        src_dir, image_ids=image_ids,
        feature_set=args.feature_set, mask_mode=args.mask_mode,
    )

    print("Extracting features from HAIR-REMOVED images...")
    feats_hr = build_feature_table(
        dst_dir, image_ids=image_ids,
        feature_set=args.feature_set, mask_mode=args.mask_mode,
    )

    X_o, y_o, g_o, ids_o = feature_matrix(feats_orig, labels)
    X_h, y_h, g_h, ids_h = feature_matrix(feats_hr, labels)
    assert ids_o == ids_h and (y_o == y_h).all() and (g_o == g_h).all()

    print()
    print("=== Eval on ORIGINAL images (sanity vs main-line memory number) ===")
    res_o = oof_bagged_eval(X_o, y_o, g_o, classes, seeds,
                            k=args.k_features, C=args.C)
    print("per-seed:")
    print(res_o["per_seed"].to_string(index=False))
    print("bagged:", res_o["bagged"])

    print()
    print("=== Eval on HAIR-REMOVED images ===")
    res_h = oof_bagged_eval(X_h, y_h, g_h, classes, seeds,
                            k=args.k_features, C=args.C)
    print("per-seed:")
    print(res_h["per_seed"].to_string(index=False))
    print("bagged:", res_h["bagged"])

    print()
    print("=== Delta (hair-removed minus original) ===")
    delta = {k: res_h["bagged"][k] - res_o["bagged"][k] for k in res_o["bagged"]}
    print(delta)
    print()
    print("=== Baseline memory number ===")
    print(BASELINE_BAGGED)

    summary = pd.DataFrame([
        {"variant": "original",        **res_o["bagged"]},
        {"variant": "hair_removed",    **res_h["bagged"]},
        {"variant": "delta_hr_minus_orig", **delta},
    ])
    summary_path = REPO / args.summary_csv
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"\nsaved: {summary_path}")


if __name__ == "__main__":
    main()

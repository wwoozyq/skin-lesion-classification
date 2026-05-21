import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_traditional_score_search import (  # noqa: E402
    _evaluate_oof,
    _per_class_metrics,
    _pred_from_weighted_proba,
    _splits,
)
from run_traditional_shortlist_stability import ENSEMBLES, _candidates, _load_xy  # noqa: E402
from src.config import LABELS  # noqa: E402
from src.dataset import load_labels  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.utils import base_id_from_image_id, ensure_dir  # noqa: E402


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _group_mean_proba(proba, image_ids):
    df = pd.DataFrame(proba, columns=LABELS)
    df["base_id"] = image_ids.map(base_id_from_image_id).values
    group_mean = df.groupby("base_id")[LABELS].transform("mean")
    return group_mean.to_numpy(dtype=float)


def _group_level_y_pred(y, pred, image_ids):
    data = pd.DataFrame({
        "base_id": image_ids.map(base_id_from_image_id).values,
        "y": y.values,
        "pred": pred.values,
    })
    group_rows = []
    for base_id, group in data.groupby("base_id", sort=False):
        true_labels = group["y"].unique()
        if len(true_labels) != 1:
            raise RuntimeError(f"Inconsistent labels within group {base_id}: {true_labels}")
        group_rows.append({
            "base_id": base_id,
            "y": true_labels[0],
            "pred": group["pred"].iloc[0],
            "consistent": group["pred"].nunique() == 1,
        })
    grouped = pd.DataFrame(group_rows)
    return grouped["y"], grouped["pred"], grouped["consistent"].mean()


def _metrics_row(run_id, seed, candidate, y, pred, image_ids, postprocess, smoothing, weights=None):
    metrics = compute_metrics(y, pred, labels=LABELS)
    group_y, group_pred, consistency = _group_level_y_pred(y, pred, image_ids)
    group_metrics = compute_metrics(group_y, group_pred, labels=LABELS)
    return {
        "run_id": run_id,
        "seed": int(seed),
        "feature_set": candidate.get("feature_set", "ensemble"),
        "mask_mode": candidate.get("mask_mode", "mixed"),
        "classifier": candidate.get("classifier_label", "soft_vote"),
        "k_features": candidate.get("k_features", "mixed"),
        "postprocess": postprocess,
        "smoothing": smoothing,
        "fixed_weights": "" if weights is None else ",".join(f"{label}:{weight:g}" for label, weight in zip(LABELS, weights)),
        "image_accuracy": metrics["accuracy"],
        "image_macro_f1": metrics["macro_f1"],
        "image_balanced_accuracy": metrics["balanced_accuracy"],
        "group_accuracy": group_metrics["accuracy"],
        "group_macro_f1": group_metrics["macro_f1"],
        "group_balanced_accuracy": group_metrics["balanced_accuracy"],
        "within_group_consistency": consistency,
        **_per_class_metrics(y, pred),
    }


def _proba_to_pred(proba, weights=None):
    if weights is None:
        weights = [1.0, 1.0, 1.0]
    return _pred_from_weighted_proba(proba, weights)


def run_consistency(data_dir, output_dir, seeds, n_splits=5, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for group consistency analysis.")
    labels["image_id"] = labels["image_id"].astype(str)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    rows = []
    cache_dir = Path("outputs/cache")
    candidates = _candidates()

    for seed in seeds:
        proba_store = {}
        y_ref = None
        image_ref = None

        for candidate in candidates:
            X, y, image_ids = _load_xy(
                data_dir,
                labels,
                candidate["feature_set"],
                candidate["mask_mode"],
                cache_dir,
                use_cache,
            )
            if y_ref is None:
                y_ref = y
                image_ref = image_ids
            elif not image_ids.equals(image_ref):
                raise RuntimeError(f"Image order mismatch for {candidate['run_id']}")

            splits = _splits(image_ids, y, n_splits=n_splits, random_state=int(seed))
            _, pred, proba = _evaluate_oof(X, y, splits, candidate["classifier"], candidate["k_features"])
            if proba is None:
                continue

            proba_store[candidate["run_id"]] = proba
            rows.append(_metrics_row(candidate["run_id"], seed, candidate, y, pred, image_ids, "argmax", "none"))

            smoothed = _group_mean_proba(proba, image_ids)
            smoothed_pred = _proba_to_pred(smoothed)
            rows.append(
                _metrics_row(
                    candidate["run_id"] + "__group_mean",
                    seed,
                    candidate,
                    y,
                    smoothed_pred,
                    image_ids,
                    "argmax",
                    "group_mean_proba",
                )
            )

            if candidate["fixed_weights"] is not None:
                weighted_pred = _proba_to_pred(proba, candidate["fixed_weights"])
                rows.append(
                    _metrics_row(
                        candidate["run_id"] + "__fixed_weights",
                        seed,
                        candidate,
                        y,
                        weighted_pred,
                        image_ids,
                        "fixed_weight_argmax",
                        "none",
                        candidate["fixed_weights"],
                    )
                )
                weighted_smoothed_pred = _proba_to_pred(smoothed, candidate["fixed_weights"])
                rows.append(
                    _metrics_row(
                        candidate["run_id"] + "__fixed_weights_group_mean",
                        seed,
                        candidate,
                        y,
                        weighted_smoothed_pred,
                        image_ids,
                        "fixed_weight_argmax",
                        "group_mean_proba",
                        candidate["fixed_weights"],
                    )
                )

            print(
                f"seed={int(seed):>8d} {candidate['run_id']:32s} "
                f"image_acc={compute_metrics(y, pred, labels=LABELS)['accuracy']:.4f} "
                f"group_smooth_acc={compute_metrics(y, smoothed_pred, labels=LABELS)['accuracy']:.4f}",
                flush=True,
            )

        for ensemble_name, members in ENSEMBLES.items():
            if not all(member in proba_store for member in members):
                continue
            proba = np.mean([proba_store[member] for member in members], axis=0)
            pred = _proba_to_pred(proba)
            ensemble_meta = {
                "feature_set": "ensemble",
                "mask_mode": "mixed",
                "classifier_label": "soft_vote",
                "k_features": "mixed",
            }
            rows.append(_metrics_row(ensemble_name, seed, ensemble_meta, y_ref, pred, image_ref, "soft_vote", "none"))

            smoothed = _group_mean_proba(proba, image_ref)
            smoothed_pred = _proba_to_pred(smoothed)
            rows.append(
                _metrics_row(
                    ensemble_name + "__group_mean",
                    seed,
                    ensemble_meta,
                    y_ref,
                    smoothed_pred,
                    image_ref,
                    "soft_vote",
                    "group_mean_proba",
                )
            )
            print(
                f"seed={int(seed):>8d} {ensemble_name:32s} "
                f"image_acc={compute_metrics(y_ref, pred, labels=LABELS)['accuracy']:.4f} "
                f"group_smooth_acc={compute_metrics(y_ref, smoothed_pred, labels=LABELS)['accuracy']:.4f}",
                flush=True,
            )

        pd.DataFrame(rows).to_csv(output_dir / "group_consistency_partial.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "group_consistency_results.csv", index=False)
    summary = (
        result.groupby(["run_id", "feature_set", "mask_mode", "classifier", "k_features", "postprocess", "smoothing", "fixed_weights"])
        .agg(
            image_accuracy_mean=("image_accuracy", "mean"),
            image_accuracy_std=("image_accuracy", "std"),
            image_macro_f1_mean=("image_macro_f1", "mean"),
            image_macro_f1_std=("image_macro_f1", "std"),
            group_accuracy_mean=("group_accuracy", "mean"),
            group_accuracy_std=("group_accuracy", "std"),
            group_macro_f1_mean=("group_macro_f1", "mean"),
            group_macro_f1_std=("group_macro_f1", "std"),
            within_group_consistency_mean=("within_group_consistency", "mean"),
        )
        .reset_index()
        .sort_values(["image_accuracy_mean", "image_macro_f1_mean"], ascending=False)
    )
    summary.to_csv(output_dir / "group_consistency_summary.csv", index=False)
    _write_markdown(summary, output_dir / "group_consistency_summary.md")
    return result, summary


def _write_markdown(summary, output_md):
    top = summary.head(20)
    lines = [
        "# Group Consistency Analysis",
        "",
        "This analysis averages out-of-fold probabilities across images that share the same original lesion id.",
        "It is a robustness/presentation extension, not a replacement for the single-image grouped CV score.",
        "",
        "| rank | run | smoothing | image acc mean | image macro-F1 mean | group acc mean | group macro-F1 mean | consistency mean |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        lines.append(
            f"| {rank} | `{row['run_id']}` | `{row['smoothing']}` | "
            f"{row['image_accuracy_mean']:.4f} | {row['image_macro_f1_mean']:.4f} | "
            f"{row['group_accuracy_mean']:.4f} | {row['group_macro_f1_mean']:.4f} | "
            f"{row['within_group_consistency_mean']:.4f} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/metrics/traditional_group_consistency")
    parser.add_argument("--seeds", default="127")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    _, summary = run_consistency(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seeds=_parse_list(args.seeds),
        n_splits=args.n_splits,
        use_cache=not args.no_cache,
    )
    print("\nGroup consistency summary:")
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

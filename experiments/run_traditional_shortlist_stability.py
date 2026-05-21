import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_traditional_score_search import (  # noqa: E402
    _evaluate_oof,
    _feature_table,
    _per_class_metrics,
    _pred_from_weighted_proba,
    _splits,
)
from src.config import LABELS  # noqa: E402
from src.dataset import load_labels  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.utils import ensure_dir  # noqa: E402


def _candidates():
    return [
        {
            "run_id": "abc_clean_lr_c03_k180",
            "feature_set": "all_abcd_grouped",
            "mask_mode": "clean",
            "classifier": LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", random_state=42),
            "classifier_label": "lr_c03",
            "k_features": "180",
            "fixed_weights": [1.1, 1.0, 0.9],
        },
        {
            "run_id": "abc_raw_lr_c03_k140",
            "feature_set": "all_abcd_grouped",
            "mask_mode": "raw",
            "classifier": LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced", random_state=42),
            "classifier_label": "lr_c03",
            "k_features": "140",
            "fixed_weights": [0.9, 0.75, 1.25],
        },
        {
            "run_id": "final_abcd_raw_gb_all",
            "feature_set": "final_abcd_grouped",
            "mask_mode": "raw",
            "classifier": GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=2,
                random_state=42,
            ),
            "classifier_label": "gb",
            "k_features": "all",
            "fixed_weights": [0.75, 1.25, 1.25],
        },
        {
            "run_id": "final_abcd_raw_hgb_all",
            "feature_set": "final_abcd_grouped",
            "mask_mode": "raw",
            "classifier": HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                class_weight="balanced",
                random_state=42,
            ),
            "classifier_label": "hgb",
            "k_features": "all",
            "fixed_weights": [0.75, 1.25, 0.75],
        },
        {
            "run_id": "final_melnv_raw_rf_k140",
            "feature_set": "final_melnv",
            "mask_mode": "raw",
            "classifier": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "classifier_label": "rf",
            "k_features": "140",
            "fixed_weights": None,
        },
        {
            "run_id": "final_melnv_raw_rf_k100",
            "feature_set": "final_melnv",
            "mask_mode": "raw",
            "classifier": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "classifier_label": "rf",
            "k_features": "100",
            "fixed_weights": [0.9, 0.75, 1.25],
        },
        {
            "run_id": "final_melnv_clean_rf_k100",
            "feature_set": "final_melnv",
            "mask_mode": "clean",
            "classifier": RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            "classifier_label": "rf",
            "k_features": "100",
            "fixed_weights": [1.0, 0.75, 1.25],
        },
    ]


ENSEMBLES = {
    "seed127_top5_soft_vote": [
        "abc_clean_lr_c03_k180",
        "abc_raw_lr_c03_k140",
        "final_abcd_raw_gb_all",
        "final_abcd_raw_hgb_all",
        "final_melnv_raw_rf_k140",
    ],
    "seed127_top5_plus_melnv100_soft_vote": [
        "abc_clean_lr_c03_k180",
        "abc_raw_lr_c03_k140",
        "final_abcd_raw_gb_all",
        "final_abcd_raw_hgb_all",
        "final_melnv_raw_rf_k100",
    ],
}


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_xy(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    feature_df = _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache)
    data = labels[["image_id", "dx"]].merge(feature_df, on="image_id", how="inner")
    X = data.drop(columns=["image_id", "dx"]).reset_index(drop=True)
    y = data["dx"].reset_index(drop=True)
    image_ids = data["image_id"].reset_index(drop=True)
    return X, y, image_ids


def _metric_row(run_id, seed, candidate, metrics, pred, postprocess, weights=None):
    return {
        "run_id": run_id,
        "seed": int(seed),
        "feature_set": candidate.get("feature_set", "ensemble"),
        "mask_mode": candidate.get("mask_mode", "mixed"),
        "classifier": candidate.get("classifier_label", "soft_vote"),
        "k_features": candidate.get("k_features", "mixed"),
        "postprocess": postprocess,
        "fixed_weights": "" if weights is None else ",".join(f"{label}:{weight:g}" for label, weight in zip(LABELS, weights)),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        **_per_class_metrics(candidate["y"], pred),
    }


def run_shortlist(data_dir, output_dir, seeds, n_splits=5, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for shortlist stability.")
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
            metrics, pred, proba = _evaluate_oof(X, y, splits, candidate["classifier"], candidate["k_features"])
            candidate_for_row = {**candidate, "y": y}
            rows.append(_metric_row(candidate["run_id"], seed, candidate_for_row, metrics, pred, "argmax"))
            if proba is not None:
                proba_store[candidate["run_id"]] = proba
                if candidate["fixed_weights"] is not None:
                    weighted_pred = _pred_from_weighted_proba(proba, candidate["fixed_weights"])
                    weighted_metrics = compute_metrics(y, weighted_pred, labels=LABELS)
                    rows.append(
                        _metric_row(
                            candidate["run_id"] + "__fixed_weights",
                            seed,
                            candidate_for_row,
                            weighted_metrics,
                            weighted_pred,
                            "fixed_weight_argmax",
                            candidate["fixed_weights"],
                        )
                    )

            print(
                f"seed={int(seed):>8d} {candidate['run_id']:32s} "
                f"acc={metrics['accuracy']:.4f} macro={metrics['macro_f1']:.4f}",
                flush=True,
            )

        for ensemble_name, members in ENSEMBLES.items():
            if not all(member in proba_store for member in members):
                continue
            proba = np.mean([proba_store[member] for member in members], axis=0)
            pred = _pred_from_weighted_proba(proba, [1.0, 1.0, 1.0])
            metrics = compute_metrics(y_ref, pred, labels=LABELS)
            rows.append(
                _metric_row(
                    ensemble_name,
                    seed,
                    {"feature_set": "ensemble", "mask_mode": "mixed", "classifier_label": "soft_vote", "k_features": "mixed", "y": y_ref},
                    metrics,
                    pred,
                    "soft_vote",
                )
            )
            print(
                f"seed={int(seed):>8d} {ensemble_name:32s} "
                f"acc={metrics['accuracy']:.4f} macro={metrics['macro_f1']:.4f}",
                flush=True,
            )

        pd.DataFrame(rows).to_csv(output_dir / "traditional_shortlist_stability_partial.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "traditional_shortlist_stability.csv", index=False)
    summary = (
        result.groupby(["run_id", "feature_set", "mask_mode", "classifier", "k_features", "postprocess", "fixed_weights"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .reset_index()
        .sort_values(["accuracy_mean", "macro_f1_mean"], ascending=False)
    )
    summary.to_csv(output_dir / "traditional_shortlist_stability_summary.csv", index=False)
    return result, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/metrics/traditional_score_search_phase2")
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    _, summary = run_shortlist(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        seeds=_parse_list(args.seeds),
        n_splits=args.n_splits,
        use_cache=not args.no_cache,
    )
    print("\nShortlist stability summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

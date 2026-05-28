"""Overnight medical-preprocessing × cascade exploration runner.

Implements the cells defined in ``docs/OVERNIGHT_EXPLORATION_PLAN.md``:

- cell 0: baseline (no preprocessing, default Stage 2 features)
- cells 1/2/3: single preprocessing (shades_of_gray / clahe_lab_l / hb_melanin)
- cell 4: B1 — add ``abcd_grouped`` to Stage 2 feature set
- cells 5/6: best A × B1 (chosen after cells 0–4 finish)
- cell 7: shades_of_gray+clahe_lab_l × B1 (tie-breaker, only if both pass)

Each cell runs the same XGBoost cascade as ledger §9
(``d2-more-trees``, ``k_features=all``) under 5-seed × 5-fold
StratifiedGroupKFold grouped by ``base_id``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is required. Install with: uv pip install --python .venv/bin/python xgboost"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.preprocess_medical import (
    PREPROCESSING_CLAHE,
    PREPROCESSING_HBMEL,
    PREPROCESSING_NONE,
    PREPROCESSING_SHADES,
    PREPROCESSING_SHADES_CLAHE,
)
from src.utils import base_id_from_image_id, ensure_dir


STAGE1_FEATURE_SET = "xgb_cascade_stage1"
STAGE1_K = "100"
STAGE1_XGB = {
    "n_estimators": 150,
    "max_depth": 2,
    "learning_rate": 0.05,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_lambda": 5.0,
}

STAGE2_XGB = {
    "n_estimators": 300,
    "max_depth": 2,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.60,
    "reg_lambda": 5.0,
}
STAGE2_K = "all"

# Ledger §9's 0.7887 was a different cascade variant; this runner pins
# the comparison anchor to cell 0's measured value, not the literature
# number. The literature number is kept here for reporting only.
LEDGER_BASELINE_BAL_ACC = 0.7887
PASS_MARGIN = 0.005


def _cache_path(cache_dir, feature_set, mask_mode, preprocessing):
    safe_name = feature_set.replace("+", "_AND_")
    safe_pp = preprocessing.replace("+", "_AND_")
    return Path(cache_dir) / f"features_cascade_{safe_name}_{mask_mode}_pp_{safe_pp}.csv"


def _load_feature_table(data_dir, labels, feature_set, mask_mode, preprocessing, cache_dir, use_cache):
    cache_path = _cache_path(cache_dir, feature_set, mask_mode, preprocessing)
    if use_cache and cache_path.exists():
        table = pd.read_csv(cache_path)
    else:
        table = build_feature_table(
            data_dir,
            image_ids=labels["image_id"].astype(str),
            feature_set=feature_set,
            mask_mode=mask_mode,
            preprocessing=preprocessing,
        )
        if use_cache:
            ensure_dir(cache_path.parent)
            table.to_csv(cache_path, index=False)
    table["image_id"] = table["image_id"].astype(str)
    merged = labels.merge(table, on="image_id", how="inner")
    if len(merged) != len(labels):
        raise ValueError(
            f"Feature table for {feature_set} (pp={preprocessing}) is missing samples."
        )
    return merged.drop(columns=["image_id", "dx"])


def _build_pipeline(xgb_kwargs, k_features, n_features, random_state, n_jobs):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    clf = XGBClassifier(
        **xgb_kwargs,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


def _make_splits(labels, n_splits, seed):
    groups = labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(placeholder, labels["dx"], groups=groups))


def _run_cell(
    *,
    cell_id,
    cell_label,
    preprocessing,
    stage2_feature_set,
    data_dir,
    seeds,
    n_splits,
    mask_mode,
    cache_dir,
    use_cache,
    n_jobs,
):
    print(f"\n=== cell {cell_id} {cell_label}: pp={preprocessing}, stage2={stage2_feature_set} ===")
    cell_started = time.time()

    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required.")
    labels = labels.copy().reset_index(drop=True)
    labels["image_id"] = labels["image_id"].astype(str)
    y_true = labels["dx"].to_numpy()
    label_array = np.asarray(LABELS)
    label_idx = {label: idx for idx, label in enumerate(LABELS)}

    X_stage1 = _load_feature_table(
        data_dir, labels, STAGE1_FEATURE_SET, mask_mode, preprocessing, cache_dir, use_cache
    )
    X_stage2 = _load_feature_table(
        data_dir, labels, stage2_feature_set, mask_mode, preprocessing, cache_dir, use_cache
    )

    y_stage1 = (y_true == "vasc").astype(int)
    mel_nv_indices = np.where(y_true != "vasc")[0]

    per_seed_probs = []
    per_seed_rows = []
    for seed in seeds:
        splits = _make_splits(labels, n_splits=n_splits, seed=seed)
        stage1_pipe = _build_pipeline(
            STAGE1_XGB,
            k_features=STAGE1_K,
            n_features=X_stage1.shape[1],
            random_state=seed,
            n_jobs=n_jobs,
        )
        stage1_oof = np.zeros((len(labels), 2), dtype=np.float32)
        for train_idx, valid_idx in splits:
            sw = compute_sample_weight("balanced", y_stage1[train_idx])
            stage1_pipe.fit(X_stage1.iloc[train_idx], y_stage1[train_idx], clf__sample_weight=sw)
            stage1_oof[valid_idx] = stage1_pipe.predict_proba(X_stage1.iloc[valid_idx])
        p_vasc = stage1_oof[:, 1]

        stage2_oof = np.zeros((len(labels), 2), dtype=np.float32)
        for train_idx, valid_idx in splits:
            train_mel_nv = train_idx[np.isin(train_idx, mel_nv_indices)]
            y_stage2 = (y_true[train_mel_nv] == "nv").astype(int)
            pipe = _build_pipeline(
                STAGE2_XGB,
                k_features=STAGE2_K,
                n_features=X_stage2.shape[1],
                random_state=seed,
                n_jobs=n_jobs,
            )
            sw = compute_sample_weight("balanced", y_stage2)
            pipe.fit(X_stage2.iloc[train_mel_nv], y_stage2, clf__sample_weight=sw)
            stage2_oof[valid_idx] = pipe.predict_proba(X_stage2.iloc[valid_idx])
        p_mel = stage2_oof[:, 0]
        p_nv = stage2_oof[:, 1]

        probs = np.zeros((len(labels), len(LABELS)), dtype=np.float32)
        probs[:, label_idx["vasc"]] = p_vasc
        probs[:, label_idx["mel"]] = (1.0 - p_vasc) * p_mel
        probs[:, label_idx["nv"]] = (1.0 - p_vasc) * p_nv

        pred = label_array[probs.argmax(axis=1)]
        seed_metrics = compute_metrics(y_true, pred, labels=LABELS)
        per_seed_probs.append(probs)
        per_seed_rows.append({
            "cell_id": cell_id,
            "cell_label": cell_label,
            "preprocessing": preprocessing,
            "stage2_feature_set": stage2_feature_set,
            "seed": seed,
            "balanced_accuracy": seed_metrics["balanced_accuracy"],
            "macro_f1": seed_metrics["macro_f1"],
            "accuracy": seed_metrics["accuracy"],
        })
        print(
            f"  seed {seed}: bal_acc={seed_metrics['balanced_accuracy']:.4f} "
            f"macro_f1={seed_metrics['macro_f1']:.4f} acc={seed_metrics['accuracy']:.4f}"
        )

    bagged_probs = np.mean(per_seed_probs, axis=0)
    bagged_pred = label_array[bagged_probs.argmax(axis=1)]
    bagged_metrics = compute_metrics(y_true, bagged_pred, labels=LABELS)
    seed_bal = [row["balanced_accuracy"] for row in per_seed_rows]
    seed_f1 = [row["macro_f1"] for row in per_seed_rows]
    seed_acc = [row["accuracy"] for row in per_seed_rows]

    summary_row = {
        "cell_id": cell_id,
        "cell_label": cell_label,
        "preprocessing": preprocessing,
        "stage2_feature_set": stage2_feature_set,
        "bagged_balanced_accuracy": bagged_metrics["balanced_accuracy"],
        "bagged_macro_f1": bagged_metrics["macro_f1"],
        "bagged_accuracy": bagged_metrics["accuracy"],
        "per_seed_bal_acc_mean": float(np.mean(seed_bal)),
        "per_seed_bal_acc_std": float(np.std(seed_bal)),
        "per_seed_macro_f1_mean": float(np.mean(seed_f1)),
        "per_seed_macro_f1_std": float(np.std(seed_f1)),
        "per_seed_accuracy_mean": float(np.mean(seed_acc)),
        "per_seed_accuracy_std": float(np.std(seed_acc)),
        "wall_time_min": (time.time() - cell_started) / 60.0,
    }
    print(
        f"  BAGGED: bal_acc={bagged_metrics['balanced_accuracy']:.4f} "
        f"macro_f1={bagged_metrics['macro_f1']:.4f} acc={bagged_metrics['accuracy']:.4f} "
        f"(wall {summary_row['wall_time_min']:.1f} min)"
    )
    return summary_row, per_seed_rows


def _verdict(summary_row, baseline_row):
    bagged_bal = summary_row["bagged_balanced_accuracy"]
    bagged_f1 = summary_row["bagged_macro_f1"]
    base_bal = baseline_row["bagged_balanced_accuracy"]
    base_f1 = baseline_row["bagged_macro_f1"]
    if bagged_bal >= base_bal + PASS_MARGIN and bagged_f1 >= base_f1 - 0.005:
        return "pass"
    if bagged_bal < base_bal - PASS_MARGIN:
        return "regress"
    return "no_change"


def _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv):
    ensure_dir(Path(summary_csv).parent)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    pd.DataFrame(per_seed_rows).to_csv(per_seed_csv, index=False)


def run_overnight(
    data_dir,
    seeds,
    n_splits,
    mask_mode,
    cache_dir,
    use_cache,
    n_jobs,
    summary_csv,
    per_seed_csv,
    status_json,
    include_stack_cells,
):
    started = time.time()
    summary_rows = []
    per_seed_rows = []

    cells_core = [
        ("0", "baseline", PREPROCESSING_NONE, "xgb_cascade_stage2"),
        ("1", "A1_shades_of_gray", PREPROCESSING_SHADES, "xgb_cascade_stage2"),
        ("2", "A2_clahe_lab_l", PREPROCESSING_CLAHE, "xgb_cascade_stage2"),
        ("3", "A3_hb_melanin", PREPROCESSING_HBMEL, "xgb_cascade_stage2"),
        ("4", "B1_stage2_plus_abcd", PREPROCESSING_NONE, "xgb_cascade_stage2_abcd"),
    ]

    for cell_id, cell_label, pp, fs in cells_core:
        summary_row, seed_rows = _run_cell(
            cell_id=cell_id,
            cell_label=cell_label,
            preprocessing=pp,
            stage2_feature_set=fs,
            data_dir=data_dir,
            seeds=seeds,
            n_splits=n_splits,
            mask_mode=mask_mode,
            cache_dir=cache_dir,
            use_cache=use_cache,
            n_jobs=n_jobs,
        )
        summary_rows.append(summary_row)
        per_seed_rows.extend(seed_rows)
        _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv)

    baseline_row = summary_rows[0]
    # No sanity gate — stack cells run regardless. Baseline value is logged
    # so the morning review can see how cell 0 compares to ledger §9.

    # Decide stack cells from A cells that PASSED
    a_cells = [(row, _verdict(row, baseline_row)) for row in summary_rows[1:4]]
    a_passing = [row for row, verdict in a_cells if verdict == "pass"]
    a_passing.sort(key=lambda r: r["bagged_balanced_accuracy"], reverse=True)
    print(f"\nA-cell verdicts: {[(r['cell_label'], v) for r, v in a_cells]}")
    print(f"A-cells passing: {[r['cell_label'] for r in a_passing]}")

    next_cell_id = 5
    stacks_run = []
    if include_stack_cells and a_passing:
        # cell 5: best A + B1
        best_a = a_passing[0]
        stack_pp = best_a["preprocessing"]
        stack_label = f"{best_a['cell_label']}_plus_B1"
        summary_row, seed_rows = _run_cell(
            cell_id=str(next_cell_id),
            cell_label=stack_label,
            preprocessing=stack_pp,
            stage2_feature_set="xgb_cascade_stage2_abcd",
            data_dir=data_dir,
            seeds=seeds,
            n_splits=n_splits,
            mask_mode=mask_mode,
            cache_dir=cache_dir,
            use_cache=use_cache,
            n_jobs=n_jobs,
        )
        summary_rows.append(summary_row)
        per_seed_rows.extend(seed_rows)
        stacks_run.append(stack_label)
        next_cell_id += 1
        _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv)

        # cell 6: second A + B1 (only if there are 2+ passing A cells)
        if len(a_passing) >= 2:
            second_a = a_passing[1]
            stack_pp = second_a["preprocessing"]
            stack_label = f"{second_a['cell_label']}_plus_B1"
            summary_row, seed_rows = _run_cell(
                cell_id=str(next_cell_id),
                cell_label=stack_label,
                preprocessing=stack_pp,
                stage2_feature_set="xgb_cascade_stage2_abcd",
                data_dir=data_dir,
                seeds=seeds,
                n_splits=n_splits,
                mask_mode=mask_mode,
                cache_dir=cache_dir,
                use_cache=use_cache,
                n_jobs=n_jobs,
            )
            summary_rows.append(summary_row)
            per_seed_rows.extend(seed_rows)
            stacks_run.append(stack_label)
            next_cell_id += 1
            _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv)

        # cell 7: A1+A2 combo + B1 (only if BOTH A1 and A2 passed)
        a_pass_labels = {r["cell_label"] for r in a_passing}
        if "A1_shades_of_gray" in a_pass_labels and "A2_clahe_lab_l" in a_pass_labels:
            stack_label = "A1_plus_A2_plus_B1"
            summary_row, seed_rows = _run_cell(
                cell_id=str(next_cell_id),
                cell_label=stack_label,
                preprocessing=PREPROCESSING_SHADES_CLAHE,
                stage2_feature_set="xgb_cascade_stage2_abcd",
                data_dir=data_dir,
                seeds=seeds,
                n_splits=n_splits,
                mask_mode=mask_mode,
                cache_dir=cache_dir,
                use_cache=use_cache,
                n_jobs=n_jobs,
            )
            summary_rows.append(summary_row)
            per_seed_rows.extend(seed_rows)
            stacks_run.append(stack_label)
            next_cell_id += 1
            _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv)

    # Final verdict per cell
    for row in summary_rows:
        row["verdict"] = _verdict(row, baseline_row) if row["cell_id"] != "0" else "baseline"
    _flush(summary_rows, per_seed_rows, summary_csv, per_seed_csv)

    best = max(summary_rows, key=lambda r: r["bagged_balanced_accuracy"])
    status = {
        "status": "success",
        "wall_time_min": (time.time() - started) / 60.0,
        "ledger_baseline_bagged_balanced_accuracy": LEDGER_BASELINE_BAL_ACC,
        "cell0_bagged_balanced_accuracy": baseline_row["bagged_balanced_accuracy"],
        "n_cells_run": len(summary_rows),
        "stacks_run": stacks_run,
        "best_cell": best["cell_label"],
        "best_cell_bagged_balanced_accuracy": best["bagged_balanced_accuracy"],
        "best_cell_bagged_macro_f1": best["bagged_macro_f1"],
        "best_cell_bagged_accuracy": best["bagged_accuracy"],
        "beats_cell0": best["bagged_balanced_accuracy"] > baseline_row["bagged_balanced_accuracy"] + PASS_MARGIN,
        "beats_ledger_baseline": best["bagged_balanced_accuracy"] > LEDGER_BASELINE_BAL_ACC + PASS_MARGIN,
        "summary_csv": str(summary_csv),
        "per_seed_csv": str(per_seed_csv),
    }
    ensure_dir(Path(status_json).parent)
    Path(status_json).write_text(json.dumps(status, indent=2))
    print(f"\nOVERALL: {status}")
    return status, summary_rows, per_seed_rows


def _parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Overnight medical-preprocessing × cascade exploration.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--cache_dir", default="outputs/cache")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--summary_csv",
        default="outputs/metrics/overnight_exploration_summary.csv",
    )
    parser.add_argument(
        "--per_seed_csv",
        default="outputs/metrics/overnight_exploration_per_seed.csv",
    )
    parser.add_argument(
        "--status_json",
        default="outputs/metrics/overnight_exploration_status.json",
    )
    parser.add_argument("--skip_stack_cells", action="store_true")
    args = parser.parse_args()

    run_overnight(
        data_dir=Path(args.data_dir),
        seeds=[int(s) for s in _parse_csv(args.seeds)],
        n_splits=args.n_splits,
        mask_mode=args.mask_mode,
        cache_dir=Path(args.cache_dir),
        use_cache=not args.no_cache,
        n_jobs=args.n_jobs,
        summary_csv=Path(args.summary_csv),
        per_seed_csv=Path(args.per_seed_csv),
        status_json=Path(args.status_json),
        include_stack_cells=not args.skip_stack_cells,
    )


if __name__ == "__main__":
    main()

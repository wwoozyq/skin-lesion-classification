import argparse
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import DEFAULT_METRICS_DIR, DEFAULT_MODEL_DIR, LABELS
from .dataset import load_image, load_labels, load_mask
from .evaluate import compute_metrics
from .features import build_feature_table
from .utils import ensure_dir


RANDOM_STATE = 42
DEFAULT_FIGURES_DIR = Path("outputs/figures")
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def original_image_id(image_id):
    return str(image_id).split("_aug", 1)[0]


def _predict_with_mel_threshold(probabilities, class_labels, mel_threshold):
    class_labels = np.asarray(class_labels)
    mel_idx = int(np.where(class_labels == "mel")[0][0])
    non_mel_indices = np.asarray([idx for idx, label in enumerate(class_labels) if label != "mel"])

    pred = class_labels[np.argmax(probabilities, axis=1)].copy()
    non_mel_best = class_labels[non_mel_indices[np.argmax(probabilities[:, non_mel_indices], axis=1)]]
    pred = np.where(probabilities[:, mel_idx] >= mel_threshold, "mel", non_mel_best)
    return pred


def _mel_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, labels=["mel"], average="macro", zero_division=0)


def _find_best_mel_threshold(y_true, probabilities, class_labels):
    candidates = []
    for threshold in np.arange(0.20, 0.501, 0.01):
        pred = _predict_with_mel_threshold(probabilities, class_labels, threshold)
        metrics = compute_metrics(y_true, pred, labels=LABELS)
        mel_recall = _mel_recall(y_true, pred)
        score = metrics["macro_f1"] + metrics["balanced_accuracy"] + 0.25 * mel_recall
        candidates.append({
            "mel_threshold": float(round(threshold, 2)),
            "mel_recall": float(mel_recall),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "score": float(score),
            "pred": pred,
            "metrics": metrics,
        })

    candidates.sort(
        key=lambda row: (
            row["score"],
            row["mel_recall"],
            row["macro_f1"],
            row["balanced_accuracy"],
        ),
        reverse=True,
    )
    return candidates[0], candidates[:5]


def _candidate_models():
    return {
        "rf_balanced_400": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "rf_balanced_leaf2": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "extra_trees_balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
        "svm_rbf_balanced_c3": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=3.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
        "svm_rbf_balanced_c10": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=10.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def _model_score(metrics, mel_recall):
    return metrics["macro_f1"] + metrics["balanced_accuracy"] + 0.25 * mel_recall


def _metrics_row(experiment, model_name, decision, threshold, y_true, y_pred, metrics):
    return {
        "experiment": experiment,
        "model": model_name,
        "decision": decision,
        "mel_threshold": threshold if threshold is not None else "",
        "mel_recall": _mel_recall(y_true, y_pred),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }


def _abcd_category(feature_name):
    name = feature_name.lower()
    if any(key in name for key in ["asymmetry", "center_offset"]):
        return "A"
    if any(key in name for key in [
        "perimeter",
        "circularity",
        "compactness",
        "solidity",
        "convex",
        "radial",
        "radius",
        "roughness",
        "boundary",
        "circle_area_ratio",
    ]):
        return "B"
    if any(key in name for key in [
        "rgb",
        "hsv",
        "lab",
        "color",
        "saturation",
        "dark",
        "bright",
        "hue",
        "brown",
        "black",
        "white",
        "blue",
        "red",
    ]):
        return "C"
    if any(key in name for key in [
        "area",
        "diameter",
        "radius",
        "bbox_width",
        "bbox_height",
        "bbox_diagonal",
        "enclosing_circle",
        "major_axis",
        "minor_axis",
    ]):
        return "D"
    return "Other"


def _save_feature_importance(model, feature_columns, metrics_dir, figures_dir):
    clf = model.named_steps.get("clf")
    if not hasattr(clf, "feature_importances_"):
        print("Feature importance skipped: selected classifier has no feature_importances_.")
        return None

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_df["abcd_category"] = importance_df["feature"].map(_abcd_category)
    top20 = importance_df.head(20).copy()
    top20.to_csv(metrics_dir / "feature_importance_top20.csv", index=False)

    plot_df = top20.iloc[::-1]
    colors = {
        "A": "#4C78A8",
        "B": "#F58518",
        "C": "#54A24B",
        "D": "#E45756",
        "Other": "#9D9D9D",
    }
    plt.figure(figsize=(10, 7))
    plt.barh(
        plot_df["feature"],
        plot_df["importance"],
        color=[colors.get(cat, "#9D9D9D") for cat in plot_df["abcd_category"]],
    )
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.title("Top 20 ABCD Feature Importances")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in colors.items()
        if label in set(top20["abcd_category"])
    ]
    plt.legend(handles=handles, title="ABCD")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance_top20.png", dpi=200)
    plt.close()
    return top20


def _save_mel_nv_confusion(y_true, y_pred, metrics_dir):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    mel_total = int((y_true == "mel").sum())
    nv_total = int((y_true == "nv").sum())
    mel_as_nv = int(((y_true == "mel") & (y_pred == "nv")).sum())
    nv_as_mel = int(((y_true == "nv") & (y_pred == "mel")).sum())
    rows = [
        {
            "error_type": "mel_predicted_as_nv",
            "count": mel_as_nv,
            "denominator": mel_total,
            "ratio": mel_as_nv / mel_total if mel_total else 0.0,
        },
        {
            "error_type": "nv_predicted_as_mel",
            "count": nv_as_mel,
            "denominator": nv_total,
            "ratio": nv_as_mel / nv_total if nv_total else 0.0,
        },
    ]
    confusion_df = pd.DataFrame(rows)
    confusion_df.to_csv(metrics_dir / "mel_nv_confusion.csv", index=False)
    return confusion_df


def _save_misclassified_cases(data_dir, data, y_true, y_pred, figures_dir, max_cases=12):
    case_dir = ensure_dir(figures_dir / "misclassified_cases")
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    cases = data[["image_id", "dx"]].copy().reset_index(drop=True)
    cases["pred"] = y_pred
    cases = cases[cases["dx"] != cases["pred"]].copy()
    cases.to_csv(case_dir / "misclassified_cases.csv", index=False)

    def save_grid(subset, filename, title):
        subset = subset.head(max_cases)
        if subset.empty:
            return
        cols = 4
        rows = int(np.ceil(len(subset) / cols))
        plt.figure(figsize=(cols * 3.0, rows * 3.2))
        for idx, row in enumerate(subset.itertuples(index=False), start=1):
            image = load_image(data_dir, row.image_id)
            mask = load_mask(data_dir, row.image_id)
            ax = plt.subplot(rows, cols, idx)
            ax.imshow(image)
            ax.contour(mask, levels=[0.5], colors="yellow", linewidths=1)
            ax.set_title(f"{row.image_id}\nT:{row.dx} P:{row.pred}", fontsize=9)
            ax.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(case_dir / filename, dpi=180)
        plt.close()

    save_grid(cases[(cases["dx"] == "mel") & (cases["pred"] == "nv")], "mel_predicted_as_nv.png", "mel predicted as nv")
    save_grid(cases[(cases["dx"] == "nv") & (cases["pred"] == "mel")], "nv_predicted_as_mel.png", "nv predicted as mel")
    save_grid(cases.head(max_cases), "misclassified_examples.png", "misclassified examples")


def _make_grouped_cv(n_splits=5):
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        ), "StratifiedGroupKFold"
    except TypeError:
        return GroupKFold(n_splits=n_splits), "GroupKFold"


def _validate_grouped_splits(cv, X, y, groups):
    rows = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        overlap = train_groups.intersection(val_groups)
        if overlap:
            raise RuntimeError(f"Data leakage detected in fold {fold}: {sorted(overlap)[:5]}")
        rows.append({
            "fold": fold,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
            "train_groups": int(len(train_groups)),
            "val_groups": int(len(val_groups)),
            "val_mel": int((y.iloc[val_idx] == "mel").sum()),
            "val_nv": int((y.iloc[val_idx] == "nv").sum()),
            "val_vasc": int((y.iloc[val_idx] == "vasc").sum()),
        })
    return pd.DataFrame(rows)


def _evaluate_model(name, model, X, y, groups, cv):
    probabilities = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict_proba")
    class_labels = np.asarray(LABELS)
    pred = class_labels[np.argmax(probabilities, axis=1)]
    metrics = compute_metrics(y, pred, labels=LABELS)
    mel_recall = _mel_recall(y, pred)
    best_threshold, top_thresholds = _find_best_mel_threshold(y, probabilities, class_labels)
    threshold_metrics = best_threshold["metrics"]
    threshold_score = _model_score(threshold_metrics, best_threshold["mel_recall"])

    return {
        "name": name,
        "model": model,
        "probabilities": probabilities,
        "argmax_pred": pred,
        "argmax_metrics": metrics,
        "argmax_mel_recall": mel_recall,
        "threshold": best_threshold["mel_threshold"],
        "threshold_pred": best_threshold["pred"],
        "threshold_metrics": threshold_metrics,
        "threshold_mel_recall": best_threshold["mel_recall"],
        "threshold_score": threshold_score,
        "top_thresholds": top_thresholds,
    }


def _evaluate_feature_selection(X, y, groups, cv, feature_counts):
    rows = []
    n_features = X.shape[1]
    for k in feature_counts:
        k = min(k, n_features)
        model = Pipeline([
            ("select", SelectKBest(score_func=f_classif, k=k)),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                C=3.0,
                gamma="scale",
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ])
        result = _evaluate_model(f"svm_rbf_c3_select_top{k}", model, X, y, groups, cv)
        rows.append(_metrics_row(
            "feature_selection",
            result["name"],
            "argmax",
            None,
            y,
            result["argmax_pred"],
            result["argmax_metrics"],
        ))
        rows.append(_metrics_row(
            "feature_selection",
            result["name"],
            "mel_threshold",
            result["threshold"],
            y,
            result["threshold_pred"],
            result["threshold_metrics"],
        ))
    return pd.DataFrame(rows)


def _hierarchical_predictions(X, y, groups, cv):
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    stage1_pred = np.empty(len(y), dtype=object)
    stage2_mel_prob = np.zeros(len(y), dtype=np.float64)
    argmax_pred = np.empty(len(y), dtype=object)

    stage1_template = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])
    stage2_template = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=3.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        )),
    ])

    for train_idx, val_idx in cv.split(X, y, groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_stage1 = np.where(y_train == "vasc", "vasc", "non_vasc")

        stage1 = clone(stage1_template)
        stage1.fit(X_train, y_stage1)
        val_stage1 = stage1.predict(X_val)
        stage1_pred[val_idx] = val_stage1

        non_vasc_train = y_train != "vasc"
        stage2 = clone(stage2_template)
        stage2.fit(X_train.loc[non_vasc_train], y_train.loc[non_vasc_train])
        stage2_probs = stage2.predict_proba(X_val)
        stage2_labels = np.asarray(stage2.classes_)
        mel_idx = int(np.where(stage2_labels == "mel")[0][0])
        stage2_mel_prob[val_idx] = stage2_probs[:, mel_idx]
        stage2_pred = stage2_labels[np.argmax(stage2_probs, axis=1)]
        argmax_pred[val_idx] = np.where(val_stage1 == "vasc", "vasc", stage2_pred)

    return stage1_pred, stage2_mel_prob, argmax_pred


def _evaluate_hierarchical(X, y, groups, cv):
    stage1_pred, mel_prob, argmax_pred = _hierarchical_predictions(X, y, groups, cv)
    argmax_metrics = compute_metrics(y, argmax_pred, labels=LABELS)
    rows = [_metrics_row("hierarchical", "vasc_then_mel_nv_svm_c3", "argmax", None, y, argmax_pred, argmax_metrics)]

    candidates = []
    for threshold in np.arange(0.20, 0.501, 0.01):
        pred = np.where(stage1_pred == "vasc", "vasc", np.where(mel_prob >= threshold, "mel", "nv"))
        metrics = compute_metrics(y, pred, labels=LABELS)
        mel_recall = _mel_recall(y, pred)
        candidates.append({
            "threshold": float(round(threshold, 2)),
            "pred": pred,
            "metrics": metrics,
            "score": _model_score(metrics, mel_recall),
        })
    candidates.sort(
        key=lambda row: (
            row["score"],
            row["metrics"]["macro_f1"],
            row["metrics"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best = candidates[0]
    rows.append(_metrics_row(
        "hierarchical",
        "vasc_then_mel_nv_svm_c3",
        "mel_threshold",
        best["threshold"],
        y,
        best["pred"],
        best["metrics"],
    ))
    return pd.DataFrame(rows), best


def train_ml(data_dir, feature_set="all"):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for training.")

    feature_df = build_feature_table(data_dir, labels["image_id"].astype(str), feature_set)
    data = feature_df.merge(labels, on="image_id", how="inner")
    X = data.drop(columns=["image_id", "dx"])
    y = data["dx"]
    groups = data["image_id"].map(original_image_id)

    cv, cv_name = _make_grouped_cv(n_splits=5)
    split_summary = _validate_grouped_splits(cv, X, y, groups)
    print(f"Cross-validation: {cv_name}")
    print(f"Samples: {len(data)}, original-image groups: {groups.nunique()}")
    print("Fold leakage check: no original-image group appears in both train and validation.")
    print(split_summary.to_string(index=False))

    results = []
    for name, model in _candidate_models().items():
        print(f"Evaluating {name}...")
        results.append(_evaluate_model(name, model, X, y, groups, cv))

    print("Evaluating feature selection top-k experiments...")
    feature_selection_df = _evaluate_feature_selection(X, y, groups, cv, [20, 30, 50, 80])
    print(feature_selection_df.to_string(index=False))

    print("Evaluating hierarchical classifier...")
    hierarchical_df, hierarchical_best = _evaluate_hierarchical(X, y, groups, cv)
    print(hierarchical_df.to_string(index=False))

    results.sort(
        key=lambda row: (
            row["threshold_score"],
            row["threshold_metrics"]["macro_f1"],
            row["threshold_metrics"]["balanced_accuracy"],
            row["threshold_mel_recall"],
        ),
        reverse=True,
    )
    best_result = results[0]
    model = best_result["model"]
    pred = best_result["argmax_pred"]
    metrics = best_result["argmax_metrics"]
    threshold_metrics = best_result["threshold_metrics"]
    top_thresholds = best_result["top_thresholds"]

    print("Default argmax metrics")
    print("Model:", best_result["name"])
    print("Accuracy:", metrics["accuracy"])
    print("Macro-F1:", metrics["macro_f1"])
    print("Balanced accuracy:", metrics["balanced_accuracy"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print(metrics["classification_report"])
    print("Best mel-threshold metrics")
    print("Model:", best_result["name"])
    print("Mel threshold:", best_result["threshold"])
    print("Accuracy:", threshold_metrics["accuracy"])
    print("Macro-F1:", threshold_metrics["macro_f1"])
    print("Balanced accuracy:", threshold_metrics["balanced_accuracy"])
    print("Confusion matrix:")
    print(threshold_metrics["confusion_matrix"])
    print(threshold_metrics["classification_report"])
    print("Top mel thresholds:")
    for row in top_thresholds:
        print(
            f"threshold={row['mel_threshold']:.2f}, "
            f"mel_recall={row['mel_recall']:.3f}, "
            f"macro_f1={row['macro_f1']:.3f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"accuracy={row['accuracy']:.3f}"
        )

    model.fit(X, y)
    ensure_dir(DEFAULT_MODEL_DIR)
    ensure_dir(DEFAULT_METRICS_DIR)
    ensure_dir(DEFAULT_FIGURES_DIR)
    model_path = DEFAULT_MODEL_DIR / "ml_baseline.joblib"
    joblib.dump({
        "model": model,
        "feature_set": feature_set,
        "feature_columns": list(X.columns),
        "mel_threshold": best_result["threshold"],
        "model_name": best_result["name"],
        "cv": cv_name,
        "grouped_by": "original_image_id",
        "random_state": RANDOM_STATE,
    }, model_path)

    metric_rows = []
    for result in results:
        metric_rows.append({
            "model": result["name"],
            "feature_set": feature_set,
            "cv": cv_name,
            "decision": "argmax",
            "mel_threshold": "",
            "mel_recall": result["argmax_mel_recall"],
            "accuracy": result["argmax_metrics"]["accuracy"],
            "macro_f1": result["argmax_metrics"]["macro_f1"],
            "balanced_accuracy": result["argmax_metrics"]["balanced_accuracy"],
        })
        metric_rows.append({
            "model": result["name"],
            "feature_set": feature_set,
            "cv": cv_name,
            "decision": "mel_threshold",
            "mel_threshold": result["threshold"],
            "mel_recall": result["threshold_mel_recall"],
            "accuracy": result["threshold_metrics"]["accuracy"],
            "macro_f1": result["threshold_metrics"]["macro_f1"],
            "balanced_accuracy": result["threshold_metrics"]["balanced_accuracy"],
        })
    pd.DataFrame(metric_rows).to_csv(DEFAULT_METRICS_DIR / "ml_baseline_metrics.csv", index=False)
    split_summary.to_csv(DEFAULT_METRICS_DIR / "grouped_cv_split_summary.csv", index=False)
    feature_selection_df.to_csv(DEFAULT_METRICS_DIR / "feature_selection_results.csv", index=False)
    hierarchical_df.to_csv(DEFAULT_METRICS_DIR / "hierarchical_results.csv", index=False)
    pd.DataFrame(
        hierarchical_best["metrics"]["confusion_matrix"],
        index=LABELS,
        columns=LABELS,
    ).to_csv(DEFAULT_METRICS_DIR / "hierarchical_best_confusion_matrix.csv")
    pd.DataFrame(
        threshold_metrics["confusion_matrix"],
        index=LABELS,
        columns=LABELS,
    ).to_csv(DEFAULT_METRICS_DIR / "ml_best_confusion_matrix.csv")
    (DEFAULT_METRICS_DIR / "ml_best_classification_report.txt").write_text(
        threshold_metrics["classification_report"],
        encoding="utf-8",
    )
    mel_nv_confusion = _save_mel_nv_confusion(y, best_result["threshold_pred"], DEFAULT_METRICS_DIR)
    print("Mel vs nv confusion:")
    print(mel_nv_confusion.to_string(index=False))
    importance_model = _candidate_models()["rf_balanced_leaf2"]
    importance_model.fit(X, y)
    top20 = _save_feature_importance(importance_model, list(X.columns), DEFAULT_METRICS_DIR, DEFAULT_FIGURES_DIR)
    if top20 is not None:
        print("Feature importance model: rf_balanced_leaf2")
        print("Top 20 feature importance:")
        print(top20.to_string(index=False))
    _save_misclassified_cases(data_dir, data, y, best_result["threshold_pred"], DEFAULT_FIGURES_DIR)

    print(f"Saved model to {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_set", default="all", choices=["all", "color", "shape", "texture"])
    args = parser.parse_args()
    train_ml(Path(args.data_dir), args.feature_set)


if __name__ == "__main__":
    main()


import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_labels
from src.evaluate import compute_metrics
from src.features import build_feature_table
from src.utils import base_id_from_image_id, ensure_dir


warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.")


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache=True):
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"features_{feature_set}_{mask_mode}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    feature_df = build_feature_table(
        data_dir,
        labels["image_id"].astype(str),
        feature_set=feature_set,
        mask_mode=mask_mode,
    )
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def _splits(data, n_splits, random_state):
    y = data["dx"]
    groups = data["image_id"].map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(data.drop(columns=["dx"]), y, groups=groups))


def _classifiers(names):
    grid = []
    if "lr" in names:
        for c_value in [0.3, 1.0]:
            grid.append((
                f"lr_C{c_value}",
                LogisticRegression(C=c_value, max_iter=2000, class_weight="balanced", random_state=42),
            ))
    if "rf" in names:
        grid.append((
            "rf_depth15",
            RandomForestClassifier(
                n_estimators=400,
                max_depth=15,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ))
    if "svm" in names:
        for c_value in [3.0, 10.0]:
            grid.append((
                f"svm_C{c_value}",
                SVC(
                    C=c_value,
                    gamma="scale",
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
            ))
    return grid


def _pipeline(classifier, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(score_func=f_classif, k=k_value)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def _metric_row(experiment, feature_set, model_name, k_features, n_features, threshold, y_true, y_pred):
    metrics = compute_metrics(y_true, y_pred, labels=LABELS)
    row = {
        "experiment": experiment,
        "feature_set": feature_set,
        "model": model_name,
        "k_features": k_features,
        "n_features": n_features,
        "mel_threshold": threshold if threshold is not None else "",
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
    }
    for label in LABELS:
        row[f"{label}_recall"] = recall_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)
    return row


def _oof_predictions(model, X, y, splits):
    pred = pd.Series(index=X.index, dtype=object)
    probabilities = np.zeros((len(X), len(LABELS)), dtype=np.float64)
    for train_idx, valid_idx in splits:
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred.iloc[valid_idx] = fitted.predict(X.iloc[valid_idx])
        fold_proba = fitted.predict_proba(X.iloc[valid_idx])
        for class_idx, class_label in enumerate(fitted.classes_):
            if class_label in LABELS:
                probabilities[valid_idx, LABELS.index(class_label)] = fold_proba[:, class_idx]
    return pred.to_numpy(), probabilities


def _predict_with_mel_threshold(probabilities, threshold):
    mel_idx = LABELS.index("mel")
    non_mel_indices = [idx for idx, label in enumerate(LABELS) if label != "mel"]
    non_mel_best = np.asarray(LABELS)[non_mel_indices][
        np.argmax(probabilities[:, non_mel_indices], axis=1)
    ]
    return np.where(probabilities[:, mel_idx] >= threshold, "mel", non_mel_best)


def _best_mel_threshold(y, probabilities):
    candidates = []
    for threshold in np.arange(0.20, 0.561, 0.01):
        pred = _predict_with_mel_threshold(probabilities, float(round(threshold, 2)))
        metrics = compute_metrics(y, pred, labels=LABELS)
        candidates.append((metrics["macro_f1"], metrics["balanced_accuracy"], float(round(threshold, 2)), pred))
    candidates.sort(reverse=True, key=lambda row: (row[0], row[1]))
    return candidates[0]


def _hierarchical_predictions(model, X, y, splits):
    y = pd.Series(y).reset_index(drop=True)
    stage1_pred = np.empty(len(y), dtype=object)
    stage2_argmax = np.empty(len(y), dtype=object)
    stage2_mel_prob = np.zeros(len(y), dtype=np.float64)

    for train_idx, valid_idx in splits:
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train = y.iloc[train_idx]

        stage1 = clone(model)
        stage1.fit(X_train, np.where(y_train == "vasc", "vasc", "non_vasc"))
        valid_stage1 = stage1.predict(X_valid)
        stage1_pred[valid_idx] = valid_stage1

        non_vasc_train = y_train != "vasc"
        stage2 = clone(model)
        stage2.fit(X_train.loc[non_vasc_train], y_train.loc[non_vasc_train])
        valid_proba = stage2.predict_proba(X_valid)
        valid_classes = np.asarray(stage2.classes_)
        mel_idx = int(np.where(valid_classes == "mel")[0][0])
        stage2_mel_prob[valid_idx] = valid_proba[:, mel_idx]
        stage2_argmax[valid_idx] = stage2.predict(X_valid)

    argmax_pred = np.where(stage1_pred == "vasc", "vasc", stage2_argmax)
    return stage1_pred, stage2_mel_prob, argmax_pred


def _best_hierarchical_threshold(y, stage1_pred, mel_prob):
    candidates = []
    for threshold in np.arange(0.20, 0.561, 0.01):
        pred = np.where(stage1_pred == "vasc", "vasc", np.where(mel_prob >= threshold, "mel", "nv"))
        metrics = compute_metrics(y, pred, labels=LABELS)
        candidates.append((metrics["macro_f1"], metrics["balanced_accuracy"], float(round(threshold, 2)), pred))
    candidates.sort(reverse=True, key=lambda row: (row[0], row[1]))
    return candidates[0]


def run_experiment(
    data_dir,
    feature_sets,
    classifiers,
    k_features,
    mask_mode,
    output_csv,
    n_splits=5,
    random_state=127,
    use_cache=True,
):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required for integration experiment.")
    labels["image_id"] = labels["image_id"].astype(str)

    rows = []
    cache_dir = Path("outputs/cache")
    for feature_set in feature_sets:
        feature_df = _feature_table(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache)
        data = feature_df.merge(labels, on="image_id", how="inner")
        X = data.drop(columns=["image_id", "dx"])
        y = data["dx"].reset_index(drop=True)
        X = X.reset_index(drop=True)
        splits = _splits(data[["image_id", "dx"]].reset_index(drop=True), n_splits, random_state)

        for model_name, classifier in _classifiers(classifiers):
            for k_value in k_features:
                model = _pipeline(classifier, k_value, X.shape[1])
                pred, probabilities = _oof_predictions(model, X, y, splits)
                rows.append(_metric_row("argmax", feature_set, model_name, k_value, X.shape[1], None, y, pred))

                _, _, threshold, threshold_pred = _best_mel_threshold(y, probabilities)
                rows.append(_metric_row("mel_threshold", feature_set, model_name, k_value, X.shape[1], threshold, y, threshold_pred))

                stage1_pred, mel_prob, hierarchical_pred = _hierarchical_predictions(model, X, y, splits)
                rows.append(_metric_row("hierarchical_argmax", feature_set, model_name, k_value, X.shape[1], None, y, hierarchical_pred))
                _, _, hierarchical_threshold, hierarchical_threshold_pred = _best_hierarchical_threshold(y, stage1_pred, mel_prob)
                rows.append(_metric_row(
                    "hierarchical_mel_threshold",
                    feature_set,
                    model_name,
                    k_value,
                    X.shape[1],
                    hierarchical_threshold,
                    y,
                    hierarchical_threshold_pred,
                ))

                latest = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).head(1).iloc[0]
                print(
                    f"{feature_set:22s} {model_name:10s} k={k_value:>4s} "
                    f"best_so_far={latest['experiment']} macro_f1={latest['macro_f1']:.4f} "
                    f"accuracy={latest['accuracy']:.4f}"
                )

    result = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_sets", default="all_boundary,all_abcd_grouped,final_abcd_grouped")
    parser.add_argument("--classifiers", default="lr,rf,svm")
    parser.add_argument("--k_features", default="80,100,140")
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/abcd_grouped_integration.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    result = run_experiment(
        data_dir=Path(args.data_dir),
        feature_sets=_parse_list(args.feature_sets),
        classifiers=_parse_list(args.classifiers),
        k_features=_parse_list(args.k_features),
        mask_mode=args.mask_mode,
        output_csv=Path(args.output_csv),
        n_splits=args.n_splits,
        random_state=args.random_state,
        use_cache=not args.no_cache,
    )
    print("\nTop results:")
    print(result.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

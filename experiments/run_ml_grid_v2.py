"""experiments/run_ml_grid_v2.py

v2: 任意 5 模块自由组合的特征集网格搜索 + 排序柱状图。

5 个可组合积木：
    all (= color + shape + texture), contrast, abcd_v2, boundary, melnv

用 '+' 拼接组合，例如:
    all                      # 仅基线
    boundary+melnv           # 旧 features.py 不支持的组合
    all+contrast+boundary    # 三块组合
    final / final_melnv      # 兼容旧别名

命令行示例:
    python experiments/run_ml_grid_v2.py --data_dir data\Data_Proj2
    python experiments/run_ml_grid_v2.py --data_dir data\Data_Proj2 \
        --feature_sets all_31 --classifiers lr,svm --k_features 100

输出:
    outputs/metrics/ml_grid_v2.csv          全部结果, 按 balanced_accuracy 降序
    outputs/figures/ml_grid_v2_top.png      Top-N 柱状图
    outputs/cache/features_v2_*.csv         每个 feature_set 的特征缓存
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS
from src.dataset import load_image, load_labels, load_mask
from src.evaluate import compute_metrics
from src.features_abcd import extract_abcd_v2_features
from src.features_boundary import extract_boundary_features
from src.features_color import extract_color_features
from src.features_contrast import extract_contrast_features
from src.features_melnv import extract_melnv_features
from src.features_shape import extract_shape_features
from src.features_texture import extract_texture_features
from src.preprocess import prepare_mask
from src.utils import base_id_from_image_id, ensure_dir


# ----------------------------------------------------------------------------
# v2 核心: 自由解析 feature_set 字符串
# ----------------------------------------------------------------------------
ATOMIC_EXTRACTORS = {
    "color":    extract_color_features,
    "shape":    extract_shape_features,
    "texture":  extract_texture_features,
    "contrast": extract_contrast_features,
    "abcd_v2":  extract_abcd_v2_features,
    "boundary": extract_boundary_features,
    "melnv":    extract_melnv_features,
}
ALL_BUNDLE = ("color", "shape", "texture")
COMBO_BLOCKS = ("all", "contrast", "abcd_v2", "boundary", "melnv")


def resolve_modules(feature_set):
    """把 'all+boundary+melnv' 解析为底层原子模块列表 (去重保序)。"""
    if feature_set == "final":
        return list(ALL_BUNDLE) + ["contrast", "abcd_v2", "boundary"]
    if feature_set == "final_melnv":
        return list(ALL_BUNDLE) + ["contrast", "abcd_v2", "boundary", "melnv"]

    parts = [p.strip() for p in feature_set.split("+") if p.strip()]
    if not parts:
        raise ValueError(f"Empty feature_set: '{feature_set}'")

    modules = []
    for part in parts:
        if part == "all":
            modules.extend(ALL_BUNDLE)
        elif part in ATOMIC_EXTRACTORS:
            modules.append(part)
        else:
            raise ValueError(
                f"Unknown feature block '{part}'. "
                f"Allowed: {list(COMBO_BLOCKS)} or any of {list(ATOMIC_EXTRACTORS)}."
            )

    seen, deduped = set(), []
    for m in modules:
        if m not in seen:
            seen.add(m)
            deduped.append(m)
    return deduped


def enumerate_all_combos():
    """从 5 个积木里生成 2^5 - 1 = 31 种非空组合 (用 '+' 拼接)。"""
    combos = []
    for r in range(1, len(COMBO_BLOCKS) + 1):
        for combo in itertools.combinations(COMBO_BLOCKS, r):
            combos.append("+".join(combo))
    return combos


def extract_features_for_image_v2(image, mask, feature_set):
    features = {}
    for name in resolve_modules(feature_set):
        features.update(ATOMIC_EXTRACTORS[name](image, mask))
    return features


def build_feature_table_v2(data_dir, image_ids, feature_set, mask_mode):
    rows = []
    for image_id in tqdm(image_ids, desc=f"  features[{feature_set}]", leave=False):
        image = load_image(data_dir, image_id)
        mask = prepare_mask(load_mask(data_dir, image_id), mask_mode=mask_mode)
        row = {"image_id": str(image_id)}
        row.update(extract_features_for_image_v2(image, mask, feature_set))
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 缓存 + Pipeline + 分类器网格 (与 run_ml_grid.py 行为一致)
# ----------------------------------------------------------------------------
def _safe_cache_name(feature_set):
    return feature_set.replace("+", "_AND_")


def _feature_table_cached(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    ensure_dir(cache_dir)
    cache_path = cache_dir / f"features_v2_{_safe_cache_name(feature_set)}_{mask_mode}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    feature_df = build_feature_table_v2(
        data_dir, labels["image_id"].astype(str), feature_set, mask_mode
    )
    feature_df.to_csv(cache_path, index=False)
    return feature_df


def _classifier_grid(names):
    grid = []
    if "svm" in names:
        for c_value in [10, 30, 100]:
            grid.append((
                "svm", f"C={c_value}",
                SVC(kernel="rbf", C=c_value, gamma="scale",
                    class_weight="balanced", random_state=42),
            ))
    if "rf" in names:
        for max_depth in [10, 15]:
            grid.append((
                "rf", f"depth={max_depth}",
                RandomForestClassifier(
                    n_estimators=300, max_depth=max_depth, max_features="sqrt",
                    class_weight="balanced", random_state=42, n_jobs=-1,
                ),
            ))
    if "lr" in names:
        for c_value in [0.3, 1.0]:
            grid.append((
                "lr", f"C={c_value}",
                LogisticRegression(C=c_value, max_iter=2000,
                                   class_weight="balanced", random_state=42),
            ))
    if "knn" in names:
        for n_neighbors in [5, 9]:
            grid.append((
                "knn", f"k={n_neighbors}",
                KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
            ))
    return grid


def _pipeline(classifier, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


def _make_splits(data, n_splits, random_state):
    y = data["dx"]
    groups = data["image_id"].map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(data.drop(columns=["dx"]), y, groups=groups))


# ----------------------------------------------------------------------------
# 主流程: 网格搜索
# ----------------------------------------------------------------------------
def run_grid_v2(data_dir, feature_sets, classifiers, k_features, mask_modes,
                output_csv, n_splits=5, random_state=127, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv is required.")
    labels["image_id"] = labels["image_id"].astype(str)

    clf_grid = _classifier_grid(classifiers)
    total = len(mask_modes) * len(feature_sets) * len(clf_grid) * len(k_features)
    print(f"\n>>> Total runs: {total}  "
          f"({len(feature_sets)} feature_sets x {len(clf_grid)} clf-variants "
          f"x {len(k_features)} k_features x {len(mask_modes)} mask_modes)\n")

    rows = []
    cache_dir = Path("outputs/cache")
    for mask_mode in mask_modes:
        for feature_set in feature_sets:
            print(f"--- {feature_set}  (mask={mask_mode}) ---")
            feature_df = _feature_table_cached(
                data_dir, labels, feature_set, mask_mode, cache_dir, use_cache
            )
            data = feature_df.merge(labels, on="image_id", how="inner")
            X = data.drop(columns=["image_id", "dx"])
            y = data["dx"]
            splits = _make_splits(data[["image_id", "dx"]], n_splits, random_state)

            for clf_name, clf_params, classifier in clf_grid:
                for k_value in k_features:
                    model = _pipeline(classifier, k_value, X.shape[1])
                    pred = pd.Series(index=data.index, dtype=object)
                    for train_idx, valid_idx in splits:
                        model.fit(X.iloc[train_idx], y.iloc[train_idx])
                        pred.iloc[valid_idx] = model.predict(X.iloc[valid_idx])
                    metrics = compute_metrics(y, pred, labels=LABELS)
                    rows.append({
                        "feature_set":       feature_set,
                        "n_blocks":          feature_set.count("+") + 1,
                        "mask_mode":         mask_mode,
                        "classifier":        clf_name,
                        "classifier_params": clf_params,
                        "k_features":        k_value,
                        "n_features":        X.shape[1],
                        "accuracy":          metrics["accuracy"],
                        "macro_f1":          metrics["macro_f1"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                    })
                    print(f"    {clf_name:<3s} {clf_params:<10s} k={k_value:<4s} "
                          f"bal_acc={metrics['balanced_accuracy']:.4f}  "
                          f"macro_f1={metrics['macro_f1']:.4f}")

    result = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved grid results to {output_csv}")
    return result


# ----------------------------------------------------------------------------
# 柱状图: 按 balanced_accuracy 降序, 横向, 按分类器配色
# ----------------------------------------------------------------------------
CLF_PALETTE = {
    "lr":  "#4C78A8",
    "svm": "#F58518",
    "rf":  "#54A24B",
    "knn": "#B279A2",
}


def plot_top_combos(result_df, output_png, top_n=20, metric="balanced_accuracy"):
    top = (result_df.sort_values(metric, ascending=False)
                    .head(top_n)
                    .reset_index(drop=True))

    top["label"] = top.apply(
        lambda r: f"{r['feature_set']}  |  {r['classifier']}({r['classifier_params']})  |  k={r['k_features']}",
        axis=1,
    )

    fig_h = max(5.0, 0.42 * len(top) + 1.6)
    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(top))[::-1]  # top-1 显示在最上方
    colors = [CLF_PALETTE.get(c, "#888888") for c in top["classifier"]]
    bars = ax.barh(y_pos, top[metric], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72, zorder=3)

    # 标注每条柱的数值
    xmax = float(top[metric].max())
    xmin = float(top[metric].min())
    pad = (xmax - xmin) * 0.04 if xmax > xmin else xmax * 0.01
    for bar, value in zip(bars, top[metric]):
        ax.text(value + pad * 0.25, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center", ha="left",
                fontsize=9, color="#222222", zorder=4)

    # 突出 Top-1
    bars[0].set_edgecolor("#1f1f1f")
    bars[0].set_linewidth(1.8)
    ax.text(top[metric].iloc[0] + pad * 0.25,
            bars[0].get_y() + bars[0].get_height() / 2 + 0.45,
            "★ best", color="#c62828", fontsize=10, weight="bold",
            va="center", ha="left", zorder=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["label"], fontsize=9)
    ax.set_xlabel("Balanced Accuracy" if metric == "balanced_accuracy"
                  else metric.replace("_", " ").title(),
                  fontsize=11, color="#333333")
    left = max(0.0, xmin - (xmax - xmin) * 0.08)
    right = min(1.0, xmax + (xmax - xmin) * 0.18 + 0.01)
    ax.set_xlim(left, right)
    ax.set_title(f"Top-{top_n} feature x classifier x k  (sorted by {metric})",
                 fontsize=13, weight="bold", pad=12, color="#1f1f1f")

    # 分类器图例
    present = top["classifier"].unique().tolist()
    handles = [Patch(facecolor=CLF_PALETTE.get(name, "#888"), label=name)
               for name in ["lr", "svm", "rf", "knn"] if name in present]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=10, title="Classifier", title_fontsize=10)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#cccccc")
    ax.tick_params(axis="x", colors="#444444")
    ax.tick_params(axis="y", colors="#222222")
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    ensure_dir(Path(output_png).parent)
    plt.savefig(output_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved bar chart to {output_png}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _expand_feature_sets(values):
    """支持 'all_31' 关键字: 一次性展开 31 种非空组合。"""
    out = []
    for v in values:
        if v == "all_31":
            out.extend(enumerate_all_combos())
        else:
            out.append(v)
    seen, deduped = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def main():
    parser = argparse.ArgumentParser(
        description="v2: free-form feature combination grid search + sorted bar chart."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--feature_sets",
        default="all_31",
        help="Comma-separated. 'all_31' enumerates all 31 non-empty combos of "
             "{all, contrast, abcd_v2, boundary, melnv}. "
             "Use '+' to compose blocks, e.g. 'boundary+melnv,all+contrast+boundary'.",
    )
    parser.add_argument("--classifiers", default="lr,svm",
                        help="Subset of {lr, svm, rf, knn}.")
    parser.add_argument("--k_features", default="100",
                        help="Comma-separated, e.g. 'all,60,100,160'.")
    parser.add_argument("--mask_modes", default="raw",
                        help="Comma-separated subset of {raw, clean}.")
    parser.add_argument("--output_csv",
                        default="outputs/metrics/ml_grid_v2.csv")
    parser.add_argument("--output_png",
                        default="outputs/figures/ml_grid_v2_top.png")
    parser.add_argument("--top_n", type=int, default=20)
    parser.add_argument("--metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "macro_f1", "accuracy"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=127)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    feature_sets = _expand_feature_sets(_parse_list(args.feature_sets))
    print(f"Resolved {len(feature_sets)} feature_sets: {feature_sets}")

    result = run_grid_v2(
        data_dir=Path(args.data_dir),
        feature_sets=feature_sets,
        classifiers=_parse_list(args.classifiers),
        k_features=_parse_list(args.k_features),
        mask_modes=_parse_list(args.mask_modes),
        output_csv=Path(args.output_csv),
        n_splits=args.n_splits,
        random_state=args.random_state,
        use_cache=not args.no_cache,
    )

    print(f"\n=== Top 10 by {args.metric} ===")
    print(result.head(10).to_string(index=False))

    plot_top_combos(result, Path(args.output_png),
                    top_n=args.top_n, metric=args.metric)


if __name__ == "__main__":
    main()

"""experiments/run_ensemble_v4.py

v4: 在 v3 找到的最佳 (feature_set, classifier) 基础上做集成提升:
    1. 多种子 bagging   - 平均同一模型在不同 CV 种子下的概率
    2. Stacking         - 基学习器 OOF 概率喂给元学习器 (默认 LR)
    3. 横评             - 同一张图里对比 "单模型 bagged" vs "stacking bagged"

依赖: pip install xgboost lightgbm  (沿用 v3 缓存, 不重提特征)

命令行示例 (单种子, 快速验证):
    python experiments/run_ensemble_v4.py --data_dir data\Data_Proj2 --seeds 127

完整 5 种子复现:
    python experiments/run_ensemble_v4.py --data_dir data\Data_Proj2 \
        --feature_set contrast+abcd_v2+boundary \
        --base_classifiers xgb,lr,lgbm \
        --seeds 42,127,2024,3407,520

输出:
    outputs/metrics/ml_ensemble_v4.csv      所有方法 x 种子组合的结果 + 跨种子 bagged
    outputs/figures/ml_ensemble_v4_top.png  方法对比柱状图 (按 balanced_accuracy 降序)
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore",
                        message=".*X does not have valid feature names.*",
                        category=UserWarning)

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

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ----------------------------------------------------------------------------
# 特征组合解析 (复用 v2/v3 缓存)
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


def resolve_modules(feature_set):
    if feature_set == "final":
        return list(ALL_BUNDLE) + ["contrast", "abcd_v2", "boundary"]
    if feature_set == "final_melnv":
        return list(ALL_BUNDLE) + ["contrast", "abcd_v2", "boundary", "melnv"]
    parts = [p.strip() for p in feature_set.split("+") if p.strip()]
    modules = []
    for part in parts:
        if part == "all":
            modules.extend(ALL_BUNDLE)
        elif part in ATOMIC_EXTRACTORS:
            modules.append(part)
        else:
            raise ValueError(f"Unknown block '{part}'.")
    seen, deduped = set(), []
    for m in modules:
        if m not in seen:
            seen.add(m); deduped.append(m)
    return deduped


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


def _feature_table_cached(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    ensure_dir(cache_dir)
    safe = feature_set.replace("+", "_AND_")
    cache_path = cache_dir / f"features_v2_{safe}_{mask_mode}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    df = build_feature_table_v2(data_dir, labels["image_id"].astype(str),
                                feature_set, mask_mode)
    df.to_csv(cache_path, index=False)
    return df


# ----------------------------------------------------------------------------
# 基学习器构造 (v3 找到的最佳超参; 想换就改这里)
# ----------------------------------------------------------------------------
def _build_base_classifier(name):
    """返回 (Pipeline, needs_sample_weight) ."""
    needs_sw = False
    if name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost not installed.")
        clf = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.60, reg_lambda=3.0,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        )
        needs_sw = True
    elif name == "lgbm":
        if not HAS_LGBM:
            raise ImportError("lightgbm not installed.")
        clf = LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05,
            min_child_samples=30, reg_lambda=3.0,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
    elif name == "lr":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42,
        )
    elif name == "svm":
        clf = SVC(
            kernel="rbf", C=30, gamma="scale", probability=True,
            class_weight="balanced", random_state=42,
        )
    else:
        raise ValueError(f"Unknown base classifier '{name}'.")
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    return pipe, needs_sw


def _build_meta(name):
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000,
                                  class_weight="balanced", random_state=42)
    if name == "lr_strong":
        return LogisticRegression(C=10.0, max_iter=2000,
                                  class_weight="balanced", random_state=42)
    raise ValueError(f"Unknown meta '{name}'.")


# ----------------------------------------------------------------------------
# Grouped CV 工具
# ----------------------------------------------------------------------------
def _make_splits(data, n_splits, random_state):
    y = data["dx"]
    groups = data["image_id"].map(base_id_from_image_id)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(data.drop(columns=["dx"]), y, groups=groups))


def _oof_proba(model, X, y_enc, splits, n_classes, needs_sw):
    """grouped CV 下生成 OOF predict_proba (n_samples, n_classes).
    每折用 train 子集 fit, 在 val 子集上 predict_proba."""
    oof = np.zeros((len(X), n_classes), dtype=np.float32)
    for tr, va in splits:
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y_enc.iloc[tr].values
        if needs_sw:
            sw = compute_sample_weight("balanced", y_tr)
            model.fit(X_tr, y_tr, clf__sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        oof[va] = model.predict_proba(X_va)
    return oof


# ----------------------------------------------------------------------------
# 主流程: 多种子 bagging + stacking
# ----------------------------------------------------------------------------
def run_ensemble_v4(data_dir, feature_set, base_names, meta_name,
                    seeds, n_splits, mask_mode, output_csv, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv required.")
    labels["image_id"] = labels["image_id"].astype(str)

    cache_dir = Path("outputs/cache")
    feature_df = _feature_table_cached(data_dir, labels, feature_set, mask_mode,
                                       cache_dir, use_cache)
    data = feature_df.merge(labels, on="image_id", how="inner")
    X = data.drop(columns=["image_id", "dx"])
    y = data["dx"]

    le = LabelEncoder().fit(LABELS)
    y_enc = pd.Series(le.transform(y), index=y.index)
    n_classes = len(LABELS)

    print(f"\n>>> feature_set: {feature_set}  n_features={X.shape[1]}  n_samples={len(X)}")
    print(f">>> base: {base_names}  meta: {meta_name}  seeds: {seeds}\n")

    # 每个种子 -> 每个基学习器的 OOF 概率 + 元学习器的 OOF 概率
    seed_base_probs = {name: [] for name in base_names}   # name -> [seed]: (n, 3)
    seed_meta_probs = []                                   # [seed]: (n, 3)
    per_seed_rows = []

    for seed in seeds:
        print(f"--- seed={seed} ---")
        splits = _make_splits(data[["image_id", "dx"]], n_splits, seed)

        # Step 1: 各基学习器的 OOF 概率
        base_probs_this_seed = {}
        for name in base_names:
            pipe, needs_sw = _build_base_classifier(name)
            probs = _oof_proba(pipe, X, y_enc, splits, n_classes, needs_sw)
            base_probs_this_seed[name] = probs
            seed_base_probs[name].append(probs)

            # 单基学习器分数 (该 seed)
            pred_enc = probs.argmax(axis=1)
            pred = le.inverse_transform(pred_enc)
            m = compute_metrics(y, pred, labels=LABELS)
            per_seed_rows.append({
                "method": f"single_{name}", "seed": seed,
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            })
            print(f"  single {name:<5s}  bal_acc={m['balanced_accuracy']:.4f}  "
                  f"macro_f1={m['macro_f1']:.4f}")

        # Step 2: 把基学习器概率拼成 meta 特征, 跑 OOF 元学习器
        meta_X = pd.DataFrame(
            np.hstack([base_probs_this_seed[name] for name in base_names]),
            columns=[f"{n}_p{c}" for n in base_names for c in LABELS],
        )
        meta = _build_meta(meta_name)
        meta_pipe = Pipeline([("scaler", StandardScaler()), ("clf", meta)])
        meta_probs = _oof_proba(meta_pipe, meta_X, y_enc, splits, n_classes, needs_sw=False)
        seed_meta_probs.append(meta_probs)

        # 该 seed 的 stacking 分数
        pred_enc = meta_probs.argmax(axis=1)
        pred = le.inverse_transform(pred_enc)
        m = compute_metrics(y, pred, labels=LABELS)
        per_seed_rows.append({
            "method": f"stack({'+'.join(base_names)})->{meta_name}", "seed": seed,
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        })
        print(f"  STACK  ->{meta_name:<3s}  bal_acc={m['balanced_accuracy']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")

    # ------------------------------------------------------------------------
    # 跨种子 bagging: 平均概率 -> argmax
    # ------------------------------------------------------------------------
    bagged_rows = []

    def _bagged_row(method, probs_list):
        avg = np.mean(probs_list, axis=0)
        pred_enc = avg.argmax(axis=1)
        pred = le.inverse_transform(pred_enc)
        m = compute_metrics(y, pred, labels=LABELS)
        return {
            "method": method, "seed": "bagged",
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        }

    print("\n=== bagged across seeds ===")
    for name in base_names:
        row = _bagged_row(f"bagged_{name}", seed_base_probs[name])
        bagged_rows.append(row)
        print(f"  bagged {name:<5s}  bal_acc={row['balanced_accuracy']:.4f}  "
              f"macro_f1={row['macro_f1']:.4f}")
    row = _bagged_row(f"bagged_stack({'+'.join(base_names)})->{meta_name}",
                      seed_meta_probs)
    bagged_rows.append(row)
    print(f"  bagged STACK  bal_acc={row['balanced_accuracy']:.4f}  "
          f"macro_f1={row['macro_f1']:.4f}")

    # 输出 CSV
    result = pd.DataFrame(per_seed_rows + bagged_rows)
    result = result.sort_values("balanced_accuracy", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return result, bagged_rows


# ----------------------------------------------------------------------------
# 对比柱状图: 各方法的 bagged + 各种子分布
# ----------------------------------------------------------------------------
METHOD_PALETTE = {
    "single": "#cccccc",
    "bagged": "#4C78A8",
    "stack":  "#E45756",
}


def _method_color(method):
    if method.startswith("bagged_stack"):
        return "#c62828"          # 高亮 stacking bagged
    if method.startswith("bagged_"):
        return METHOD_PALETTE["bagged"]
    if method.startswith("stack"):
        return "#f08080"          # stacking 单种子
    return METHOD_PALETTE["single"]


def plot_methods(result_df, output_png, metric="balanced_accuracy"):
    """画两种视图: 上面 bagged 的横向条形, 下面单种子的散点 (方法分组)."""
    bagged = result_df[result_df["seed"] == "bagged"].copy()
    per_seed = result_df[result_df["seed"] != "bagged"].copy()

    bagged = bagged.sort_values(metric, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.55 * len(bagged) + 2)), dpi=120)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(bagged))
    colors = [_method_color(m) for m in bagged["method"]]
    bars = ax.barh(y_pos, bagged[metric], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72, zorder=3)

    # 数值标注
    xmax = float(bagged[metric].max())
    xmin = float(bagged[metric].min())
    pad = (xmax - xmin) * 0.04 if xmax > xmin else xmax * 0.01
    for bar, val in zip(bars, bagged[metric]):
        ax.text(val + pad * 0.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=10, color="#222222", zorder=4)

    # 单种子散点叠加, 看方差
    method_to_y = dict(zip(bagged["method"], y_pos))
    for _, row in per_seed.iterrows():
        # per_seed 的 method 名跟 bagged 不同 ('single_xgb' vs 'bagged_xgb')
        # 我们映射: single_X -> bagged_X, stack(...) -> bagged_stack(...)
        m_seed = row["method"]
        if m_seed.startswith("single_"):
            base = m_seed.split("single_")[1]
            target = f"bagged_{base}"
        elif m_seed.startswith("stack"):
            target = f"bagged_{m_seed}"
        else:
            continue
        if target in method_to_y:
            ax.scatter(row[metric], method_to_y[target],
                       color="#444444", s=22, zorder=5, alpha=0.65,
                       edgecolor="white", linewidth=0.5)

    # 高亮最佳 bagged
    best_idx = bagged[metric].idxmax()
    best_bar = bars[best_idx]
    best_bar.set_edgecolor("#1f1f1f")
    best_bar.set_linewidth(1.8)
    ax.text(bagged[metric].iloc[best_idx] + pad * 0.25,
            best_bar.get_y() + best_bar.get_height() / 2 + 0.42,
            "★ best", color="#c62828", fontsize=11, weight="bold",
            va="center", ha="left", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bagged["method"], fontsize=10)
    ax.set_xlabel("Balanced Accuracy" if metric == "balanced_accuracy"
                  else metric.replace("_", " ").title(),
                  fontsize=11, color="#333333")
    left = max(0.0, xmin - (xmax - xmin) * 0.10)
    right = min(1.0, xmax + (xmax - xmin) * 0.20 + 0.01)
    ax.set_xlim(left, right)
    ax.set_title(f"Single vs Bagging vs Stacking  (sorted by bagged {metric})",
                 fontsize=13, weight="bold", pad=12, color="#1f1f1f")

    handles = [
        Patch(facecolor="#c62828", label="Stacking (bagged)"),
        Patch(facecolor=METHOD_PALETTE["bagged"], label="Single model (bagged)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
                   markersize=7, label="Per-seed score"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=10)

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
    print(f"Saved chart to {output_png}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="v4: multi-seed bagging + stacking on top of v3 winners."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--feature_set", default="contrast+abcd_v2+boundary",
                        help="v3 找到的赢家. 用 '+' 拼接.")
    parser.add_argument("--base_classifiers", default="xgb,lr,lgbm",
                        help="Subset of {xgb, lr, lgbm, svm}.")
    parser.add_argument("--meta_classifier", default="lr",
                        choices=["lr", "lr_strong"])
    parser.add_argument("--seeds", default="42,127,2024,3407,520",
                        help="跑这些种子的 grouped CV, 然后概率平均 bagging.")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/ml_ensemble_v4.csv")
    parser.add_argument("--output_png", default="outputs/figures/ml_ensemble_v4_top.png")
    parser.add_argument("--metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "macro_f1", "accuracy"])
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in _parse_list(args.seeds)]
    base_names = _parse_list(args.base_classifiers)

    result, _ = run_ensemble_v4(
        data_dir=Path(args.data_dir),
        feature_set=args.feature_set,
        base_names=base_names,
        meta_name=args.meta_classifier,
        seeds=seeds,
        n_splits=args.n_splits,
        mask_mode=args.mask_mode,
        output_csv=Path(args.output_csv),
        use_cache=not args.no_cache,
    )

    print(f"\n=== Top rows by {args.metric} ===")
    print(result.head(12).to_string(index=False))

    plot_methods(result, Path(args.output_png), metric=args.metric)


if __name__ == "__main__":
    main()

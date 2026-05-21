"""experiments/run_ensemble_v5.py

v5: Multi-view stacking - 每个基学习器吃自己的最佳特征集 (而非共享一套).

相对 v4 的关键变化:
1. 每个基学习器有独立的 (classifier, feature_set, k_features) 三元组
2. XGB 用 v3 真正的赢家配置 (n=150, d=2, lr=0.05, ss=0.80, cs=0.60, lambda=5.0)
3. 跨基学习器的样本顺序通过 sort by image_id 强制对齐, 确保 OOF 概率可以 hstack

依赖: pip install xgboost
缓存: 沿用 v2 cache, 不重提特征

默认配置 (反映目前已知最佳):
    xgb on contrast+abcd_v2+boundary, k=100  (v3 winner, 0.78)
    lr  on all+boundary,              k=100  (README winner, 0.75)
    meta: LR (C=1)

命令行 (单种子快验):
    python experiments/run_ensemble_v5.py --data_dir data\Data_Proj2 --seeds 127

五种子 bagging:
    python experiments/run_ensemble_v5.py --data_dir data\Data_Proj2 \
        --seeds 42,127,2024,3407,520

自定义 base specs (格式: classifier:feature_set:k_features, 用逗号分隔):
    python experiments/run_ensemble_v5.py --data_dir data\Data_Proj2 \
        --base_specs "xgb:contrast+abcd_v2+boundary:100,lr:all+boundary:100,svm:all+boundary:60"
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


# ----------------------------------------------------------------------------
# 特征组合解析 (复用 v2/v3/v4 缓存)
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
# 基学习器构造: 把 v3 找到的具体最佳超参写死在这里
# ----------------------------------------------------------------------------
def _build_base_classifier(name):
    """返回 (classifier_instance, needs_sample_weight)"""
    if name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost not installed. Run `pip install xgboost`.")
        # v3-a winner variant 1: n=150, d=2, lr=0.05 (用户报告 0.78)
        clf = XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0,
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        )
        return clf, True
    if name == "lr":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight="balanced", random_state=42,
        )
        return clf, False
    if name == "svm":
        clf = SVC(
            kernel="rbf", C=30, gamma="scale", probability=True,
            class_weight="balanced", random_state=42,
        )
        return clf, False
    raise ValueError(f"Unknown base classifier '{name}'.")


def _build_meta(name):
    if name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000,
                                  class_weight="balanced", random_state=42)
    if name == "lr_strong":
        return LogisticRegression(C=10.0, max_iter=2000,
                                  class_weight="balanced", random_state=42)
    if name == "average":
        return None  # 等权平均, 不学习权重
    raise ValueError(f"Unknown meta '{name}'.")


def _build_pipeline(classifier, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# CV / OOF helpers
# ----------------------------------------------------------------------------
def _make_splits(master_labels, n_splits, random_state):
    """以 master_labels (按 image_id 排序后) 为基准计算 splits, 所有
    base learner 复用同一组 splits, 保证 OOF 概率行对齐."""
    y = master_labels["dx"]
    groups = master_labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=master_labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(placeholder, y, groups=groups))


def _oof_proba(model, X, y_enc, splits, n_classes, needs_sw):
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
# 主流程
# ----------------------------------------------------------------------------
def run_ensemble_v5(data_dir, base_specs, meta_name, seeds, n_splits,
                    mask_mode, output_csv, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv required.")
    labels["image_id"] = labels["image_id"].astype(str)

    # 关键: master_labels 保持 label.csv 的原始行序 (与 v3 / run_ml_grid.py 一致),
    # 后续所有 base learner 的 feature_df 都 merge 到这上面, 保证行序对齐.
    # 注意: StratifiedGroupKFold(shuffle=True) 的折划分依赖输入行序,
    # 如果在这里 sort_values("image_id") 会拿到与 v3 不同的折边界, 同 seed 下
    # OOF 分数会差 ~0.005 (虽然仍在 CV 噪声范围内, 但难以与 v3 直接比较).
    master_labels = labels.reset_index(drop=True)
    y = master_labels["dx"]
    le = LabelEncoder().fit(LABELS)
    y_enc = pd.Series(le.transform(y), index=y.index)
    n_classes = len(LABELS)

    print(f"\n>>> n_samples: {len(master_labels)}")
    print(">>> Base specs:")
    for spec in base_specs:
        print(f"    {spec['classifier']:<5s} on {spec['feature_set']:<35s} k={spec['k_features']}")
    print(f">>> Meta: {meta_name}")
    print(f">>> Seeds: {seeds}\n")

    cache_dir = Path("outputs/cache")

    # 预加载所有 base spec 的特征矩阵 (一次性, 后续每个种子复用)
    base_X = []
    for spec in base_specs:
        feature_df = _feature_table_cached(
            data_dir, master_labels, spec["feature_set"], mask_mode,
            cache_dir, use_cache,
        )
        feature_df["image_id"] = feature_df["image_id"].astype(str)
        merged = master_labels.merge(feature_df, on="image_id", how="inner")
        # merged 跟 master_labels 同行序; 但 inner merge 可能丢行
        assert len(merged) == len(master_labels), (
            f"Feature table for {spec['feature_set']} missing rows "
            f"({len(merged)} vs {len(master_labels)})"
        )
        X_spec = merged.drop(columns=["image_id", "dx"])
        base_X.append(X_spec)
        print(f"    [{spec['classifier']}] X shape = {X_spec.shape}")
    print()

    seed_base_probs = [[] for _ in base_specs]   # base_idx -> [seed]: (n, 3)
    seed_meta_probs = []                          # [seed]: (n, 3)
    per_seed_rows = []

    for seed in seeds:
        print(f"--- seed={seed} ---")
        splits = _make_splits(master_labels, n_splits, seed)

        # Step 1: 每个 base learner 用自己的特征矩阵, 共用 splits
        base_probs_this_seed = []
        for i, spec in enumerate(base_specs):
            clf, needs_sw = _build_base_classifier(spec["classifier"])
            pipe = _build_pipeline(clf, spec["k_features"], base_X[i].shape[1])
            probs = _oof_proba(pipe, base_X[i], y_enc, splits, n_classes, needs_sw)
            base_probs_this_seed.append(probs)
            seed_base_probs[i].append(probs)

            pred = le.inverse_transform(probs.argmax(axis=1))
            m = compute_metrics(y, pred, labels=LABELS)
            label = f"single_{spec['classifier']}({spec['feature_set']})"
            per_seed_rows.append({
                "method": label, "seed": seed,
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
            })
            print(f"  single {spec['classifier']:<3s} on {spec['feature_set']:<32s}  "
                  f"bal_acc={m['balanced_accuracy']:.4f}  macro_f1={m['macro_f1']:.4f}")

        # Step 2: stacking
        meta_X = pd.DataFrame(
            np.hstack(base_probs_this_seed),
            columns=[f"{spec['classifier']}_p{c}"
                     for spec in base_specs for c in LABELS],
        )
        meta = _build_meta(meta_name)
        if meta is None:
            meta_probs = np.mean(base_probs_this_seed, axis=0)
            method_label = f"avg({'+'.join(s['classifier'] for s in base_specs)})"
        else:
            meta_pipe = Pipeline([("scaler", StandardScaler()), ("clf", meta)])
            meta_probs = _oof_proba(meta_pipe, meta_X, y_enc, splits,
                                    n_classes, needs_sw=False)
            method_label = f"stack({'+'.join(s['classifier'] for s in base_specs)})->{meta_name}"
        seed_meta_probs.append(meta_probs)

        pred = le.inverse_transform(meta_probs.argmax(axis=1))
        m = compute_metrics(y, pred, labels=LABELS)
        per_seed_rows.append({
            "method": method_label, "seed": seed,
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        })
        print(f"  {method_label:<55s}  bal_acc={m['balanced_accuracy']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")

    # 跨种子 bagging
    bagged_rows = []

    def _bagged(method, probs_list):
        avg = np.mean(probs_list, axis=0)
        pred = le.inverse_transform(avg.argmax(axis=1))
        m = compute_metrics(y, pred, labels=LABELS)
        return {
            "method": method, "seed": "bagged",
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        }

    print("\n=== bagged across seeds ===")
    for i, spec in enumerate(base_specs):
        label = f"bagged_{spec['classifier']}({spec['feature_set']})"
        row = _bagged(label, seed_base_probs[i])
        bagged_rows.append(row)
        print(f"  bagged {spec['classifier']:<3s} on {spec['feature_set']:<32s}  "
              f"bal_acc={row['balanced_accuracy']:.4f}  macro_f1={row['macro_f1']:.4f}")
    final_label = f"bagged_{method_label}"
    row = _bagged(final_label, seed_meta_probs)
    bagged_rows.append(row)
    print(f"  {final_label:<55s}  bal_acc={row['balanced_accuracy']:.4f}  "
          f"macro_f1={row['macro_f1']:.4f}")

    result = pd.DataFrame(per_seed_rows + bagged_rows)
    result = result.sort_values("balanced_accuracy", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return result


# ----------------------------------------------------------------------------
# 对比柱状图
# ----------------------------------------------------------------------------
def _method_color(method):
    if method.startswith("bagged_stack") or method.startswith("bagged_avg"):
        return "#c62828"
    if method.startswith("bagged_"):
        return "#4C78A8"
    if method.startswith("stack") or method.startswith("avg"):
        return "#f08080"
    return "#cccccc"


def plot_methods(result_df, output_png, metric="balanced_accuracy"):
    bagged = result_df[result_df["seed"] == "bagged"].copy()
    per_seed = result_df[result_df["seed"] != "bagged"].copy()
    bagged = bagged.sort_values(metric, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, max(4.5, 0.6 * len(bagged) + 2)), dpi=120)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(bagged))
    colors = [_method_color(m) for m in bagged["method"]]
    bars = ax.barh(y_pos, bagged[metric], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72, zorder=3)

    xmax = float(bagged[metric].max())
    xmin = float(bagged[metric].min())
    pad = (xmax - xmin) * 0.04 if xmax > xmin else xmax * 0.01
    for bar, val in zip(bars, bagged[metric]):
        ax.text(val + pad * 0.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=10, color="#222222", zorder=4)

    # 单种子散点
    def _bagged_target(m_seed):
        if m_seed.startswith("single_"):
            return "bagged_" + m_seed.split("single_")[1]
        if m_seed.startswith("stack") or m_seed.startswith("avg"):
            return "bagged_" + m_seed
        return None

    method_to_y = dict(zip(bagged["method"], y_pos))
    for _, row in per_seed.iterrows():
        tgt = _bagged_target(row["method"])
        if tgt and tgt in method_to_y:
            ax.scatter(row[metric], method_to_y[tgt],
                       color="#444444", s=22, zorder=5, alpha=0.65,
                       edgecolor="white", linewidth=0.5)

    best_idx = bagged[metric].idxmax()
    best_bar = bars[best_idx]
    best_bar.set_edgecolor("#1f1f1f")
    best_bar.set_linewidth(1.8)
    ax.text(bagged[metric].iloc[best_idx] + pad * 0.25,
            best_bar.get_y() + best_bar.get_height() / 2 + 0.42,
            "★ best", color="#c62828", fontsize=11, weight="bold",
            va="center", ha="left", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bagged["method"], fontsize=9)
    ax.set_xlabel("Balanced Accuracy" if metric == "balanced_accuracy"
                  else metric.replace("_", " ").title(),
                  fontsize=11, color="#333333")
    left = max(0.0, xmin - (xmax - xmin) * 0.10)
    right = min(1.0, xmax + (xmax - xmin) * 0.20 + 0.01)
    ax.set_xlim(left, right)
    ax.set_title(f"Multi-view stacking  (sorted by bagged {metric})",
                 fontsize=13, weight="bold", pad=12, color="#1f1f1f")

    handles = [
        Patch(facecolor="#c62828", label="Stacking / Average (bagged)"),
        Patch(facecolor="#4C78A8", label="Single base (bagged)"),
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
def _parse_base_specs(value):
    """格式: 'xgb:contrast+abcd_v2+boundary:100,lr:all+boundary:100'"""
    specs = []
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Bad spec '{entry}'. Expected 'classifier:feature_set:k_features'."
            )
        clf, fs, k = parts
        specs.append({
            "classifier": clf.strip(),
            "feature_set": fs.strip(),
            "k_features": k.strip(),
        })
    return specs


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="v5: Multi-view stacking - each base learner uses its own best feature_set."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--base_specs",
        default="xgb:contrast+abcd_v2+boundary:100,lr:all+boundary:100",
        help="Comma-separated triples 'classifier:feature_set:k_features'. "
             "Default reflects v3 + README winners.",
    )
    parser.add_argument("--meta_classifier", default="lr",
                        choices=["lr", "lr_strong", "average"],
                        help="'average' = 等权平均概率, 不训练 meta-LR.")
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/ml_ensemble_v5.csv")
    parser.add_argument("--output_png", default="outputs/figures/ml_ensemble_v5_top.png")
    parser.add_argument("--metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "macro_f1", "accuracy"])
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    base_specs = _parse_base_specs(args.base_specs)
    seeds = [int(s) for s in _parse_list(args.seeds)]

    result = run_ensemble_v5(
        data_dir=Path(args.data_dir),
        base_specs=base_specs,
        meta_name=args.meta_classifier,
        seeds=seeds,
        n_splits=args.n_splits,
        mask_mode=args.mask_mode,
        output_csv=Path(args.output_csv),
        use_cache=not args.no_cache,
    )

    print(f"\n=== All rows by {args.metric} ===")
    print(result.to_string(index=False))

    plot_methods(result, Path(args.output_png), metric=args.metric)


if __name__ == "__main__":
    main()

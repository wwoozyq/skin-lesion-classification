"""experiments/run_hierarchical_v6.py

v6: 二级 (hierarchical / cascade) 分类器, 专门对症下药 mel/nv 混淆.

架构:
    Stage 1:  vasc vs (mel ∪ nv)            二分类
    Stage 2:  mel vs nv (仅在非 vasc 样本)   二分类

每个阶段可以独立选 classifier / feature_set / k_features.
对比基线: 同样 seed/splits 下的 flat 3 分类 XGB.

两种 cascade 模式:
    soft (默认): 概率组合
        P(vasc) = p1(vasc)
        P(mel)  = (1 - p1(vasc)) * p2(mel)
        P(nv)   = (1 - p1(vasc)) * p2(nv)
        最终 argmax
    hard: 阈值判定
        if p1(vasc) > thresh -> vasc
        else -> argmax(stage2)

依赖: pip install xgboost
缓存: 沿用 features_v2_*.csv

命令行示例 (5 种子 bagging, 默认配置):
    python experiments/run_hierarchical_v6.py --data_dir data\Data_Proj2

自定义 stage:
    python experiments/run_hierarchical_v6.py --data_dir data\Data_Proj2 \
        --stage1 "xgb:contrast+abcd_v2+boundary:100" \
        --stage2 "xgb:all+boundary+melnv:120" \
        --mode soft

只跑一个种子快验:
    python experiments/run_hierarchical_v6.py --data_dir data\Data_Proj2 --seeds 127
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore",
                        message=".*X does not have valid feature names.*",
                        category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import LABELS  # ["mel", "nv", "vasc"]
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
# 特征组合 (与 v2/v3/v4/v5 共享缓存)
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
# 分类器工厂 (沿用 v5 / v3-a 的最佳超参)
# ----------------------------------------------------------------------------
def _build_classifier(name, n_classes):
    """二分类和多分类 XGB 参数略不同 (objective)."""
    if name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost not installed.")
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        eval_metric = "logloss" if n_classes == 2 else "mlogloss"
        clf = XGBClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0,
            objective=objective, eval_metric=eval_metric,
            tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
        )
        return clf, True
    if name == "lr":
        clf = LogisticRegression(C=1.0, max_iter=2000,
                                 class_weight="balanced", random_state=42)
        return clf, False
    if name == "svm":
        clf = SVC(kernel="rbf", C=30, gamma="scale", probability=True,
                  class_weight="balanced", random_state=42)
        return clf, False
    raise ValueError(f"Unknown classifier '{name}'.")


def _build_pipeline(classifier, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# CV / 拟合工具
# ----------------------------------------------------------------------------
def _make_splits(master_labels, n_splits, random_state):
    y = master_labels["dx"]
    groups = master_labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=master_labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(placeholder, y, groups=groups))


def _fit_predict_proba(spec, X_train, y_train, X_val, n_classes):
    """构造 pipeline -> fit -> predict_proba on val.
    spec = {"classifier", "feature_set", "k_features"}"""
    clf, needs_sw = _build_classifier(spec["classifier"], n_classes)
    pipe = _build_pipeline(clf, spec["k_features"], X_train.shape[1])
    if needs_sw:
        sw = compute_sample_weight("balanced", y_train)
        pipe.fit(X_train, y_train, clf__sample_weight=sw)
    else:
        pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_val)


# ----------------------------------------------------------------------------
# 主流程: 二级 cascade + flat 基线对比
# ----------------------------------------------------------------------------
def run_hierarchical_v6(data_dir, stage1_spec, stage2_spec, flat_spec,
                        mode, vasc_threshold, seeds, n_splits, mask_mode,
                        output_csv, use_cache=True):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv required.")
    labels["image_id"] = labels["image_id"].astype(str)
    master_labels = labels.reset_index(drop=True)  # 与 v5 保持一致的 CSV 行序

    n = len(master_labels)
    y_str = master_labels["dx"].values         # 'mel'/'nv'/'vasc'
    print(f"\n>>> n_samples: {n}")
    print(f">>> Stage 1: {stage1_spec['classifier']:<5s} on "
          f"{stage1_spec['feature_set']:<35s} k={stage1_spec['k_features']}")
    print(f">>> Stage 2: {stage2_spec['classifier']:<5s} on "
          f"{stage2_spec['feature_set']:<35s} k={stage2_spec['k_features']}")
    print(f">>> Flat baseline: {flat_spec['classifier']:<5s} on "
          f"{flat_spec['feature_set']:<35s} k={flat_spec['k_features']}")
    print(f">>> Mode: {mode}   vasc_threshold (hard mode): {vasc_threshold}")
    print(f">>> Seeds: {seeds}\n")

    # 预加载三个 spec 的特征矩阵 (相同的 spec 会复用缓存)
    cache_dir = Path("outputs/cache")
    feature_cache = {}
    for spec in (stage1_spec, stage2_spec, flat_spec):
        fs = spec["feature_set"]
        if fs in feature_cache:
            continue
        df = _feature_table_cached(data_dir, master_labels, fs, mask_mode,
                                   cache_dir, use_cache)
        df["image_id"] = df["image_id"].astype(str)
        merged = master_labels.merge(df, on="image_id", how="inner")
        assert len(merged) == n
        X = merged.drop(columns=["image_id", "dx"])
        feature_cache[fs] = X
        print(f"    [{fs}] X shape = {X.shape}")
    print()

    # 类别索引映射
    LABEL_IDX = {label: i for i, label in enumerate(LABELS)}   # mel=0, nv=1, vasc=2
    vasc_idx = LABEL_IDX["vasc"]
    mel_idx = LABEL_IDX["mel"]
    nv_idx = LABEL_IDX["nv"]

    seed_cascade_probs = []   # [seed]: (n, 3) 3 类概率
    seed_flat_probs = []
    per_seed_rows = []

    for seed in seeds:
        print(f"--- seed={seed} ---")
        splits = _make_splits(master_labels, n_splits, seed)

        cascade_probs = np.zeros((n, 3), dtype=np.float32)
        flat_probs = np.zeros((n, 3), dtype=np.float32)

        # 跟踪 stage 1/2 自身的二分类指标
        s1_correct, s1_total = 0, 0
        s2_correct, s2_total = 0, 0

        X_s1 = feature_cache[stage1_spec["feature_set"]]
        X_s2 = feature_cache[stage2_spec["feature_set"]]
        X_flat = feature_cache[flat_spec["feature_set"]]

        for fold_idx, (tr, va) in enumerate(splits):
            y_tr_str = y_str[tr]
            y_va_str = y_str[va]

            # === Stage 1: vasc vs not_vasc ===
            y_tr_s1 = (y_tr_str == "vasc").astype(int)
            X_tr_s1 = X_s1.iloc[tr]
            X_va_s1 = X_s1.iloc[va]
            proba_s1 = _fit_predict_proba(stage1_spec, X_tr_s1, y_tr_s1,
                                          X_va_s1, n_classes=2)
            # proba_s1[:, 1] = P(vasc)
            p_vasc = proba_s1[:, 1]

            # stage 1 自身准确率
            s1_pred = (p_vasc > 0.5).astype(int)
            s1_true = (y_va_str == "vasc").astype(int)
            s1_correct += int((s1_pred == s1_true).sum())
            s1_total += len(s1_true)

            # === Stage 2: mel vs nv (仅在训练集的 mel+nv 子集上 fit) ===
            mn_mask_tr = (y_tr_str != "vasc")
            X_tr_s2 = X_s2.iloc[tr].iloc[mn_mask_tr]
            y_tr_s2 = (y_tr_str[mn_mask_tr] == "nv").astype(int)  # mel=0, nv=1

            X_va_s2 = X_s2.iloc[va]
            proba_s2 = _fit_predict_proba(stage2_spec, X_tr_s2, y_tr_s2,
                                          X_va_s2, n_classes=2)
            # proba_s2[:, 0]=P(mel), proba_s2[:, 1]=P(nv)
            p_mel = proba_s2[:, 0]
            p_nv = proba_s2[:, 1]

            # stage 2 自身准确率 (只算真实的 mel/nv 样本)
            mn_mask_va = (y_va_str != "vasc")
            if mn_mask_va.any():
                s2_pred = np.where(p_mel[mn_mask_va] > p_nv[mn_mask_va],
                                   "mel", "nv")
                s2_correct += int((s2_pred == y_va_str[mn_mask_va]).sum())
                s2_total += int(mn_mask_va.sum())

            # === 组合成 3 类概率 ===
            if mode == "soft":
                # P(vasc) = p_vasc
                # P(mel)  = (1 - p_vasc) * p_mel
                # P(nv)   = (1 - p_vasc) * p_nv
                cascade_probs[va, vasc_idx] = p_vasc
                cascade_probs[va, mel_idx] = (1 - p_vasc) * p_mel
                cascade_probs[va, nv_idx] = (1 - p_vasc) * p_nv
            elif mode == "hard":
                # 硬阈值: p_vasc > thresh -> vasc; 否则 stage2 argmax
                is_vasc = p_vasc > vasc_threshold
                cascade_probs[va, vasc_idx] = np.where(is_vasc, 1.0, 0.0)
                # 非 vasc 时, mel/nv 由 stage2 概率决定 (归一到剩余概率 1)
                cascade_probs[va, mel_idx] = np.where(is_vasc, 0.0, p_mel)
                cascade_probs[va, nv_idx] = np.where(is_vasc, 0.0, p_nv)
            else:
                raise ValueError(f"Unknown mode '{mode}'.")

            # === Flat baseline: 3 类直接训 ===
            X_tr_flat = X_flat.iloc[tr]
            X_va_flat = X_flat.iloc[va]
            # 用 LABEL_IDX 把字符串映射成 0/1/2
            y_tr_flat = np.vectorize(LABEL_IDX.get)(y_tr_str).astype(int)
            proba_flat = _fit_predict_proba(flat_spec, X_tr_flat, y_tr_flat,
                                            X_va_flat, n_classes=3)
            # XGB 输出列顺序按 sklearn classes_, 但因为我们 fit 时用 0/1/2,
            # XGB 学习到的 classes_ 一般是 [0, 1, 2], 列对应 LABELS 顺序
            flat_probs[va, :] = proba_flat

        seed_cascade_probs.append(cascade_probs)
        seed_flat_probs.append(flat_probs)

        # 本 seed 的最终预测和指标
        cascade_pred = np.array(LABELS)[cascade_probs.argmax(axis=1)]
        flat_pred = np.array(LABELS)[flat_probs.argmax(axis=1)]
        m_c = compute_metrics(y_str, cascade_pred, labels=LABELS)
        m_f = compute_metrics(y_str, flat_pred, labels=LABELS)

        per_seed_rows.append({
            "method": f"cascade_{mode}", "seed": seed,
            "balanced_accuracy": m_c["balanced_accuracy"],
            "macro_f1": m_c["macro_f1"], "accuracy": m_c["accuracy"],
        })
        per_seed_rows.append({
            "method": "flat_3class", "seed": seed,
            "balanced_accuracy": m_f["balanced_accuracy"],
            "macro_f1": m_f["macro_f1"], "accuracy": m_f["accuracy"],
        })

        print(f"  stage1 (vasc detect)  acc={s1_correct/s1_total:.4f}")
        print(f"  stage2 (mel vs nv)    acc={s2_correct/max(s2_total,1):.4f}")
        print(f"  cascade ({mode})      bal_acc={m_c['balanced_accuracy']:.4f}  "
              f"macro_f1={m_c['macro_f1']:.4f}  acc={m_c['accuracy']:.4f}")
        print(f"  flat 3-class          bal_acc={m_f['balanced_accuracy']:.4f}  "
              f"macro_f1={m_f['macro_f1']:.4f}  acc={m_f['accuracy']:.4f}")

    # ------------------------------------------------------------------------
    # 跨种子 bagging
    # ------------------------------------------------------------------------
    bagged_rows = []

    def _bagged(method, probs_list):
        avg = np.mean(probs_list, axis=0)
        pred = np.array(LABELS)[avg.argmax(axis=1)]
        m = compute_metrics(y_str, pred, labels=LABELS)
        # 同时返回混淆矩阵, 方便 final print
        return ({
            "method": method, "seed": "bagged",
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        }, m)

    print("\n=== bagged across seeds ===")
    row_c, m_c_bag = _bagged(f"bagged_cascade_{mode}", seed_cascade_probs)
    row_f, m_f_bag = _bagged("bagged_flat_3class", seed_flat_probs)
    bagged_rows.extend([row_c, row_f])

    print(f"  bagged cascade ({mode}) bal_acc={row_c['balanced_accuracy']:.4f}  "
          f"macro_f1={row_c['macro_f1']:.4f}")
    print(f"  bagged flat 3-class     bal_acc={row_f['balanced_accuracy']:.4f}  "
          f"macro_f1={row_f['macro_f1']:.4f}")

    delta = row_c["balanced_accuracy"] - row_f["balanced_accuracy"]
    print(f"\n  >>> cascade vs flat  Δ balanced_accuracy = {delta:+.4f}")

    print("\n  Cascade confusion matrix (rows=true, cols=pred):")
    cm_df = pd.DataFrame(m_c_bag["confusion_matrix"], index=LABELS, columns=LABELS)
    print(cm_df.to_string())
    print("\n  Flat confusion matrix (rows=true, cols=pred):")
    cm_df_f = pd.DataFrame(m_f_bag["confusion_matrix"], index=LABELS, columns=LABELS)
    print(cm_df_f.to_string())

    result = pd.DataFrame(per_seed_rows + bagged_rows)
    result = result.sort_values("balanced_accuracy", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return result


# ----------------------------------------------------------------------------
# 柱状图: cascade vs flat, 5 种子散点
# ----------------------------------------------------------------------------
def _method_color(method):
    if method.startswith("bagged_cascade"):
        return "#c62828"
    if method.startswith("bagged_flat"):
        return "#4C78A8"
    if method.startswith("cascade"):
        return "#f08080"
    return "#a3c4e0"


def plot_methods(result_df, output_png, metric="balanced_accuracy"):
    bagged = result_df[result_df["seed"] == "bagged"].copy()
    per_seed = result_df[result_df["seed"] != "bagged"].copy()
    bagged = bagged.sort_values(metric, ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(bagged) + 2.2)), dpi=120)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(bagged))
    colors = [_method_color(m) for m in bagged["method"]]
    bars = ax.barh(y_pos, bagged[metric], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.6, zorder=3)

    xmax = float(bagged[metric].max())
    xmin = float(bagged[metric].min())
    pad = (xmax - xmin) * 0.04 if xmax > xmin else xmax * 0.01
    for bar, val in zip(bars, bagged[metric]):
        ax.text(val + pad * 0.25, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left",
                fontsize=11, color="#222222", zorder=4)

    # per-seed 散点
    def _target(m_seed):
        if m_seed.startswith("cascade_"):
            return f"bagged_{m_seed}"
        if m_seed == "flat_3class":
            return "bagged_flat_3class"
        return None

    method_to_y = dict(zip(bagged["method"], y_pos))
    for _, row in per_seed.iterrows():
        tgt = _target(row["method"])
        if tgt and tgt in method_to_y:
            ax.scatter(row[metric], method_to_y[tgt],
                       color="#444444", s=28, zorder=5, alpha=0.7,
                       edgecolor="white", linewidth=0.6)

    # 高亮 best
    best_idx = bagged[metric].idxmax()
    best_bar = bars[best_idx]
    best_bar.set_edgecolor("#1f1f1f")
    best_bar.set_linewidth(1.8)
    ax.text(bagged[metric].iloc[best_idx] + pad * 0.25,
            best_bar.get_y() + best_bar.get_height() / 2 + 0.32,
            "★ best", color="#c62828", fontsize=11, weight="bold",
            va="center", ha="left", zorder=6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bagged["method"], fontsize=11)
    ax.set_xlabel("Balanced Accuracy" if metric == "balanced_accuracy"
                  else metric.replace("_", " ").title(),
                  fontsize=12, color="#333333")
    left = max(0.0, xmin - (xmax - xmin) * 0.10 - 0.005)
    right = min(1.0, xmax + (xmax - xmin) * 0.20 + 0.01)
    ax.set_xlim(left, right)
    ax.set_title(f"Hierarchical (vasc -> mel/nv) vs Flat 3-class  "
                 f"(sorted by bagged {metric})",
                 fontsize=13, weight="bold", pad=12, color="#1f1f1f")

    handles = [
        Patch(facecolor="#c62828", label="Cascade (bagged)"),
        Patch(facecolor="#4C78A8", label="Flat 3-class (bagged)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444",
                   markersize=8, label="Per-seed score"),
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
def _parse_spec(value):
    """格式 'classifier:feature_set:k_features'."""
    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad spec '{value}'. "
                         f"Expected 'classifier:feature_set:k_features'.")
    return {
        "classifier": parts[0].strip(),
        "feature_set": parts[1].strip(),
        "k_features": parts[2].strip(),
    }


def _parse_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="v6: hierarchical (vasc -> mel/nv) classifier vs flat 3-class."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--stage1", default="xgb:contrast+abcd_v2+boundary:100",
        help="Stage 1 (vasc vs rest) spec. Default = v3 winner.",
    )
    parser.add_argument(
        "--stage2", default="xgb:all+boundary+melnv:100",
        help="Stage 2 (mel vs nv) spec. Default uses melnv module (designed for this).",
    )
    parser.add_argument(
        "--flat", default="xgb:contrast+abcd_v2+boundary:100",
        help="Flat 3-class baseline spec (用于对比).",
    )
    parser.add_argument("--mode", default="soft", choices=["soft", "hard"])
    parser.add_argument("--vasc_threshold", type=float, default=0.5,
                        help="Hard mode only. Default 0.5.")
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/ml_hierarchical_v6.csv")
    parser.add_argument("--output_png", default="outputs/figures/ml_hierarchical_v6.png")
    parser.add_argument("--metric", default="balanced_accuracy",
                        choices=["balanced_accuracy", "macro_f1", "accuracy"])
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in _parse_list(args.seeds)]

    result = run_hierarchical_v6(
        data_dir=Path(args.data_dir),
        stage1_spec=_parse_spec(args.stage1),
        stage2_spec=_parse_spec(args.stage2),
        flat_spec=_parse_spec(args.flat),
        mode=args.mode,
        vasc_threshold=args.vasc_threshold,
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

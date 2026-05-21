"""experiments/run_final_sweep_v8.py

v8: 最终超参数扫描. 项目收尾用.

特征已在 v7 固定:
    Stage 1 feature_set = contrast+abcd_v2+boundary
    Stage 2 feature_set = all+boundary+melnv+lbp_multi+gabor+subregion

本脚本扫描:
    1) Stage 2 XGB 超参数 (8 种, 围绕 v7 winner 局部搜索)
    2) Stage 2 k_features (5 个: 80, 100, 120, 160, all)
    3) Cascade 模式 (soft + hard at vasc_threshold=0.4)
    4) 5 种子 grouped CV + 概率 bagging

Stage 1 固定 (v3 winner: xgb n=150,d=2,lr=0.05,lambda=5, k=100).
Stage 1 OOF 概率每个 seed 算一次后复用, 节省 80% 时间.

输出:
- outputs/metrics/ml_final_sweep_v8.csv         全部 (config × mode) 结果
- outputs/figures/ml_final_sweep_v8_top.png     top-N 柱状图

命令行:
    python experiments/run_final_sweep_v8.py --data_dir data\Data_Proj2
"""

import argparse
import sys
import time
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
from src.features_gabor import extract_gabor_features
from src.features_lbp_multi import extract_lbp_multi_features
from src.features_melnv import extract_melnv_features
from src.features_shape import extract_shape_features
from src.features_subregion import extract_subregion_features
from src.features_texture import extract_texture_features
from src.preprocess import prepare_mask
from src.utils import base_id_from_image_id, ensure_dir

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError("xgboost required for v8. Run `pip install xgboost`.")


# ----------------------------------------------------------------------------
# 固定特征 (v7 winner)
# ----------------------------------------------------------------------------
STAGE1_FEATURE_SET = "contrast+abcd_v2+boundary"
STAGE2_FEATURE_SET = "all+boundary+melnv+lbp_multi+gabor+subregion"

# Stage 1 固定 (v3 winner)
STAGE1_XGB = dict(
    n_estimators=150, max_depth=2, learning_rate=0.05,
    subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0,
)
STAGE1_K = "100"

# Stage 2 XGB 超参网格 (围绕 v7 winner 局部搜索)
# tuple: (label, dict of XGB kwargs)
STAGE2_XGB_VARIANTS = [
    ("v7-default      ", dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
    ("d3-mid          ", dict(n_estimators=150, max_depth=3, learning_rate=0.05,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
    ("d3-more-trees   ", dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
    ("d2-more-trees   ", dict(n_estimators=300, max_depth=2, learning_rate=0.03,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
    ("less-reg        ", dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                              subsample=0.85, colsample_bytree=0.70, reg_lambda=2.0)),
    ("strong-reg      ", dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                              subsample=0.70, colsample_bytree=0.50, reg_lambda=10.0)),
    ("slow-learner    ", dict(n_estimators=400, max_depth=2, learning_rate=0.02,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
    ("deeper          ", dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0)),
]

# Stage 2 k_features
K_FEATURES = ["80", "100", "120", "160", "all"]

# Cascade modes
MODES = [
    ("soft         ", "soft", 0.5),     # threshold ignored for soft
    ("hard@0.4     ", "hard", 0.4),
]


# ----------------------------------------------------------------------------
# 特征模块 (与 v7 一致) - 仅用于特征提取/缓存
# ----------------------------------------------------------------------------
ATOMIC_EXTRACTORS = {
    "color":     extract_color_features,
    "shape":     extract_shape_features,
    "texture":   extract_texture_features,
    "contrast":  extract_contrast_features,
    "abcd_v2":   extract_abcd_v2_features,
    "boundary":  extract_boundary_features,
    "melnv":     extract_melnv_features,
    "lbp_multi": extract_lbp_multi_features,
    "gabor":     extract_gabor_features,
    "subregion": extract_subregion_features,
}
ALL_BUNDLE = ("color", "shape", "texture")


def resolve_modules(feature_set):
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


def extract_features_for_image(image, mask, feature_set):
    features = {}
    for name in resolve_modules(feature_set):
        features.update(ATOMIC_EXTRACTORS[name](image, mask))
    return features


def build_feature_table(data_dir, image_ids, feature_set, mask_mode):
    rows = []
    for image_id in tqdm(image_ids, desc=f"  features[{feature_set}]", leave=False):
        image = load_image(data_dir, image_id)
        mask = prepare_mask(load_mask(data_dir, image_id), mask_mode=mask_mode)
        row = {"image_id": str(image_id)}
        row.update(extract_features_for_image(image, mask, feature_set))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_table_cached(data_dir, labels, feature_set, mask_mode, cache_dir, use_cache):
    ensure_dir(cache_dir)
    safe = feature_set.replace("+", "_AND_")
    cache_path = cache_dir / f"features_v2_{safe}_{mask_mode}.csv"
    if use_cache and cache_path.exists():
        return pd.read_csv(cache_path)
    df = build_feature_table(data_dir, labels["image_id"].astype(str),
                             feature_set, mask_mode)
    df.to_csv(cache_path, index=False)
    return df


# ----------------------------------------------------------------------------
# 模型构造
# ----------------------------------------------------------------------------
def _build_xgb(xgb_kwargs, n_classes):
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    return XGBClassifier(
        **xgb_kwargs,
        objective=objective, eval_metric=eval_metric,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )


def _build_pipeline(xgb_kwargs, n_classes, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    steps.append(("clf", _build_xgb(xgb_kwargs, n_classes)))
    return Pipeline(steps)


def _make_splits(master_labels, n_splits, random_state):
    y = master_labels["dx"]
    groups = master_labels["image_id"].map(base_id_from_image_id)
    placeholder = pd.DataFrame(index=master_labels.index)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(placeholder, y, groups=groups))


def _oof_proba(pipe, X, y, splits, n_classes):
    oof = np.zeros((len(X), n_classes), dtype=np.float32)
    for tr, va in splits:
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr = y[tr]
        sw = compute_sample_weight("balanced", y_tr)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
        oof[va] = pipe.predict_proba(X_va)
    return oof


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def run_sweep_v8(data_dir, seeds, n_splits, mask_mode, output_csv,
                 output_per_seed_csv, use_cache=True):
    t_start = time.time()

    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError("label.csv required.")
    labels["image_id"] = labels["image_id"].astype(str)
    master_labels = labels.reset_index(drop=True)
    n = len(master_labels)
    y_str = master_labels["dx"].values
    LABEL_IDX = {label: i for i, label in enumerate(LABELS)}
    vasc_idx, mel_idx, nv_idx = LABEL_IDX["vasc"], LABEL_IDX["mel"], LABEL_IDX["nv"]

    n_configs = len(STAGE2_XGB_VARIANTS) * len(K_FEATURES) * len(MODES)
    print(f"\n>>> n_samples: {n}")
    print(f">>> Stage 1 (固定): xgb {STAGE1_XGB} on {STAGE1_FEATURE_SET}, k={STAGE1_K}")
    print(f">>> Stage 2 (固定 feature_set): {STAGE2_FEATURE_SET}")
    print(f">>> Stage 2 scan:")
    print(f"      XGB variants : {len(STAGE2_XGB_VARIANTS)}")
    print(f"      k_features   : {K_FEATURES}")
    print(f"      modes        : {[m[0].strip() for m in MODES]}")
    print(f">>> Total configs: {n_configs}")
    print(f">>> Seeds: {seeds}")
    print(f">>> Total CV runs: {n_configs * len(seeds)}\n")

    # 预加载特征
    cache_dir = Path("outputs/cache")
    print(">>> Pre-loading feature tables...")
    feature_cache = {}
    for fs in [STAGE1_FEATURE_SET, STAGE2_FEATURE_SET]:
        df = _feature_table_cached(data_dir, master_labels, fs, mask_mode,
                                   cache_dir, use_cache)
        df["image_id"] = df["image_id"].astype(str)
        merged = master_labels.merge(df, on="image_id", how="inner")
        assert len(merged) == n
        feature_cache[fs] = merged.drop(columns=["image_id", "dx"])
        print(f"    [{fs[:55]:<55s}] X shape = {feature_cache[fs].shape}")
    print()

    X_s1 = feature_cache[STAGE1_FEATURE_SET]
    X_s2 = feature_cache[STAGE2_FEATURE_SET]

    # config_probs[(xgb_label, k, mode_label)] = list of (n, 3) per seed
    config_probs = {}
    stage1_acc_per_seed = []

    for s_idx, seed in enumerate(seeds):
        print(f"--- seed {seed} ({s_idx + 1}/{len(seeds)}) ---")
        splits = _make_splits(master_labels, n_splits, seed)

        # ----- Stage 1 OOF (固定, 每个 seed 算一次) -----
        t = time.time()
        y_s1 = (y_str == "vasc").astype(int)
        s1_pipe = _build_pipeline(STAGE1_XGB, 2, STAGE1_K, X_s1.shape[1])
        oof_s1 = _oof_proba(s1_pipe, X_s1, y_s1, splits, n_classes=2)
        p_vasc = oof_s1[:, 1]
        s1_acc = float(((p_vasc > 0.5).astype(int) == y_s1).mean())
        stage1_acc_per_seed.append(s1_acc)
        print(f"  Stage 1 OOF acc (vasc vs rest): {s1_acc:.4f}  "
              f"[{time.time() - t:.1f}s]")

        # ----- Sweep Stage 2 -----
        mn_idx = np.where(y_str != "vasc")[0]
        for xgb_label, xgb_kwargs in STAGE2_XGB_VARIANTS:
            for k in K_FEATURES:
                t = time.time()
                # Stage 2 OOF (binary mel vs nv, train on train ∩ {mel, nv})
                s2_oof = np.zeros((n, 2), dtype=np.float32)
                for tr, va in splits:
                    mn_tr = tr[np.isin(tr, mn_idx)]
                    y_tr_s2 = (y_str[mn_tr] == "nv").astype(int)
                    pipe = _build_pipeline(xgb_kwargs, 2, k, X_s2.shape[1])
                    sw = compute_sample_weight("balanced", y_tr_s2)
                    pipe.fit(X_s2.iloc[mn_tr], y_tr_s2,
                             clf__sample_weight=sw)
                    s2_oof[va] = pipe.predict_proba(X_s2.iloc[va])
                p_mel = s2_oof[:, 0]
                p_nv = s2_oof[:, 1]

                # 组合成 3 类概率, soft / hard 两种模式
                for mode_label, mode, thresh in MODES:
                    casc = np.zeros((n, 3), dtype=np.float32)
                    if mode == "soft":
                        casc[:, vasc_idx] = p_vasc
                        casc[:, mel_idx] = (1 - p_vasc) * p_mel
                        casc[:, nv_idx] = (1 - p_vasc) * p_nv
                    else:  # hard
                        is_vasc = p_vasc > thresh
                        casc[:, vasc_idx] = np.where(is_vasc, 1.0, 0.0)
                        casc[:, mel_idx] = np.where(is_vasc, 0.0, p_mel)
                        casc[:, nv_idx] = np.where(is_vasc, 0.0, p_nv)

                    key = (xgb_label.strip(), k, mode_label.strip())
                    config_probs.setdefault(key, []).append(casc)

                # 打印本轮 (xgb, k) 的 soft 模式分数监控
                soft_casc = config_probs[(xgb_label.strip(), k, "soft")][s_idx]
                soft_pred = np.array(LABELS)[soft_casc.argmax(axis=1)]
                ba = compute_metrics(y_str, soft_pred, labels=LABELS)["balanced_accuracy"]
                print(f"  [{xgb_label} k={k:<3s}]  soft bal_acc={ba:.4f}  "
                      f"[{time.time() - t:.1f}s]")

    # ------------------------------------------------------------------------
    # 聚合 bagged 结果 + 各 seed 单独结果 (long format)
    # ------------------------------------------------------------------------
    print("\n>>> Aggregating bagged + per-seed results...\n")
    rows = []
    per_seed_rows = []

    def _variant_kwarg(label, key):
        return next(v[1][key] for v in STAGE2_XGB_VARIANTS
                    if v[0].strip() == label)

    for (xgb_label, k, mode_label), probs_list in config_probs.items():
        # 各 seed 单独的 OOF 指标
        per_seed_bal = []
        per_seed_f1 = []
        per_seed_acc = []
        for seed_idx, probs in enumerate(probs_list):
            seed_val = seeds[seed_idx]
            p_pred = np.array(LABELS)[probs.argmax(axis=1)]
            m = compute_metrics(y_str, p_pred, labels=LABELS)
            per_seed_bal.append(m["balanced_accuracy"])
            per_seed_f1.append(m["macro_f1"])
            per_seed_acc.append(m["accuracy"])
            per_seed_rows.append({
                "stage2_xgb_variant": xgb_label,
                "stage2_k_features": k,
                "cascade_mode": mode_label,
                "seed": seed_val,
                "balanced_accuracy": m["balanced_accuracy"],
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
            })
        per_seed_bal = np.asarray(per_seed_bal)

        # bagged: 概率平均 -> argmax
        avg = np.mean(probs_list, axis=0)
        pred = np.array(LABELS)[avg.argmax(axis=1)]
        m_bag = compute_metrics(y_str, pred, labels=LABELS)

        rows.append({
            "stage2_xgb_variant": xgb_label,
            "stage2_k_features": k,
            "cascade_mode": mode_label,
            "stage2_n_estimators": _variant_kwarg(xgb_label, "n_estimators"),
            "stage2_max_depth": _variant_kwarg(xgb_label, "max_depth"),
            "stage2_learning_rate": _variant_kwarg(xgb_label, "learning_rate"),
            "stage2_reg_lambda": _variant_kwarg(xgb_label, "reg_lambda"),
            "bagged_balanced_accuracy": m_bag["balanced_accuracy"],
            "bagged_macro_f1": m_bag["macro_f1"],
            "bagged_accuracy": m_bag["accuracy"],
            "per_seed_bal_acc_mean": float(per_seed_bal.mean()),
            "per_seed_bal_acc_std": float(per_seed_bal.std()),
            "per_seed_bal_acc_min": float(per_seed_bal.min()),
            "per_seed_bal_acc_max": float(per_seed_bal.max()),
        })

    result = pd.DataFrame(rows).sort_values("bagged_balanced_accuracy", ascending=False)
    ensure_dir(Path(output_csv).parent)
    result.to_csv(output_csv, index=False)
    print(f"Saved {len(result)} bagged rows to {output_csv}")

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(
        ["seed", "balanced_accuracy"], ascending=[True, False]
    )
    ensure_dir(Path(output_per_seed_csv).parent)
    per_seed_df.to_csv(output_per_seed_csv, index=False)
    print(f"Saved {len(per_seed_df)} per-seed rows to {output_per_seed_csv}")

    print(f"\n>>> Average Stage 1 vasc detection accuracy across seeds: "
          f"{np.mean(stage1_acc_per_seed):.4f}")
    print(f">>> Total wall time: {(time.time() - t_start) / 60:.1f} min")

    return result, per_seed_df


# ----------------------------------------------------------------------------
# 柱状图: top-N configs (bagged 用 5 位小数; single-seed 用 4 位)
# ----------------------------------------------------------------------------
def plot_top_configs(result_df, output_png, top_n=30,
                     metric_col="bagged_balanced_accuracy",
                     std_col=None,
                     precision=4,
                     title=None,
                     xlabel=None,
                     x_left=0.30,
                     x_right_pad=0.06):
    top = result_df.sort_values(metric_col, ascending=False).head(top_n).reset_index(drop=True)

    # 右对齐变体名 (宽度 14), 让所有 'k=' 起点在同一列, 避免长度差异造成视觉不齐
    top["label"] = top.apply(
        lambda r: f"{r['stage2_xgb_variant']:>14s}  k={r['stage2_k_features']:>3s}  "
                  f"{r['cascade_mode']}",
        axis=1,
    )

    fig_h = max(5.0, 0.34 * len(top) + 1.6)
    fig, ax = plt.subplots(figsize=(13, fig_h), dpi=120)
    fig.patch.set_facecolor("white")

    y_pos = np.arange(len(top))[::-1]
    colors = ["#c62828" if m == "soft" else "#4C78A8" for m in top["cascade_mode"]]
    bars = ax.barh(y_pos, top[metric_col], color=colors,
                   edgecolor="white", linewidth=0.6, height=0.72, zorder=3)

    xmax = float(top[metric_col].max())

    # 数值标注: 用相对值的小固定偏移, 避免大范围 x 轴下偏移爆炸
    has_std = std_col is not None and std_col in top.columns
    label_offset = 0.003   # ~0.3% of full [0,1] axis, 紧贴柱端
    for bar, val, std in zip(bars, top[metric_col],
                              top[std_col] if has_std else [0.0] * len(top)):
        text = f"{val:.{precision}f}"
        if has_std:
            text = f"{text}  (±{std:.3f})"
        ax.text(val + label_offset, bar.get_y() + bar.get_height() / 2,
                text, va="center", ha="left",
                fontsize=9, color="#222222", zorder=4)

    bars[0].set_edgecolor("#1f1f1f")
    bars[0].set_linewidth(1.8)
    ax.text(top[metric_col].iloc[0] + label_offset,
            bars[0].get_y() + bars[0].get_height() / 2 + 0.55,
            "★ best", color="#c62828", fontsize=10, weight="bold",
            va="center", ha="left", zorder=5)

    ax.set_yticks(y_pos)
    # 显式 DejaVu Sans Mono (matplotlib 自带, 保证真等宽)
    ax.set_yticklabels(top["label"], fontsize=9, fontname="DejaVu Sans Mono")
    ax.set_xlabel(xlabel or "Balanced Accuracy",
                  fontsize=11, color="#333333")
    # 固定 x 轴起点 + 紧凑右边距 (绝对值, 不再随 range 缩放)
    ax.set_xlim(x_left, min(1.0, xmax + x_right_pad))
    ax.set_title(title or f"v8 Final Sweep: top-{top_n}",
                 fontsize=12, weight="bold", pad=12, color="#1f1f1f")

    handles = [
        Patch(facecolor="#c62828", label="Cascade soft"),
        Patch(facecolor="#4C78A8", label="Cascade hard @ thresh=0.4"),
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
        description="v8 final sweep: Stage 2 hyperparameters × k × cascade mode."
    )
    parser.add_argument("--data_dir", default=None,
                        help="只有非 --plot_only 时必填.")
    parser.add_argument("--seeds", default="42,127,2024,3407,520")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--output_csv", default="outputs/metrics/ml_final_sweep_v8.csv")
    parser.add_argument("--output_per_seed_csv",
                        default="outputs/metrics/ml_final_sweep_v8_per_seed.csv")
    parser.add_argument("--output_png", default="outputs/figures/ml_final_sweep_v8_top.png")
    parser.add_argument("--output_png_single_seed",
                        default="outputs/figures/ml_final_sweep_v8_top_seed{seed}.png",
                        help="使用 {seed} 占位符自动填入种子号.")
    parser.add_argument("--seed_for_chart", type=int, default=127,
                        help="额外为这个种子单独画一张排序柱状图.")
    parser.add_argument("--top_n", type=int, default=30,
                        help="柱状图显示前 N 个配置 (默认 30).")
    parser.add_argument("--x_left", type=float, default=0.30,
                        help="柱状图 x 轴起点 (默认 0.30).")
    parser.add_argument("--x_right_pad", type=float, default=0.06,
                        help="柱状图 x 轴右边绝对留白 (默认 0.06).")
    parser.add_argument("--plot_only", action="store_true",
                        help="跳过训练, 直接从已有 CSV 重新生成柱状图.")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    seeds = [int(s) for s in _parse_list(args.seeds)]

    if args.plot_only:
        # 从已有 CSV 直接读 - 不跑训练
        bagged_path = Path(args.output_csv)
        per_seed_path = Path(args.output_per_seed_csv)
        if not bagged_path.exists():
            raise FileNotFoundError(f"--plot_only requires {bagged_path}. "
                                    f"Run without --plot_only first.")
        if not per_seed_path.exists():
            raise FileNotFoundError(f"--plot_only requires {per_seed_path}. "
                                    f"Run without --plot_only first.")
        print(f">>> --plot_only: loading from\n    {bagged_path}\n    {per_seed_path}")
        result = pd.read_csv(bagged_path)
        per_seed_df = pd.read_csv(per_seed_path)
    else:
        if args.data_dir is None:
            parser.error("--data_dir is required (unless --plot_only).")
        result, per_seed_df = run_sweep_v8(
            data_dir=Path(args.data_dir),
            seeds=seeds,
            n_splits=args.n_splits,
            mask_mode=args.mask_mode,
            output_csv=Path(args.output_csv),
            output_per_seed_csv=Path(args.output_per_seed_csv),
            use_cache=not args.no_cache,
        )

    print(f"\n=== Top {args.top_n} configs by bagged_balanced_accuracy ===")
    cols = ["stage2_xgb_variant", "stage2_k_features", "cascade_mode",
            "bagged_balanced_accuracy", "bagged_macro_f1",
            "per_seed_bal_acc_mean", "per_seed_bal_acc_std"]
    print(result[cols].head(args.top_n).to_string(index=False))

    # 图 1: 多种子 bagged - 5 位小数
    plot_top_configs(
        result, Path(args.output_png), top_n=args.top_n,
        metric_col="bagged_balanced_accuracy",
        std_col="per_seed_bal_acc_std",
        precision=5,
        title=f"v8 Final Sweep (bagged across {len(seeds)} seeds): top-{args.top_n}",
        xlabel=f"Bagged Balanced Accuracy  (±std across {len(seeds)} seeds)",
        x_left=args.x_left,
        x_right_pad=args.x_right_pad,
    )

    # 图 2: 指定 single seed 排序
    if args.seed_for_chart in seeds:
        seed_df = per_seed_df[per_seed_df["seed"] == args.seed_for_chart].copy()
        if len(seed_df) > 0:
            single_path = Path(str(args.output_png_single_seed).format(
                seed=args.seed_for_chart))
            print(f"\n=== Top {args.top_n} configs at seed={args.seed_for_chart} ===")
            single_cols = ["stage2_xgb_variant", "stage2_k_features",
                           "cascade_mode", "balanced_accuracy", "macro_f1"]
            print(seed_df[single_cols].head(args.top_n).to_string(index=False))
            plot_top_configs(
                seed_df, single_path, top_n=args.top_n,
                metric_col="balanced_accuracy",
                std_col=None,
                precision=4,
                title=f"v8 Single-seed ranking at seed={args.seed_for_chart}: top-{args.top_n}",
                xlabel=f"Balanced Accuracy (seed={args.seed_for_chart}, single 5-fold OOF)",
                x_left=args.x_left,
                x_right_pad=args.x_right_pad,
            )
    else:
        print(f"\n[warn] seed_for_chart={args.seed_for_chart} not in --seeds, "
              f"skipping single-seed chart.")


if __name__ == "__main__":
    main()

"""src/train_ml_v2.py

train_ml.py v2: 用 v8 找到的最佳 cascade 配置, 在全量数据上训练两个 Stage,
保存为单个 joblib bundle, 后续可被 run_v2.py 加载做推理.

固定配置:
    Stage 1 (vasc vs rest):
        feature_set = contrast+abcd_v2+boundary
        k = 100
        XGB: n=150, d=2, lr=0.05, ss=0.8, cs=0.6, lambda=5  (v3 winner)

    Stage 2 (mel vs nv):
        feature_set = all+boundary+melnv+lbp_multi+gabor+subregion
        k = 100
        XGB: n=300, d=2, lr=0.03, ss=0.8, cs=0.6, lambda=5  (v8 d2-more-trees)

    Cascade mode = soft  (P(vasc), P(mel)=P(not_vasc)*P(mel|2nd),
                           P(nv)=P(not_vasc)*P(nv|2nd))

命令行:
    python -m src.train_ml_v2 --data_dir data\Data_Proj2

输出: outputs/models/cascade_v2_d2moretrees_k100_soft.joblib
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm

warnings.filterwarnings("ignore",
                        message=".*X does not have valid feature names.*",
                        category=UserWarning)

from .config import LABELS, DEFAULT_MODEL_DIR
from .dataset import load_image, load_labels, load_mask
from .features_abcd import extract_abcd_v2_features
from .features_boundary import extract_boundary_features
from .features_color import extract_color_features
from .features_contrast import extract_contrast_features
from .features_gabor import extract_gabor_features
from .features_lbp_multi import extract_lbp_multi_features
from .features_melnv import extract_melnv_features
from .features_shape import extract_shape_features
from .features_subregion import extract_subregion_features
from .features_texture import extract_texture_features
from .preprocess import prepare_mask
from .utils import ensure_dir

from xgboost import XGBClassifier


# ----------------------------------------------------------------------------
# 固定配置
# ----------------------------------------------------------------------------
STAGE1 = {
    "feature_set": "contrast+abcd_v2+boundary",
    "k_features": "100",
    "xgb_kwargs": dict(
        n_estimators=150, max_depth=2, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0,
    ),
}

STAGE2 = {
    "feature_set": "all+boundary+melnv+lbp_multi+gabor+subregion",
    "k_features": "100",
    "xgb_kwargs": dict(
        n_estimators=300, max_depth=2, learning_rate=0.03,
        subsample=0.80, colsample_bytree=0.60, reg_lambda=5.0,
    ),
}

CASCADE_MODE = "soft"
VASC_THRESHOLD = 0.5    # 仅 hard 模式使用, 保留以便兼容

DEFAULT_BUNDLE_PATH = DEFAULT_MODEL_DIR / "cascade_v2_d2moretrees_k100_soft.joblib"


# ----------------------------------------------------------------------------
# 特征模块 (与 v7/v8 一致, 复制以保持 src/ 独立可用)
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
            raise ValueError(f"Unknown feature block '{part}'.")
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


def build_feature_table(data_dir, image_ids, feature_set, mask_mode, desc=None):
    rows = []
    for image_id in tqdm(image_ids, desc=desc or f"features[{feature_set[:30]}]"):
        image = load_image(data_dir, image_id)
        mask = prepare_mask(load_mask(data_dir, image_id), mask_mode=mask_mode)
        row = {"image_id": str(image_id)}
        row.update(extract_features_for_image(image, mask, feature_set))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_table_cached(data_dir, image_ids, feature_set, mask_mode,
                          cache_dir, use_cache):
    """复用 v2/v3/v8 的 features_v2_*.csv 缓存."""
    if use_cache:
        ensure_dir(cache_dir)
        safe = feature_set.replace("+", "_AND_")
        cache_path = cache_dir / f"features_v2_{safe}_{mask_mode}.csv"
        if cache_path.exists():
            print(f"  [cache hit] {cache_path.name}")
            return pd.read_csv(cache_path)
    df = build_feature_table(data_dir, image_ids, feature_set, mask_mode)
    if use_cache:
        df.to_csv(cache_path, index=False)
    return df


# ----------------------------------------------------------------------------
# Pipeline 构造
# ----------------------------------------------------------------------------
def _build_pipeline(xgb_kwargs, n_classes, k_features, n_features):
    steps = [("scaler", StandardScaler())]
    if k_features != "all":
        k_value = min(int(k_features), n_features)
        if k_value < n_features:
            steps.append(("select", SelectKBest(f_classif, k=k_value)))
    objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if n_classes == 2 else "mlogloss"
    clf = XGBClassifier(
        **xgb_kwargs,
        objective=objective, eval_metric=eval_metric,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )
    steps.append(("clf", clf))
    return Pipeline(steps)


# ----------------------------------------------------------------------------
# 训练主流程
# ----------------------------------------------------------------------------
def train_cascade(data_dir, mask_mode="raw", model_path=None,
                  use_cache=True, cache_dir=None):
    labels = load_labels(data_dir)
    if labels is None:
        raise FileNotFoundError(f"label.csv required in {data_dir}.")
    labels["image_id"] = labels["image_id"].astype(str)
    master = labels.reset_index(drop=True)
    y_str = master["dx"].values
    n_total = len(master)

    cache_dir = cache_dir or Path("outputs/cache")

    print(f"\n{'=' * 70}")
    print(f"Training cascade v2 on {n_total} samples (mask_mode={mask_mode})")
    print(f"{'=' * 70}")

    # ===== Stage 1: vasc vs not_vasc =====
    print(f"\n[Stage 1] feature_set = {STAGE1['feature_set']}")
    X1_df = _feature_table_cached(data_dir, master["image_id"],
                                  STAGE1["feature_set"], mask_mode,
                                  cache_dir, use_cache)
    X1_df["image_id"] = X1_df["image_id"].astype(str)
    merged1 = master.merge(X1_df, on="image_id", how="inner")
    assert len(merged1) == n_total, "Stage 1 feature table missing samples"
    X1 = merged1.drop(columns=["image_id", "dx"])
    y1 = (y_str == "vasc").astype(int)

    pipe1 = _build_pipeline(STAGE1["xgb_kwargs"], n_classes=2,
                            k_features=STAGE1["k_features"],
                            n_features=X1.shape[1])
    sw1 = compute_sample_weight("balanced", y1)
    pipe1.fit(X1, y1, clf__sample_weight=sw1)
    train_p_vasc = pipe1.predict_proba(X1)[:, 1]
    train_s1_acc = float(((train_p_vasc > 0.5).astype(int) == y1).mean())
    print(f"  Stage 1 fit on {len(X1)} samples, {X1.shape[1]} features")
    print(f"  Stage 1 train accuracy (vasc vs rest): {train_s1_acc:.4f}")

    # ===== Stage 2: mel vs nv (仅 train 在 mel+nv) =====
    print(f"\n[Stage 2] feature_set = {STAGE2['feature_set']}")
    X2_df = _feature_table_cached(data_dir, master["image_id"],
                                  STAGE2["feature_set"], mask_mode,
                                  cache_dir, use_cache)
    X2_df["image_id"] = X2_df["image_id"].astype(str)
    merged2 = master.merge(X2_df, on="image_id", how="inner")
    assert len(merged2) == n_total, "Stage 2 feature table missing samples"
    X2_full = merged2.drop(columns=["image_id", "dx"])

    mn_mask = y_str != "vasc"
    X2_mn = X2_full.loc[mn_mask].reset_index(drop=True)
    y2_mn = (y_str[mn_mask] == "nv").astype(int)   # mel=0, nv=1

    pipe2 = _build_pipeline(STAGE2["xgb_kwargs"], n_classes=2,
                            k_features=STAGE2["k_features"],
                            n_features=X2_mn.shape[1])
    sw2 = compute_sample_weight("balanced", y2_mn)
    pipe2.fit(X2_mn, y2_mn, clf__sample_weight=sw2)
    train_s2_acc = float(((pipe2.predict_proba(X2_mn)[:, 1] > 0.5).astype(int)
                          == y2_mn).mean())
    print(f"  Stage 2 fit on {len(X2_mn)} mel+nv samples, {X2_mn.shape[1]} features")
    print(f"  Stage 2 train accuracy (mel vs nv): {train_s2_acc:.4f}")

    # ===== 打包 bundle =====
    bundle = {
        "version": "train_ml_v2",
        "labels": list(LABELS),
        "mask_mode": mask_mode,
        "cascade_mode": CASCADE_MODE,
        "vasc_threshold": VASC_THRESHOLD,
        "stage1": {
            "model": pipe1,
            "feature_set": STAGE1["feature_set"],
            "feature_columns": list(X1.columns),
            "k_features": STAGE1["k_features"],
            "xgb_kwargs": STAGE1["xgb_kwargs"],
        },
        "stage2": {
            "model": pipe2,
            "feature_set": STAGE2["feature_set"],
            "feature_columns": list(X2_full.columns),
            "k_features": STAGE2["k_features"],
            "xgb_kwargs": STAGE2["xgb_kwargs"],
        },
        "metadata": {
            "trained_on_n_samples": int(n_total),
            "trained_on_mn_samples": int(mn_mask.sum()),
            "train_stage1_accuracy": train_s1_acc,
            "train_stage2_accuracy": train_s2_acc,
        },
    }

    model_path = Path(model_path) if model_path else DEFAULT_BUNDLE_PATH
    ensure_dir(model_path.parent)
    joblib.dump(bundle, model_path)
    print(f"\nSaved cascade bundle to {model_path}")
    print(f"  bundle keys: stage1, stage2, cascade_mode, labels, mask_mode, metadata")
    return bundle


def main():
    parser = argparse.ArgumentParser(
        description="Train final cascade v2 model (d2-more-trees / k=100 / soft)."
    )
    parser.add_argument("--data_dir", required=True,
                        help="目录里应包含 image/, mask/, label.csv")
    parser.add_argument("--mask_mode", default="raw", choices=["raw", "clean"])
    parser.add_argument("--model_path", default=None,
                        help=f"默认 {DEFAULT_BUNDLE_PATH}")
    parser.add_argument("--no_cache", action="store_true",
                        help="不读 outputs/cache/ 里的特征 CSV, 重新提取")
    args = parser.parse_args()

    train_cascade(
        data_dir=Path(args.data_dir),
        mask_mode=args.mask_mode,
        model_path=Path(args.model_path) if args.model_path else None,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()

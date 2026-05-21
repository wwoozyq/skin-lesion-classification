"""run_v2.py

v2 推理脚本: 加载 train_ml_v2.py 训出来的 cascade bundle, 对同结构的输入数据
(image/, mask/, [可选 label.csv]) 做特征提取 + cascade 推理, 输出预测 CSV.
如果输入目录包含 label.csv, 额外打印准确率 / macro-F1 / 平衡准确率 / 混淆矩阵.

输入目录结构 (与训练数据一致):
    input_dir/
      image/
        1.jpg
        2.jpg
        ...
      mask/
        mask_1.jpg
        mask_2.jpg
        ...
      label.csv               (可选, 有则跑评估)

命令行示例:
    # 不带 label, 仅推理
    python run_v2.py --input_dir data\test_set

    # 带 label, 推理 + 评估
    python run_v2.py --input_dir data\Data_Proj2 --output_csv output_v2.csv
"""

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore",
                        message=".*X does not have valid feature names.*",
                        category=UserWarning)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.dataset import list_image_ids, load_image, load_labels, load_mask
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


# ----------------------------------------------------------------------------
# 特征 (与 train_ml_v2.py 共用同一组 10 个原子模块)
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


# ----------------------------------------------------------------------------
# Cascade 推理
# ----------------------------------------------------------------------------
def predict_cascade(bundle, X1_df, X2_df, image_ids):
    """对齐列顺序 -> Stage 1/2 概率 -> soft/hard 组合 -> 3 类预测."""
    stage1 = bundle["stage1"]
    stage2 = bundle["stage2"]

    # 列顺序与训练时一致 (KeyError 会提示缺哪列)
    missing1 = set(stage1["feature_columns"]) - set(X1_df.columns)
    missing2 = set(stage2["feature_columns"]) - set(X2_df.columns)
    if missing1:
        raise KeyError(f"Stage 1 features missing in input: {sorted(missing1)[:5]}...")
    if missing2:
        raise KeyError(f"Stage 2 features missing in input: {sorted(missing2)[:5]}...")

    X1 = X1_df[stage1["feature_columns"]]
    X2 = X2_df[stage2["feature_columns"]]

    p_vasc = stage1["model"].predict_proba(X1)[:, 1]
    proba2 = stage2["model"].predict_proba(X2)
    p_mel_given_not_vasc = proba2[:, 0]
    p_nv_given_not_vasc = proba2[:, 1]

    n = len(image_ids)
    final_probs = np.zeros((n, 3), dtype=np.float32)
    label_to_idx = {l: i for i, l in enumerate(bundle["labels"])}
    vasc_idx = label_to_idx["vasc"]
    mel_idx = label_to_idx["mel"]
    nv_idx = label_to_idx["nv"]

    if bundle["cascade_mode"] == "soft":
        final_probs[:, vasc_idx] = p_vasc
        final_probs[:, mel_idx] = (1 - p_vasc) * p_mel_given_not_vasc
        final_probs[:, nv_idx] = (1 - p_vasc) * p_nv_given_not_vasc
    else:  # hard
        thresh = bundle.get("vasc_threshold", 0.5)
        is_vasc = p_vasc > thresh
        final_probs[:, vasc_idx] = np.where(is_vasc, 1.0, 0.0)
        final_probs[:, mel_idx] = np.where(is_vasc, 0.0, p_mel_given_not_vasc)
        final_probs[:, nv_idx] = np.where(is_vasc, 0.0, p_nv_given_not_vasc)

    pred_labels = np.array(bundle["labels"])[final_probs.argmax(axis=1)]
    pred_df = pd.DataFrame({
        "image_id": image_ids,
        "dx": pred_labels,
        "p_mel":  final_probs[:, mel_idx],
        "p_nv":   final_probs[:, nv_idx],
        "p_vasc": final_probs[:, vasc_idx],
    })
    return pred_df, final_probs


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="v2 inference: load cascade bundle, predict, optionally evaluate."
    )
    parser.add_argument("--input_dir", required=True,
                        help="目录应有 image/, mask/, 可选 label.csv (有则评估)")
    parser.add_argument("--model_path",
                        default="outputs/models/cascade_v2_d2moretrees_k100_soft.joblib")
    parser.add_argument("--output_csv", default="output_v2.csv")
    args = parser.parse_args()

    bundle_path = Path(args.model_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model not found: {bundle_path}\n"
            f"先训练: python -m src.train_ml_v2 --data_dir <your_data_dir>"
        )
    bundle = joblib.load(bundle_path)
    print(f"Loaded bundle: {bundle.get('version', '?')}")
    print(f"  trained_on_n_samples: {bundle['metadata']['trained_on_n_samples']}")
    print(f"  cascade_mode: {bundle['cascade_mode']}")
    print(f"  Stage 1: {bundle['stage1']['feature_set']}, k={bundle['stage1']['k_features']}")
    print(f"  Stage 2: {bundle['stage2']['feature_set']}, k={bundle['stage2']['k_features']}")
    print()

    input_dir = Path(args.input_dir)
    labels = load_labels(input_dir)
    if labels is not None:
        labels["image_id"] = labels["image_id"].astype(str)
        image_ids = labels["image_id"].tolist()
        has_labels = True
        print(f"Found label.csv with {len(image_ids)} samples -> will evaluate.")
    else:
        image_ids = list_image_ids(input_dir)
        has_labels = False
        print(f"No label.csv -> {len(image_ids)} images from image/ dir, predict-only.")
    print()

    mask_mode = bundle["mask_mode"]

    # 提两组特征 (Stage 1 / Stage 2 不同 feature_set, 各跑一遍 ~1-3 分钟)
    print(f"[Stage 1] Extracting features [{bundle['stage1']['feature_set']}]...")
    X1_df = build_feature_table(input_dir, image_ids,
                                 bundle["stage1"]["feature_set"], mask_mode)

    print(f"\n[Stage 2] Extracting features [{bundle['stage2']['feature_set']}]...")
    X2_df = build_feature_table(input_dir, image_ids,
                                 bundle["stage2"]["feature_set"], mask_mode)

    # 推理
    pred_df, _ = predict_cascade(bundle, X1_df, X2_df, image_ids)

    # 保存 (主输出: image_id, dx; 概率列保留)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")
    print(f"  Predicted label distribution:")
    print(pred_df["dx"].value_counts().to_string())

    # 评估 (若有 label.csv)
    if has_labels:
        merged = labels.merge(pred_df[["image_id", "dx"]],
                              on="image_id", suffixes=("_true", "_pred"))
        if len(merged) != len(labels):
            print(f"\n[warn] {len(labels) - len(merged)} samples in label.csv "
                  f"have no prediction (image missing?). Eval on {len(merged)} samples.")

        y_true = merged["dx_true"].values
        y_pred = merged["dx_pred"].values
        metrics = compute_metrics(y_true, y_pred, labels=bundle["labels"])

        print(f"\n{'=' * 50}")
        print(f"Evaluation on {len(merged)} samples")
        print(f"{'=' * 50}")
        print(f"  Accuracy          : {metrics['accuracy']:.4f}")
        print(f"  Macro F1          : {metrics['macro_f1']:.4f}")
        print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")

        print(f"\nClassification report:")
        print(metrics["classification_report"])

        print(f"Confusion matrix (rows=true, cols=pred):")
        cm_df = pd.DataFrame(metrics["confusion_matrix"],
                             index=bundle["labels"],
                             columns=bundle["labels"])
        print(cm_df.to_string())
        print()


if __name__ == "__main__":
    main()

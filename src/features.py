import pandas as pd
from tqdm import tqdm

from .dataset import list_image_ids, load_image, load_mask
from .features_abcd import extract_abcd_v2_features
from .features_abcd_grouped import extract_abcd_grouped_features
from .features_boundary import extract_boundary_features
from .features_color import extract_color_features
from .features_contrast import extract_contrast_features
from .features_melnv import extract_melnv_features
from .features_shape import extract_shape_features
from .features_texture import extract_texture_features
from .preprocess import prepare_mask


def extract_features_for_image(image, mask, feature_set="all"):
    features = {}
    baseline_sets = {
        "all",
        "all_contrast",
        "all_abcd_v2",
        "all_boundary",
        "all_melnv",
        "all_boundary_melnv",
        "all_abcd_v2_boundary",
        "all_abcd_grouped",
        "final",
        "final_melnv",
        "final_abcd_grouped",
    }
    if feature_set in baseline_sets | {"color"}:
        features.update(extract_color_features(image, mask))
    if feature_set in baseline_sets | {"shape"}:
        features.update(extract_shape_features(image, mask))
    if feature_set in baseline_sets | {"texture"}:
        features.update(extract_texture_features(image, mask))
    if feature_set in {"contrast", "all_contrast", "final", "final_melnv", "final_abcd_grouped"}:
        features.update(extract_contrast_features(image, mask))
    if feature_set in {"abcd_v2", "all_abcd_v2", "all_abcd_v2_boundary", "final", "final_melnv", "final_abcd_grouped"}:
        features.update(extract_abcd_v2_features(image, mask))
    if feature_set in {"boundary", "all_boundary", "all_boundary_melnv", "all_abcd_v2_boundary", "final", "final_melnv", "final_abcd_grouped"}:
        features.update(extract_boundary_features(image, mask))
    if feature_set in {"abcd_grouped", "all_abcd_grouped", "final_abcd_grouped"}:
        features.update(extract_abcd_grouped_features(image, mask))
    if feature_set in {"melnv", "all_melnv", "all_boundary_melnv", "final_melnv"}:
        features.update(extract_melnv_features(image, mask))
    return features


def build_feature_table(data_dir, image_ids=None, feature_set="all", mask_mode="raw"):
    if image_ids is None:
        image_ids = list_image_ids(data_dir)

    rows = []
    for image_id in tqdm(image_ids, desc=f"Extracting {feature_set} features"):
        image = load_image(data_dir, image_id)
        mask = prepare_mask(load_mask(data_dir, image_id), mask_mode=mask_mode)
        row = {"image_id": str(image_id)}
        row.update(extract_features_for_image(image, mask, feature_set=feature_set))
        rows.append(row)
    return pd.DataFrame(rows)

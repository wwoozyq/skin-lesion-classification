import pandas as pd
from tqdm import tqdm

from .dataset import list_image_ids, load_image, load_mask
from .features_abcd import extract_abcd_features, extract_abcd_v2_features
from .features_boundary import extract_boundary_features
from .features_color import extract_color_features
from .features_shape import extract_shape_features
from .features_texture import extract_texture_features


def extract_features_for_image(image, mask, feature_set="all"):
    features = {}
    baseline_sets = {"all", "all_abcd", "all_abcd_v2", "all_boundary", "all_abcd_v2_boundary"}
    if feature_set in baseline_sets | {"color"}:
        features.update(extract_color_features(image, mask))
    if feature_set in baseline_sets | {"shape"}:
        features.update(extract_shape_features(image, mask))
    if feature_set in baseline_sets | {"texture"}:
        features.update(extract_texture_features(image, mask))
    if feature_set in {"all_abcd", "abcd"}:
        features.update(extract_abcd_features(image, mask))
    if feature_set in {"all_abcd_v2", "all_abcd_v2_boundary", "abcd_v2"}:
        features.update(extract_abcd_v2_features(image, mask))
    if feature_set in {"all_boundary", "all_abcd_v2_boundary", "boundary"}:
        features.update(extract_boundary_features(image, mask))
    return features


def build_feature_table(data_dir, image_ids=None, feature_set="all"):
    if image_ids is None:
        image_ids = list_image_ids(data_dir)

    rows = []
    for image_id in tqdm(image_ids, desc=f"Extracting {feature_set} features"):
        image = load_image(data_dir, image_id)
        mask = load_mask(data_dir, image_id)
        row = {"image_id": str(image_id)}
        row.update(extract_features_for_image(image, mask, feature_set=feature_set))
        rows.append(row)
    return pd.DataFrame(rows)

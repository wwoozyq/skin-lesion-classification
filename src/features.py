import pandas as pd
from tqdm import tqdm

from .dataset import list_image_ids, load_image, load_mask
from .features_color import extract_color_features
from .features_contrast import extract_contrast_features
from .features_shape import extract_shape_features
from .features_texture import extract_texture_features


def extract_features_for_image(image, mask, feature_set="all"):
    features = {}
    if feature_set in {"all", "all_contrast", "color"}:
        features.update(extract_color_features(image, mask))
    if feature_set in {"all", "all_contrast", "shape"}:
        features.update(extract_shape_features(image, mask))
    if feature_set in {"all", "all_contrast", "texture"}:
        features.update(extract_texture_features(image, mask))
    if feature_set in {"all_contrast", "contrast"}:
        features.update(extract_contrast_features(image, mask))
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

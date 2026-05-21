import pandas as pd
from tqdm import tqdm

from .dataset import list_image_ids, load_image, load_mask
from .features_abcd import extract_abcd_v2_features
from .features_abcd_grouped import extract_abcd_grouped_features
from .features_boundary import extract_boundary_features
from .features_color import extract_color_features
from .features_colorhist import extract_colorhist_features
from .features_contrast import extract_contrast_features
from .features_gabor import extract_gabor_features
from .features_hog import extract_hog_features
from .features_lbp_multi import extract_lbp_multi_features
from .features_melnv import extract_melnv_features
from .features_shape import extract_shape_features
from .features_subregion import extract_subregion_features
from .features_texture import extract_texture_features
from .preprocess import prepare_mask


BASE_FEATURES = ("color", "shape", "texture")

FEATURE_EXTRACTORS = {
    "color": extract_color_features,
    "shape": extract_shape_features,
    "texture": extract_texture_features,
    "contrast": extract_contrast_features,
    "abcd_v2": extract_abcd_v2_features,
    "boundary": extract_boundary_features,
    "abcd_grouped": extract_abcd_grouped_features,
    "colorhist": extract_colorhist_features,
    "hog": extract_hog_features,
    "melnv": extract_melnv_features,
    "lbp_multi": extract_lbp_multi_features,
    "gabor": extract_gabor_features,
    "subregion": extract_subregion_features,
}

FEATURE_ALIASES = {
    "all": BASE_FEATURES,
    "color": ("color",),
    "shape": ("shape",),
    "texture": ("texture",),
    "contrast": ("contrast",),
    "abcd_v2": ("abcd_v2",),
    "boundary": ("boundary",),
    "abcd_grouped": ("abcd_grouped",),
    "colorhist": ("colorhist",),
    "hog": ("hog",),
    "melnv": ("melnv",),
    "lbp_multi": ("lbp_multi",),
    "gabor": ("gabor",),
    "subregion": ("subregion",),
    "all_contrast": BASE_FEATURES + ("contrast",),
    "all_abcd_v2": BASE_FEATURES + ("abcd_v2",),
    "all_boundary": BASE_FEATURES + ("boundary",),
    "all_melnv": BASE_FEATURES + ("melnv",),
    "all_boundary_melnv": BASE_FEATURES + ("boundary", "melnv"),
    "all_abcd_v2_boundary": BASE_FEATURES + ("abcd_v2", "boundary"),
    "all_abcd_grouped": BASE_FEATURES + ("abcd_grouped",),
    "all_colorhist": BASE_FEATURES + ("colorhist",),
    "all_hog": BASE_FEATURES + ("hog",),
    "all_abcd_colorhist": BASE_FEATURES + ("abcd_grouped", "colorhist"),
    "all_abcd_hog": BASE_FEATURES + ("abcd_grouped", "hog"),
    "all_abcd_hog_colorhist": BASE_FEATURES + ("abcd_grouped", "hog", "colorhist"),
    "final": BASE_FEATURES + ("contrast", "abcd_v2", "boundary"),
    "final_melnv": BASE_FEATURES + ("contrast", "abcd_v2", "boundary", "melnv"),
    "final_abcd_grouped": BASE_FEATURES + ("contrast", "abcd_v2", "boundary", "abcd_grouped"),
    "all_lbp_multi": BASE_FEATURES + ("lbp_multi",),
    "all_gabor": BASE_FEATURES + ("gabor",),
    "all_subregion": BASE_FEATURES + ("subregion",),
    "all_texture_plus": BASE_FEATURES + ("lbp_multi", "gabor", "subregion"),
    "all_boundary_melnv_texture_plus": BASE_FEATURES + (
        "boundary",
        "melnv",
        "lbp_multi",
        "gabor",
        "subregion",
    ),
    "xgb_cascade_stage1": ("contrast", "abcd_v2", "boundary"),
    "xgb_cascade_stage2": BASE_FEATURES + (
        "boundary",
        "melnv",
        "lbp_multi",
        "gabor",
        "subregion",
    ),
    "early_fusion_core": BASE_FEATURES + (
        "abcd_grouped",
        "boundary",
        "melnv",
        "lbp_multi",
        "gabor",
        "subregion",
    ),
    "early_fusion_full": BASE_FEATURES + (
        "contrast",
        "abcd_v2",
        "boundary",
        "abcd_grouped",
        "melnv",
        "lbp_multi",
        "gabor",
        "subregion",
    ),
}


def resolve_feature_blocks(feature_set):
    if feature_set in FEATURE_ALIASES:
        blocks = FEATURE_ALIASES[feature_set]
    elif "+" in feature_set:
        blocks = []
        for part in [item.strip() for item in feature_set.split("+") if item.strip()]:
            if part in FEATURE_ALIASES:
                blocks.extend(FEATURE_ALIASES[part])
            else:
                raise ValueError(f"Unknown feature block or alias: {part}")
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    seen = set()
    deduped = []
    for block in blocks:
        if block not in seen:
            seen.add(block)
            deduped.append(block)
    return deduped


def extract_features_for_image(image, mask, feature_set="all"):
    features = {}
    for block in resolve_feature_blocks(feature_set):
        features.update(FEATURE_EXTRACTORS[block](image, mask))
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

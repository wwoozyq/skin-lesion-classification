import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _region_pixels(image, mask):
    pixels = image[mask]
    if pixels.size == 0:
        pixels = image.reshape(-1, image.shape[-1])
    return pixels


def _channel_stats(prefix, pixels):
    values = {}
    for idx, name in enumerate(["c0", "c1", "c2"]):
        channel = pixels[:, idx].astype(np.float32)
        values[f"{prefix}_{name}_mean"] = float(channel.mean())
        values[f"{prefix}_{name}_std"] = float(channel.std())
        values[f"{prefix}_{name}_p10"] = float(np.percentile(channel, 10))
        values[f"{prefix}_{name}_p50"] = float(np.percentile(channel, 50))
        values[f"{prefix}_{name}_p90"] = float(np.percentile(channel, 90))
    return values


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _entropy(values, bins, value_range):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist[hist > 0].astype(np.float64) / float(total)
    return float(-(probs * np.log2(probs)).sum())


def _color_bin_features(prefix, pixels, bin_size=32):
    if pixels.size == 0:
        return {
            f"{prefix}_unique_color_bins": 0.0,
            f"{prefix}_color_bin_ratio": 0.0,
        }

    quantized = np.clip(pixels.astype(np.int16) // bin_size, 0, 255 // bin_size)
    unique_bins = np.unique(quantized, axis=0).shape[0]
    max_bins = (256 // bin_size) ** pixels.shape[1]
    return {
        f"{prefix}_unique_color_bins": float(unique_bins),
        f"{prefix}_color_bin_ratio": _safe_div(unique_bins, max_bins),
    }


def _channel_diversity(prefix, pixels):
    values = {}
    for idx, name in enumerate(["c0", "c1", "c2"]):
        channel = pixels[:, idx].astype(np.float32)
        p10 = float(np.percentile(channel, 10))
        p25 = float(np.percentile(channel, 25))
        p75 = float(np.percentile(channel, 75))
        p90 = float(np.percentile(channel, 90))
        values[f"{prefix}_{name}_range_80"] = p90 - p10
        values[f"{prefix}_{name}_iqr"] = p75 - p25
        values[f"{prefix}_{name}_entropy"] = _entropy(channel, bins=16, value_range=(0, 255))
    return values


def _distance_from_mean(prefix, pixels):
    if pixels.size == 0:
        return {
            f"{prefix}_mean_color_distance": 0.0,
            f"{prefix}_p90_color_distance": 0.0,
        }

    values = pixels.astype(np.float32)
    center = values.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(values - center, axis=1)
    return {
        f"{prefix}_mean_color_distance": float(distances.mean()),
        f"{prefix}_p90_color_distance": float(np.percentile(distances, 90)),
    }


def _extreme_channel_ratios(prefix, pixels):
    values = {}
    for idx, name in enumerate(["c0", "c1", "c2"]):
        channel = pixels[:, idx].astype(np.float32)
        if channel.size == 0:
            values[f"{prefix}_{name}_low_ratio"] = 0.0
            values[f"{prefix}_{name}_high_ratio"] = 0.0
            values[f"{prefix}_{name}_coef_var"] = 0.0
            continue
        values[f"{prefix}_{name}_low_ratio"] = float((channel < 64).mean())
        values[f"{prefix}_{name}_high_ratio"] = float((channel > 192).mean())
        values[f"{prefix}_{name}_coef_var"] = _safe_div(float(channel.std()), float(channel.mean()))
    return values


def _lesion_background_contrast(prefix, image_space, mask):
    lesion_pixels = image_space[mask]
    background_pixels = image_space[~mask]
    if lesion_pixels.size == 0 or background_pixels.size == 0:
        return {
            f"{prefix}_background_contrast": 0.0,
        }

    lesion_mean = lesion_pixels.astype(np.float32).mean(axis=0)
    background_mean = background_pixels.astype(np.float32).mean(axis=0)
    return {
        f"{prefix}_background_contrast": float(np.linalg.norm(lesion_mean - background_mean)),
    }


def _center_border_color_difference(prefix, image_space, mask):
    if not mask.any():
        return {
            f"{prefix}_center_border_color_distance": 0.0,
            f"{prefix}_center_border_c0_diff": 0.0,
            f"{prefix}_center_border_c1_diff": 0.0,
            f"{prefix}_center_border_c2_diff": 0.0,
        }

    ys, xs = np.where(mask)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    erosion_iter = max(1, min(bbox_h, bbox_w) // 8)
    center_mask = ndi.binary_erosion(mask, iterations=erosion_iter)
    if center_mask.sum() < 10:
        erosion_iter = max(1, min(bbox_h, bbox_w) // 16)
        center_mask = ndi.binary_erosion(mask, iterations=erosion_iter)
    if center_mask.sum() < 10:
        center_mask = mask

    border_mask = np.logical_and(mask, np.logical_not(center_mask))
    if border_mask.sum() < 10:
        eroded_once = ndi.binary_erosion(mask)
        border_mask = np.logical_xor(mask, eroded_once)
    if border_mask.sum() == 0:
        border_mask = mask

    center_mean = image_space[center_mask].astype(np.float32).mean(axis=0)
    border_mean = image_space[border_mask].astype(np.float32).mean(axis=0)
    diff = center_mean - border_mean
    return {
        f"{prefix}_center_border_color_distance": float(np.linalg.norm(diff)),
        f"{prefix}_center_border_c0_diff": float(diff[0]),
        f"{prefix}_center_border_c1_diff": float(diff[1]),
        f"{prefix}_center_border_c2_diff": float(diff[2]),
    }


def _abcd_color_features(rgb_pixels, hsv_pixels, lab_pixels):
    features = {}
    features.update(_channel_diversity("rgb", rgb_pixels))
    features.update(_channel_diversity("hsv", hsv_pixels))
    features.update(_channel_diversity("lab", lab_pixels))
    features.update(_color_bin_features("rgb", rgb_pixels))
    features.update(_color_bin_features("hsv", hsv_pixels))
    features.update(_color_bin_features("lab", lab_pixels))
    features.update(_distance_from_mean("rgb", rgb_pixels))
    features.update(_distance_from_mean("lab", lab_pixels))
    features.update(_extreme_channel_ratios("rgb", rgb_pixels))
    features.update(_extreme_channel_ratios("hsv", hsv_pixels))
    features.update(_extreme_channel_ratios("lab", lab_pixels))

    hsv = hsv_pixels.astype(np.float32)
    rgb = rgb_pixels.astype(np.float32)
    value = hsv[:, 2]
    saturation = hsv[:, 1]
    hue = hsv[:, 0]
    features["dark_pixel_ratio"] = float((value < 80).mean()) if value.size else 0.0
    features["bright_pixel_ratio"] = float((value > 200).mean()) if value.size else 0.0
    features["high_saturation_ratio"] = float((saturation > 128).mean()) if saturation.size else 0.0
    features["low_saturation_ratio"] = float((saturation < 40).mean()) if saturation.size else 0.0
    features["hue_entropy"] = _entropy(hue, bins=18, value_range=(0, 255)) if hue.size else 0.0

    if hsv.size and rgb.size:
        red_like = ((hue < 18) | (hue > 235)) & (saturation > 60) & (value > 50)
        black_like = value < 55
        dark_brown_like = (value >= 55) & (value < 115) & (saturation > 45)
        light_brown_like = (value >= 115) & (value < 190) & (saturation > 35)
        white_like = (value > 185) & (saturation < 45)
        blue_gray_like = (hue > 125) & (hue < 185) & (saturation > 25) & (value > 55)
        color_masks = [
            red_like,
            black_like,
            dark_brown_like,
            light_brown_like,
            white_like,
            blue_gray_like,
        ]
        color_names = ["red", "black", "dark_brown", "light_brown", "white", "blue_gray"]
        color_ratios = []
        for name, color_mask in zip(color_names, color_masks):
            ratio = float(color_mask.mean())
            features[f"abcd_{name}_ratio"] = ratio
            color_ratios.append(ratio)
        features["abcd_color_count"] = float(sum(ratio > 0.02 for ratio in color_ratios))
        features["abcd_color_coverage"] = float(sum(color_ratios))
        features["rgb_channel_spread_mean"] = float((rgb.max(axis=1) - rgb.min(axis=1)).mean())
    else:
        for name in ["red", "black", "dark_brown", "light_brown", "white", "blue_gray"]:
            features[f"abcd_{name}_ratio"] = 0.0
        features["abcd_color_count"] = 0.0
        features["abcd_color_coverage"] = 0.0
        features["rgb_channel_spread_mean"] = 0.0
    return features


def extract_color_features(image, mask):
    mask = np.asarray(mask).astype(bool)
    features = {}
    rgb_pixels = _region_pixels(image, mask)
    features.update(_channel_stats("rgb", rgb_pixels))

    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hsv_pixels = _region_pixels(hsv, mask)
    features.update(_channel_stats("hsv", hsv_pixels))

    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    lab_pixels = _region_pixels(lab, mask)
    features.update(_channel_stats("lab", lab_pixels))
    features.update(_abcd_color_features(rgb_pixels, hsv_pixels, lab_pixels))
    features.update(_lesion_background_contrast("rgb", image, mask))
    features.update(_lesion_background_contrast("hsv", hsv, mask))
    features.update(_lesion_background_contrast("lab", lab, mask))
    features.update(_center_border_color_difference("rgb", image, mask))
    features.update(_center_border_color_difference("hsv", hsv, mask))
    features.update(_center_border_color_difference("lab", lab, mask))
    return features

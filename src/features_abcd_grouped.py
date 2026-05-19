import math

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError


def _safe_div(numerator, denominator, cap=1_000_000.0):
    if abs(denominator) < 1e-8:
        return 0.0
    value = float(numerator / denominator)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -cap, cap))


def _safe_mask(mask):
    mask = np.asarray(mask).astype(bool)
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _entropy(values, bins=16, value_range=(0, 255)):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0].astype(np.float64) / float(total)
    return float(-(prob * np.log2(prob)).sum())


def _channel_diversity(prefix, pixels):
    values = {}
    for idx, name in enumerate(["c0", "c1", "c2"]):
        channel = pixels[:, idx].astype(np.float32)
        values[f"abcd_grouped_{prefix}_{name}_range_80"] = float(np.percentile(channel, 90) - np.percentile(channel, 10))
        values[f"abcd_grouped_{prefix}_{name}_iqr"] = float(np.percentile(channel, 75) - np.percentile(channel, 25))
        values[f"abcd_grouped_{prefix}_{name}_entropy"] = _entropy(channel)
        values[f"abcd_grouped_{prefix}_{name}_low_ratio"] = float((channel < 64).mean())
        values[f"abcd_grouped_{prefix}_{name}_high_ratio"] = float((channel > 192).mean())
        values[f"abcd_grouped_{prefix}_{name}_coef_var"] = _safe_div(float(channel.std()), float(channel.mean()))
    return values


def _color_bin_features(prefix, pixels, bin_size=32):
    quantized = np.clip(pixels.astype(np.int16) // bin_size, 0, 255 // bin_size)
    unique_bins = np.unique(quantized, axis=0).shape[0]
    max_bins = (256 // bin_size) ** pixels.shape[1]
    return {
        f"abcd_grouped_{prefix}_unique_color_bins": float(unique_bins),
        f"abcd_grouped_{prefix}_color_bin_ratio": _safe_div(unique_bins, max_bins),
    }


def _distance_features(prefix, pixels):
    values = pixels.astype(np.float32)
    center = values.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(values - center, axis=1)
    return {
        f"abcd_grouped_{prefix}_mean_color_distance": float(distances.mean()),
        f"abcd_grouped_{prefix}_p90_color_distance": float(np.percentile(distances, 90)),
    }


def _diagnostic_color_features(rgb_pixels, hsv_pixels):
    hsv = hsv_pixels.astype(np.float32)
    rgb = rgb_pixels.astype(np.float32)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]

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
    ratios = []
    features = {
        "abcd_grouped_dark_pixel_ratio": float((value < 80).mean()),
        "abcd_grouped_bright_pixel_ratio": float((value > 200).mean()),
        "abcd_grouped_high_saturation_ratio": float((saturation > 128).mean()),
        "abcd_grouped_low_saturation_ratio": float((saturation < 40).mean()),
        "abcd_grouped_hue_entropy": _entropy(hue, bins=18),
        "abcd_grouped_rgb_channel_spread_mean": float((rgb.max(axis=1) - rgb.min(axis=1)).mean()),
    }
    for name, color_mask in zip(color_names, color_masks):
        ratio = float(color_mask.mean())
        features[f"abcd_grouped_{name}_ratio"] = ratio
        ratios.append(ratio)
    features["abcd_grouped_color_count"] = float(sum(ratio > 0.02 for ratio in ratios))
    features["abcd_grouped_color_coverage"] = float(sum(ratios))
    return features


def _lesion_background_contrast(prefix, image_space, mask):
    lesion_pixels = image_space[mask]
    background_pixels = image_space[~mask]
    if lesion_pixels.size == 0 or background_pixels.size == 0:
        return {f"abcd_grouped_{prefix}_background_contrast": 0.0}
    lesion_mean = lesion_pixels.astype(np.float32).mean(axis=0)
    background_mean = background_pixels.astype(np.float32).mean(axis=0)
    return {f"abcd_grouped_{prefix}_background_contrast": float(np.linalg.norm(lesion_mean - background_mean))}


def _center_border_color_difference(prefix, image_space, mask):
    ys, xs = np.where(mask)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    center_mask = ndi.binary_erosion(mask, iterations=max(1, min(bbox_h, bbox_w) // 8))
    if center_mask.sum() < 10:
        center_mask = ndi.binary_erosion(mask, iterations=max(1, min(bbox_h, bbox_w) // 16))
    if center_mask.sum() < 10:
        center_mask = mask

    border_mask = np.logical_and(mask, np.logical_not(center_mask))
    if border_mask.sum() < 10:
        border_mask = np.logical_xor(mask, ndi.binary_erosion(mask))
    if border_mask.sum() == 0:
        border_mask = mask

    center_mean = image_space[center_mask].astype(np.float32).mean(axis=0)
    border_mean = image_space[border_mask].astype(np.float32).mean(axis=0)
    diff = center_mean - border_mean
    return {
        f"abcd_grouped_{prefix}_center_border_color_distance": float(np.linalg.norm(diff)),
        f"abcd_grouped_{prefix}_center_border_c0_diff": float(diff[0]),
        f"abcd_grouped_{prefix}_center_border_c1_diff": float(diff[1]),
        f"abcd_grouped_{prefix}_center_border_c2_diff": float(diff[2]),
    }


def _asymmetry(cropped_mask, axis):
    area = float(cropped_mask.sum())
    if area == 0:
        return 0.0
    return float(np.logical_xor(cropped_mask, np.flip(cropped_mask, axis=axis)).sum() / area)


def _moment_stats(values):
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "cv": 0.0, "range": 0.0, "p10": 0.0, "p90": 0.0}
    values = values.astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    return {
        "mean": mean,
        "std": std,
        "cv": _safe_div(std, mean),
        "range": float(values.max() - values.min()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def _convex_hull_area(ys, xs):
    if ys.size < 3:
        return 0.0
    points = []
    for dy, dx in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)):
        points.append(np.column_stack([xs + dx, ys + dy]))
    try:
        hull = ConvexHull(np.vstack(points).astype(np.float64))
    except QhullError:
        return 0.0
    return float(hull.volume)


def _ellipse_features(ys, xs):
    if ys.size < 2:
        return {
            "abcd_grouped_ellipse_major_axis": 0.0,
            "abcd_grouped_ellipse_minor_axis": 0.0,
            "abcd_grouped_ellipse_axis_ratio": 0.0,
            "abcd_grouped_ellipse_eccentricity": 0.0,
        }
    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return {
            "abcd_grouped_ellipse_major_axis": 0.0,
            "abcd_grouped_ellipse_minor_axis": 0.0,
            "abcd_grouped_ellipse_axis_ratio": 0.0,
            "abcd_grouped_ellipse_eccentricity": 0.0,
        }
    eigvals = np.maximum(np.sort(np.linalg.eigvalsh(cov))[::-1], 0.0)
    major = float(4.0 * math.sqrt(eigvals[0])) if eigvals[0] > 0 else 0.0
    minor = float(4.0 * math.sqrt(eigvals[1])) if eigvals[1] > 0 else 0.0
    eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))) if major > 0 else 0.0
    return {
        "abcd_grouped_ellipse_major_axis": major,
        "abcd_grouped_ellipse_minor_axis": minor,
        "abcd_grouped_ellipse_axis_ratio": _safe_div(major, minor),
        "abcd_grouped_ellipse_eccentricity": float(eccentricity),
    }


def extract_abcd_grouped_features(image, mask):
    mask = _safe_mask(mask)
    h, w = mask.shape[:2]
    image_area = float(max(h * w, 1))
    area = float(mask.sum())
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    bbox_area = float(max(bbox_h * bbox_w, 1))
    cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

    eroded = ndi.binary_erosion(mask)
    boundary = np.logical_xor(mask, eroded)
    boundary_ys, boundary_xs = np.where(boundary)
    perimeter = float(boundary.sum())
    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())
    boundary_distances = np.sqrt((boundary_xs - centroid_x) ** 2 + (boundary_ys - centroid_y) ** 2)
    radial = _moment_stats(boundary_distances)
    image_diagonal = math.sqrt(h * h + w * w)
    bbox_diagonal = math.sqrt(bbox_h * bbox_h + bbox_w * bbox_w)
    convex_hull_area = _convex_hull_area(boundary_ys.astype(np.float64), boundary_xs.astype(np.float64))
    solidity = min(_safe_div(area, convex_hull_area), 1.0) if convex_hull_area > 0 else 0.0
    equivalent_diameter = math.sqrt(4.0 * area / math.pi)
    enclosing_circle_radius = float(np.sqrt((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2).max())

    features = {
        "abcd_grouped_area": area,
        "abcd_grouped_mask_area_ratio": area / image_area,
        "abcd_grouped_perimeter": perimeter,
        "abcd_grouped_normalized_perimeter": _safe_div(perimeter, image_diagonal),
        "abcd_grouped_perimeter_area_ratio": _safe_div(perimeter, area),
        "abcd_grouped_bbox_width": _safe_div(bbox_w, w),
        "abcd_grouped_bbox_height": _safe_div(bbox_h, h),
        "abcd_grouped_bbox_aspect_ratio": _safe_div(bbox_w, bbox_h),
        "abcd_grouped_bbox_area_ratio": bbox_area / image_area,
        "abcd_grouped_extent": _safe_div(area, bbox_area),
        "abcd_grouped_horizontal_asymmetry": _asymmetry(cropped, axis=1),
        "abcd_grouped_vertical_asymmetry": _asymmetry(cropped, axis=0),
        "abcd_grouped_rotation_asymmetry_180": _asymmetry(cropped, axis=(0, 1)),
        "abcd_grouped_circularity": _safe_div(4.0 * math.pi * area, perimeter * perimeter),
        "abcd_grouped_compactness": _safe_div(perimeter * perimeter, area),
        "abcd_grouped_solidity": solidity,
        "abcd_grouped_convexity_defect": 1.0 - solidity if convex_hull_area > 0 else 0.0,
        "abcd_grouped_equivalent_diameter_ratio": _safe_div(equivalent_diameter, image_diagonal),
        "abcd_grouped_enclosing_circle_radius_ratio": _safe_div(enclosing_circle_radius, image_diagonal),
        "abcd_grouped_bbox_diagonal_ratio": _safe_div(bbox_diagonal, image_diagonal),
        "abcd_grouped_radial_mean": radial["mean"],
        "abcd_grouped_radial_std": radial["std"],
        "abcd_grouped_radial_cv": radial["cv"],
        "abcd_grouped_radial_range": radial["range"],
        "abcd_grouped_radial_p10": radial["p10"],
        "abcd_grouped_radial_p90": radial["p90"],
        "abcd_grouped_boundary_roughness": _safe_div(radial["std"], radial["mean"]),
        "abcd_grouped_boundary_to_bbox_ratio": _safe_div(perimeter, max(2 * (bbox_h + bbox_w), 1)),
    }
    features.update(_ellipse_features(ys, xs))

    rgb_pixels = image[mask]
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hsv_pixels = hsv[mask]
    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    lab_pixels = lab[mask]
    for prefix, pixels in [("rgb", rgb_pixels), ("hsv", hsv_pixels), ("lab", lab_pixels)]:
        features.update(_channel_diversity(prefix, pixels))
        features.update(_color_bin_features(prefix, pixels))
        features.update(_distance_features(prefix, pixels))
    features.update(_diagnostic_color_features(rgb_pixels, hsv_pixels))
    features.update(_lesion_background_contrast("rgb", image, mask))
    features.update(_lesion_background_contrast("hsv", hsv, mask))
    features.update(_lesion_background_contrast("lab", lab, mask))
    features.update(_center_border_color_difference("rgb", image, mask))
    features.update(_center_border_color_difference("hsv", hsv, mask))
    features.update(_center_border_color_difference("lab", lab, mask))
    return features

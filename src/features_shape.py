import math

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError


SHAPE_FEATURE_NAMES = [
    "mask_area_ratio",
    "area",
    "perimeter",
    "normalized_perimeter",
    "perimeter_area_ratio",
    "bbox_width",
    "bbox_height",
    "bbox_aspect_ratio",
    "bbox_area_ratio",
    "extent",
    "center_y",
    "center_x",
    "horizontal_asymmetry",
    "vertical_asymmetry",
    "rotation_asymmetry_180",
    "bbox_center_offset",
    "ellipse_major_axis",
    "ellipse_minor_axis",
    "ellipse_axis_ratio",
    "ellipse_eccentricity",
    "circularity",
    "compactness",
    "solidity",
    "convexity_defect",
    "convex_hull_area_ratio",
    "contour_area_ratio",
    "equivalent_diameter",
    "equivalent_diameter_ratio",
    "enclosing_circle_radius",
    "enclosing_circle_radius_ratio",
    "circle_area_ratio",
    "bbox_diagonal_ratio",
    "radial_mean",
    "radial_std",
    "radial_cv",
    "radial_range",
    "radius_std",
    "radius_cv",
    "radius_range",
    "radial_p10",
    "radial_p90",
    "radial_skewness",
    "radial_kurtosis",
    "boundary_roughness",
    "boundary_to_bbox_ratio",
]


def _zero_features():
    return {name: 0.0 for name in SHAPE_FEATURE_NAMES}


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _foreground_mask(mask):
    return np.asarray(mask).astype(bool)


def _boundary(mask):
    if not mask.any():
        return mask
    eroded = ndi.binary_erosion(mask)
    return np.logical_xor(mask, eroded)


def _bbox(mask):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(y_min := ys.min()), int(y_max := ys.max()), int(x_min := xs.min()), int(x_max := xs.max()), ys, xs


def _crop_to_bbox(mask, y_min, y_max, x_min, x_max):
    return mask[y_min : y_max + 1, x_min : x_max + 1]


def _asymmetry(cropped_mask, axis):
    if cropped_mask.size == 0:
        return 0.0
    area = float(cropped_mask.sum())
    if area == 0:
        return 0.0
    flipped = np.flip(cropped_mask, axis=axis)
    return float(np.logical_xor(cropped_mask, flipped).sum() / area)


def _rotation_asymmetry(cropped_mask):
    if cropped_mask.size == 0:
        return 0.0
    area = float(cropped_mask.sum())
    if area == 0:
        return 0.0
    rotated = np.rot90(cropped_mask, 2)
    return float(np.logical_xor(cropped_mask, rotated).sum() / area)


def _moment_stats(values):
    if values.size == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "range": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
        }

    values = values.astype(np.float64)
    mean = float(values.mean())
    std = float(values.std())
    centered = values - mean
    if std == 0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        z = centered / std
        skewness = float((z ** 3).mean())
        kurtosis = float((z ** 4).mean())

    return {
        "mean": mean,
        "std": std,
        "cv": _safe_div(std, mean),
        "range": float(values.max() - values.min()),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }


def _ellipse_moment_features(ys, xs, area):
    if area <= 1 or ys.size < 2:
        return {
            "ellipse_major_axis": 0.0,
            "ellipse_minor_axis": 0.0,
            "ellipse_axis_ratio": 0.0,
            "ellipse_eccentricity": 0.0,
        }

    coords = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    centered = coords - coords.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return {
            "ellipse_major_axis": 0.0,
            "ellipse_minor_axis": 0.0,
            "ellipse_axis_ratio": 0.0,
            "ellipse_eccentricity": 0.0,
        }

    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    major = float(4.0 * math.sqrt(eigvals[0])) if eigvals[0] > 0 else 0.0
    minor = float(4.0 * math.sqrt(eigvals[1])) if eigvals[1] > 0 else 0.0
    axis_ratio = _safe_div(major, minor)
    eccentricity = 0.0
    if major > 0:
        eccentricity = math.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major)))

    return {
        "ellipse_major_axis": major,
        "ellipse_minor_axis": minor,
        "ellipse_axis_ratio": axis_ratio,
        "ellipse_eccentricity": float(eccentricity),
    }


def _convex_hull_area(points):
    if points.shape[0] < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
    except QhullError:
        return 0.0
    return float(hull.volume)


def _pixel_corner_points(ys, xs):
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    corners = []
    for dy, dx in ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)):
        corners.append(np.column_stack([xs + dx, ys + dy]))
    return np.vstack(corners).astype(np.float64)


def extract_shape_features(image, mask):
    """Extract classical geometry features from a lesion mask.

    The features are intentionally traditional image-processing descriptors,
    suitable for RandomForest/XGBoost style machine-learning classifiers.
    """
    mask = _foreground_mask(mask)
    h, w = mask.shape[:2]
    image_area = float(max(h * w, 1))
    area = float(mask.sum())

    features = _zero_features()
    if area == 0:
        return features

    bbox = _bbox(mask)
    if bbox is None:
        return features

    y_min, y_max, x_min, x_max, ys, xs = bbox
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    bbox_area = float(max(bbox_h * bbox_w, 1))

    boundary = _boundary(mask)
    perimeter = float(boundary.sum())
    cropped = _crop_to_bbox(mask, y_min, y_max, x_min, x_max)

    boundary_ys, boundary_xs = np.where(boundary)
    boundary_points = _pixel_corner_points(boundary_ys.astype(np.float64), boundary_xs.astype(np.float64))
    convex_hull_area = _convex_hull_area(boundary_points)
    solidity = min(_safe_div(area, convex_hull_area), 1.0) if convex_hull_area > 0 else 0.0

    centroid_y = float(ys.mean())
    centroid_x = float(xs.mean())
    distances = np.sqrt((xs - centroid_x) ** 2 + (ys - centroid_y) ** 2)
    enclosing_circle_radius = float(distances.max()) if distances.size else 0.0
    enclosing_circle_area = math.pi * enclosing_circle_radius * enclosing_circle_radius
    boundary_distances = np.sqrt((boundary_xs - centroid_x) ** 2 + (boundary_ys - centroid_y) ** 2)
    radial = _moment_stats(boundary_distances)
    image_diagonal = math.sqrt(h * h + w * w)
    bbox_diagonal = math.sqrt(bbox_h * bbox_h + bbox_w * bbox_w)
    bbox_center_y = y_min + (bbox_h - 1) / 2.0
    bbox_center_x = x_min + (bbox_w - 1) / 2.0
    bbox_center_offset = math.sqrt((centroid_y - bbox_center_y) ** 2 + (centroid_x - bbox_center_x) ** 2)
    equivalent_diameter = math.sqrt(4.0 * area / math.pi)

    features.update(
        {
            "mask_area_ratio": area / image_area,
            "area": area,
            "perimeter": perimeter,
            "normalized_perimeter": _safe_div(perimeter, image_diagonal),
            "perimeter_area_ratio": _safe_div(perimeter, area),
            "bbox_width": _safe_div(bbox_w, w),
            "bbox_height": _safe_div(bbox_h, h),
            "bbox_aspect_ratio": _safe_div(bbox_w, bbox_h),
            "bbox_area_ratio": bbox_area / image_area,
            "extent": _safe_div(area, bbox_area),
            "center_y": _safe_div(centroid_y, max(h - 1, 1)),
            "center_x": _safe_div(centroid_x, max(w - 1, 1)),
            "horizontal_asymmetry": _asymmetry(cropped, axis=1),
            "vertical_asymmetry": _asymmetry(cropped, axis=0),
            "rotation_asymmetry_180": _rotation_asymmetry(cropped),
            "bbox_center_offset": _safe_div(bbox_center_offset, max(bbox_diagonal, 1.0)),
            "circularity": _safe_div(4.0 * math.pi * area, perimeter * perimeter),
            "compactness": _safe_div(perimeter * perimeter, area),
            "solidity": solidity,
            "convexity_defect": 1.0 - solidity if convex_hull_area > 0 else 0.0,
            "convex_hull_area_ratio": convex_hull_area / image_area,
            "contour_area_ratio": _safe_div(area, max(perimeter, 1.0)),
            "equivalent_diameter": equivalent_diameter,
            "equivalent_diameter_ratio": _safe_div(equivalent_diameter, image_diagonal),
            "enclosing_circle_radius": enclosing_circle_radius,
            "enclosing_circle_radius_ratio": _safe_div(enclosing_circle_radius, image_diagonal),
            "circle_area_ratio": _safe_div(area, enclosing_circle_area),
            "bbox_diagonal_ratio": _safe_div(bbox_diagonal, image_diagonal),
            "radial_mean": radial["mean"],
            "radial_std": radial["std"],
            "radial_cv": radial["cv"],
            "radial_range": radial["range"],
            "radius_std": radial["std"],
            "radius_cv": radial["cv"],
            "radius_range": radial["range"],
            "radial_p10": radial["p10"],
            "radial_p90": radial["p90"],
            "radial_skewness": radial["skewness"],
            "radial_kurtosis": radial["kurtosis"],
            "boundary_roughness": _safe_div(radial["std"], radial["mean"]),
            "boundary_to_bbox_ratio": _safe_div(perimeter, max(2 * (bbox_h + bbox_w), 1)),
        }
    )
    features.update(_ellipse_moment_features(ys, xs, area))
    return features

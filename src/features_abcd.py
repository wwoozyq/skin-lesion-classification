import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _safe_mask(mask):
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _foreground_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _entropy(values, bins=32, value_range=(0, 255)):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _binary_asymmetry(mask, axis):
    if not mask.any():
        return 0.0
    flipped = np.flip(mask, axis=axis)
    union = np.logical_or(mask, flipped).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(mask, flipped).sum()
    return float(1.0 - intersection / union)


def _pca_axes(mask):
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs, ys]).astype(np.float64)
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    if not np.isfinite(vh).all():
        return None
    return coords, center, vh[0], vh[1]


def _pca_asymmetry(image, mask):
    axes = _pca_axes(mask)
    empty = {
        "abcd_v2_pca_area_asymmetry_major": 0.0,
        "abcd_v2_pca_area_asymmetry_minor": 0.0,
        "abcd_v2_pca_color_asymmetry_major": 0.0,
        "abcd_v2_pca_color_asymmetry_minor": 0.0,
    }
    if axes is None:
        return empty

    coords, center, major, minor = axes
    pixels = image[mask].astype(np.float32)
    centered = coords - center
    proj_major = centered[:, 0] * major[0] + centered[:, 1] * major[1]
    proj_minor = centered[:, 0] * minor[0] + centered[:, 1] * minor[1]
    if not np.isfinite(proj_major).all() or not np.isfinite(proj_minor).all():
        return empty

    def split_stats(projection):
        pos = projection >= 0
        neg = projection < 0
        pos_count = pos.sum()
        neg_count = neg.sum()
        area_asymmetry = abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
        if pos_count == 0 or neg_count == 0:
            color_asymmetry = 0.0
        else:
            color_asymmetry = np.linalg.norm(
                pixels[pos].mean(axis=0) - pixels[neg].mean(axis=0)
            ) / 255.0
        return float(area_asymmetry), float(color_asymmetry)

    major_area, major_color = split_stats(proj_minor)
    minor_area, minor_color = split_stats(proj_major)
    return {
        "abcd_v2_pca_area_asymmetry_major": major_area,
        "abcd_v2_pca_area_asymmetry_minor": minor_area,
        "abcd_v2_pca_color_asymmetry_major": major_color,
        "abcd_v2_pca_color_asymmetry_minor": minor_color,
    }


def _border_points(mask):
    eroded = ndi.binary_erosion(mask)
    border = np.logical_xor(mask, eroded)
    ys, xs = np.where(border)
    if len(xs) == 0:
        ys, xs = np.where(mask)
    return xs.astype(np.float32), ys.astype(np.float32)


def _box_count(binary, box_size):
    h, w = binary.shape
    count = 0
    for y in range(0, h, box_size):
        for x in range(0, w, box_size):
            if binary[y:y + box_size, x:x + box_size].any():
                count += 1
    return count


def _fractal_dimension(binary):
    min_dim = min(binary.shape)
    sizes = []
    size = 2
    while size <= max(min_dim // 2, 2):
        sizes.append(size)
        size *= 2
    if len(sizes) < 2:
        return 0.0
    counts = np.array([_box_count(binary, size) for size in sizes], dtype=np.float32)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    coeff = np.polyfit(np.log(1 / np.array(sizes)[valid]), np.log(counts[valid]), 1)
    return float(coeff[0])


def _radial_border_features(mask):
    xs, ys = _border_points(mask)
    if len(xs) < 3:
        return {
            "abcd_v2_border_radial_cv": 0.0,
            "abcd_v2_border_radial_range": 0.0,
            "abcd_v2_border_roughness": 0.0,
            "abcd_v2_border_fractal_dim": 0.0,
        }
    center_x = xs.mean()
    center_y = ys.mean()
    radius = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)
    radius_mean = max(radius.mean(), 1e-6)
    angles = np.arctan2(ys - center_y, xs - center_x)
    sorted_radius = radius[np.argsort(angles)]
    roughness = np.abs(np.diff(np.r_[sorted_radius, sorted_radius[0]])).mean() / radius_mean
    border = np.zeros_like(mask, dtype=bool)
    border[ys.astype(int), xs.astype(int)] = True
    return {
        "abcd_v2_border_radial_cv": float(radius.std() / radius_mean),
        "abcd_v2_border_radial_range": float((radius.max() - radius.min()) / radius_mean),
        "abcd_v2_border_roughness": float(roughness),
        "abcd_v2_border_fractal_dim": _fractal_dimension(border),
    }


def _diagnostic_color_features(image, mask):
    pixels = image[mask].astype(np.float32)
    if pixels.size == 0:
        pixels = image.reshape(-1, 3).astype(np.float32)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    intensity = pixels.mean(axis=1)

    color_masks = [
        intensity < 45,
        (intensity > 210) & (np.abs(r - g) < 25) & (np.abs(r - b) < 25),
        (r > 120) & (r > g * 1.2) & (r > b * 1.2),
        (b > r * 0.9) & (b > g * 0.9) & (intensity < 150),
        (r > g) & (g > b) & (intensity >= 45) & (intensity < 120),
        (r > g) & (g > b) & (intensity >= 120) & (intensity < 210),
    ]
    ratios = np.array([m.mean() for m in color_masks], dtype=np.float32)
    prob = ratios[ratios > 0]
    entropy = float(-(prob * np.log2(prob)).sum()) if prob.size else 0.0

    return {
        "abcd_v2_diag_color_count": float((ratios > 0.01).sum()),
        "abcd_v2_black_ratio": float(ratios[0]),
        "abcd_v2_white_ratio": float(ratios[1]),
        "abcd_v2_red_ratio": float(ratios[2]),
        "abcd_v2_blue_gray_ratio": float(ratios[3]),
        "abcd_v2_dark_brown_ratio": float(ratios[4]),
        "abcd_v2_light_brown_ratio": float(ratios[5]),
        "abcd_v2_diag_color_entropy": entropy,
        "abcd_v2_dominant_color_ratio": float(ratios.max()) if ratios.size else 0.0,
    }


def extract_abcd_v2_features(image, mask):
    mask = _safe_mask(mask)
    h, w = mask.shape
    x0, y0, x1, y1 = _foreground_bbox(mask)
    bbox_w = max(x1 - x0, 1)
    bbox_h = max(y1 - y0, 1)
    cropped_mask = mask[y0:y1, x0:x1]
    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())

    gray = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)
    gray_pixels = gray[mask]
    gy, gx = np.gradient(gray)
    grad_pixels = np.sqrt(gx ** 2 + gy ** 2)[mask]

    features = {
        "abcd_v2_asymmetry_lr": _binary_asymmetry(cropped_mask, axis=1),
        "abcd_v2_asymmetry_ud": _binary_asymmetry(cropped_mask, axis=0),
        "abcd_v2_border_compactness": float(perimeter ** 2 / max(area, 1.0)),
        "abcd_v2_circularity": float(4.0 * np.pi * area / max(perimeter ** 2, 1.0)),
        "abcd_v2_area_ratio": float(area / max(h * w, 1)),
        "abcd_v2_bbox_diameter_ratio": float(
            np.sqrt(bbox_w ** 2 + bbox_h ** 2) / max(np.sqrt(w ** 2 + h ** 2), 1e-6)
        ),
        "abcd_v2_major_minor_ratio": float(max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)),
        "abcd_v2_gray_entropy": _entropy(gray_pixels),
        "abcd_v2_gradient_mean": float(grad_pixels.mean()) if grad_pixels.size else 0.0,
        "abcd_v2_gradient_std": float(grad_pixels.std()) if grad_pixels.size else 0.0,
    }
    features.update(_pca_asymmetry(image, mask))
    features.update(_radial_border_features(mask))
    features.update(_diagnostic_color_features(image, mask))
    return features

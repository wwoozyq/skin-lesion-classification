import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _safe_region(mask):
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _foreground_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, mask.shape[1], mask.shape[0]
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def _crop_to_bbox(array, bbox):
    x0, y0, x1, y1 = bbox
    return array[y0:y1, x0:x1]


def _resize_like_halves(left, right):
    height = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    if height == 0 or width == 0:
        return left, right
    return left[:height, :width], right[:height, :width]


def _binary_asymmetry(binary, axis):
    if binary.size == 0 or not binary.any():
        return 0.0
    flipped = np.flip(binary, axis=axis)
    intersection = np.logical_and(binary, flipped).sum()
    union = np.logical_or(binary, flipped).sum()
    if union == 0:
        return 0.0
    return float(1.0 - intersection / union)


def _half_color_asymmetry(image, mask, vertical=True):
    bbox = _foreground_bbox(mask)
    roi_img = _crop_to_bbox(image, bbox)
    roi_mask = _crop_to_bbox(mask, bbox)
    if roi_mask.size == 0 or not roi_mask.any():
        return 0.0

    if vertical:
        mid = roi_mask.shape[1] // 2
        part_a_mask = roi_mask[:, :mid]
        part_b_mask = np.fliplr(roi_mask[:, mid:])
        part_a_img = roi_img[:, :mid]
        part_b_img = np.fliplr(roi_img[:, mid:])
    else:
        mid = roi_mask.shape[0] // 2
        part_a_mask = roi_mask[:mid, :]
        part_b_mask = np.flipud(roi_mask[mid:, :])
        part_a_img = roi_img[:mid, :]
        part_b_img = np.flipud(roi_img[mid:, :])

    part_a_mask, part_b_mask = _resize_like_halves(part_a_mask, part_b_mask)
    part_a_img = part_a_img[:part_a_mask.shape[0], :part_a_mask.shape[1]]
    part_b_img = part_b_img[:part_b_mask.shape[0], :part_b_mask.shape[1]]

    pixels_a = part_a_img[part_a_mask]
    pixels_b = part_b_img[part_b_mask]
    if pixels_a.size == 0 or pixels_b.size == 0:
        return 0.0
    mean_a = pixels_a.astype(np.float32).mean(axis=0)
    mean_b = pixels_b.astype(np.float32).mean(axis=0)
    return float(np.linalg.norm(mean_a - mean_b) / 255.0)


def _entropy(values, bins=32, value_range=(0, 255)):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _channel_std_mean(image, mask):
    pixels = image[_safe_region(mask)].astype(np.float32)
    return float(pixels.std(axis=0).mean())


def _color_range_mean(image, mask):
    pixels = image[_safe_region(mask)].astype(np.float32)
    return float((np.percentile(pixels, 90, axis=0) - np.percentile(pixels, 10, axis=0)).mean())


def _color_bin_count(image, mask, bins_per_channel=4):
    pixels = image[_safe_region(mask)]
    if pixels.size == 0:
        return 0.0
    quantized = np.clip(pixels // (256 // bins_per_channel), 0, bins_per_channel - 1)
    code = (
        quantized[:, 0] * bins_per_channel * bins_per_channel
        + quantized[:, 1] * bins_per_channel
        + quantized[:, 2]
    )
    return float(len(np.unique(code)))


def _pca_axes(mask):
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return None
    coords = np.column_stack([xs, ys]).astype(np.float32)
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    major = vh[0]
    minor = vh[1]
    return coords, center, major, minor


def _pca_asymmetry_features(image, mask):
    axes = _pca_axes(mask)
    if axes is None:
        return {
            "abcd_v2_pca_area_asymmetry_major": 0.0,
            "abcd_v2_pca_area_asymmetry_minor": 0.0,
            "abcd_v2_pca_color_asymmetry_major": 0.0,
            "abcd_v2_pca_color_asymmetry_minor": 0.0,
        }

    coords, center, major, minor = axes
    pixels = image[mask].astype(np.float32)
    centered = coords - center
    proj_major = centered @ major
    proj_minor = centered @ minor

    def split_stats(proj):
        pos = proj >= 0
        neg = proj < 0
        pos_count = pos.sum()
        neg_count = neg.sum()
        area_asym = abs(pos_count - neg_count) / max(pos_count + neg_count, 1)
        if pos_count == 0 or neg_count == 0:
            color_asym = 0.0
        else:
            color_asym = np.linalg.norm(pixels[pos].mean(axis=0) - pixels[neg].mean(axis=0)) / 255.0
        return float(area_asym), float(color_asym)

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
    counts = np.array([_box_count(binary, s) for s in sizes], dtype=np.float32)
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0
    coeff = np.polyfit(np.log(1 / np.array(sizes)[valid]), np.log(counts[valid]), 1)
    return float(coeff[0])


def _border_v2_features(mask):
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
    radius_mean = radius.mean()
    radial_cv = radius.std() / max(radius_mean, 1e-6)
    radial_range = (radius.max() - radius.min()) / max(radius_mean, 1e-6)

    angles = np.arctan2(ys - center_y, xs - center_x)
    order = np.argsort(angles)
    sorted_radius = radius[order]
    roughness = np.abs(np.diff(np.r_[sorted_radius, sorted_radius[0]])).mean() / max(radius_mean, 1e-6)
    border = np.zeros_like(mask, dtype=bool)
    border[ys.astype(int), xs.astype(int)] = True

    return {
        "abcd_v2_border_radial_cv": float(radial_cv),
        "abcd_v2_border_radial_range": float(radial_range),
        "abcd_v2_border_roughness": float(roughness),
        "abcd_v2_border_fractal_dim": _fractal_dimension(border),
    }


def _diagnostic_color_features(image, mask):
    pixels = image[_safe_region(mask)].astype(np.float32)
    if pixels.size == 0:
        pixels = image.reshape(-1, 3).astype(np.float32)
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    intensity = pixels.mean(axis=1)

    black = intensity < 45
    white = (intensity > 210) & (np.abs(r - g) < 25) & (np.abs(r - b) < 25)
    red = (r > 120) & (r > g * 1.2) & (r > b * 1.2)
    blue_gray = (b > r * 0.9) & (b > g * 0.9) & (intensity < 150)
    dark_brown = (r > g) & (g > b) & (intensity >= 45) & (intensity < 120)
    light_brown = (r > g) & (g > b) & (intensity >= 120) & (intensity < 210)

    masks = [black, white, red, blue_gray, dark_brown, light_brown]
    ratios = np.array([m.mean() for m in masks], dtype=np.float32)
    present = ratios > 0.01
    prob = ratios[ratios > 0]
    entropy = float(-(prob * np.log2(prob)).sum()) if prob.size else 0.0

    return {
        "abcd_v2_diag_color_count": float(present.sum()),
        "abcd_v2_black_ratio": float(ratios[0]),
        "abcd_v2_white_ratio": float(ratios[1]),
        "abcd_v2_red_ratio": float(ratios[2]),
        "abcd_v2_blue_gray_ratio": float(ratios[3]),
        "abcd_v2_dark_brown_ratio": float(ratios[4]),
        "abcd_v2_light_brown_ratio": float(ratios[5]),
        "abcd_v2_diag_color_entropy": entropy,
        "abcd_v2_dominant_color_ratio": float(ratios.max()) if ratios.size else 0.0,
    }


def extract_abcd_features(image, mask):
    """Extract ABCD-rule inspired handcrafted features.

    A: asymmetry, B: border irregularity, C: color variation,
    D: diameter / structural scale approximation.
    """
    mask = _safe_region(mask)
    h, w = mask.shape
    bbox = _foreground_bbox(mask)
    x0, y0, x1, y1 = bbox
    bbox_w = max(x1 - x0, 1)
    bbox_h = max(y1 - y0, 1)
    bbox_area = float(bbox_w * bbox_h)
    image_area = float(max(h * w, 1))

    area = float(mask.sum())
    eroded = ndi.binary_erosion(mask)
    perimeter = float(np.logical_xor(mask, eroded).sum())
    compactness = float((perimeter ** 2) / max(area, 1.0))
    circularity = float(4.0 * np.pi * area / max(perimeter ** 2, 1.0))
    extent = float(area / bbox_area)

    cropped_mask = _crop_to_bbox(mask, bbox)
    gray = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    lab = np.asarray(Image.fromarray(image).convert("LAB"))

    gray_pixels = gray[mask]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_pixels = grad[mask]

    features = {
        # A: Asymmetry
        "abcd_asymmetry_lr": _binary_asymmetry(cropped_mask, axis=1),
        "abcd_asymmetry_ud": _binary_asymmetry(cropped_mask, axis=0),
        "abcd_color_asymmetry_lr": _half_color_asymmetry(image, mask, vertical=True),
        "abcd_color_asymmetry_ud": _half_color_asymmetry(image, mask, vertical=False),
        # B: Border
        "abcd_border_irregularity": compactness,
        "abcd_circularity": circularity,
        "abcd_extent": extent,
        "abcd_perimeter_area_ratio": float(perimeter / max(area, 1.0)),
        # C: Color
        "abcd_rgb_std_mean": _channel_std_mean(image, mask),
        "abcd_hsv_std_mean": _channel_std_mean(hsv, mask),
        "abcd_lab_std_mean": _channel_std_mean(lab, mask),
        "abcd_gray_entropy": _entropy(gray_pixels),
        "abcd_color_range_mean": _color_range_mean(image, mask),
        "abcd_color_bin_count": _color_bin_count(image, mask),
        # D: Diameter / approximate structure
        "abcd_area_ratio": float(area / image_area),
        "abcd_bbox_diameter_ratio": float(np.sqrt(bbox_w ** 2 + bbox_h ** 2) / np.sqrt(w ** 2 + h ** 2)),
        "abcd_major_minor_ratio": float(max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)),
        "abcd_gradient_mean": float(grad_pixels.mean()) if grad_pixels.size else 0.0,
        "abcd_gradient_std": float(grad_pixels.std()) if grad_pixels.size else 0.0,
    }
    return features


def extract_abcd_v2_features(image, mask):
    features = extract_abcd_features(image, mask)
    mask = _safe_region(mask)
    features.update(_pca_asymmetry_features(image, mask))
    features.update(_border_v2_features(mask))
    features.update(_diagnostic_color_features(image, mask))
    return features

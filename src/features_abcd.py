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

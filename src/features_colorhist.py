import numpy as np
from PIL import Image


def _safe_mask(mask):
    mask = np.asarray(mask).astype(bool)
    return mask if mask.any() else np.ones_like(mask, dtype=bool)


def _hist(prefix, values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = float(hist.sum())
    if total <= 0:
        hist = np.zeros(bins, dtype=np.float32)
    else:
        hist = hist.astype(np.float32) / total
    return {f"{prefix}_{idx:02d}": float(value) for idx, value in enumerate(hist)}


def _hist2d(prefix, x_values, y_values, x_bins, y_bins, x_range, y_range):
    hist, _, _ = np.histogram2d(
        x_values,
        y_values,
        bins=[x_bins, y_bins],
        range=[x_range, y_range],
    )
    total = float(hist.sum())
    if total <= 0:
        hist = np.zeros((x_bins, y_bins), dtype=np.float32)
    else:
        hist = hist.astype(np.float32) / total
    features = {}
    flat = hist.reshape(-1)
    for idx, value in enumerate(flat):
        features[f"{prefix}_{idx:02d}"] = float(value)
    return features


def _entropy(values, bins, value_range):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    prob = hist[hist > 0].astype(np.float64) / total
    return float(-(prob * np.log2(prob)).sum())


def extract_colorhist_features(image, mask):
    mask = _safe_mask(mask)
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    lab = np.asarray(Image.fromarray(image).convert("LAB"))

    hsv_pixels = hsv[mask].astype(np.float32)
    lab_pixels = lab[mask].astype(np.float32)
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    l_channel, a_channel, b_channel = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]

    features = {}
    features.update(_hist("colorhist_hue", h, bins=18, value_range=(0, 255)))
    features.update(_hist("colorhist_sat", s, bins=8, value_range=(0, 255)))
    features.update(_hist("colorhist_val", v, bins=8, value_range=(0, 255)))
    features.update(_hist("colorhist_lab_l", l_channel, bins=8, value_range=(0, 255)))
    features.update(_hist("colorhist_lab_a", a_channel, bins=8, value_range=(0, 255)))
    features.update(_hist("colorhist_lab_b", b_channel, bins=8, value_range=(0, 255)))
    features.update(_hist2d("colorhist_hs", h, s, 12, 4, (0, 255), (0, 255)))
    features.update(_hist2d("colorhist_ab", a_channel, b_channel, 6, 6, (0, 255), (0, 255)))
    features["colorhist_hue_entropy18"] = _entropy(h, bins=18, value_range=(0, 255))
    features["colorhist_sat_entropy8"] = _entropy(s, bins=8, value_range=(0, 255))
    features["colorhist_lab_ab_entropy36"] = _entropy(
        np.clip((a_channel // 43) * 6 + (b_channel // 43), 0, 35),
        bins=36,
        value_range=(0, 36),
    )
    return features

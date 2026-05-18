import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _safe_pixels(image, region):
    pixels = image[region]
    if pixels.size == 0:
        pixels = image.reshape(-1, image.shape[-1])
    return pixels.astype(np.float32)


def _safe_values(values, region):
    pixels = values[region]
    if pixels.size == 0:
        pixels = values.reshape(-1)
    return pixels.astype(np.float32)


def _background_ring(mask, iterations=8):
    dilated = ndi.binary_dilation(mask, iterations=iterations)
    ring = np.logical_and(dilated, ~mask)
    if ring.any():
        return ring
    fallback = ~mask
    return fallback if fallback.any() else np.ones_like(mask, dtype=bool)


def _entropy(values, bins=32, value_range=(0, 255)):
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _channel_contrast(prefix, lesion_pixels, background_pixels):
    features = {}
    for idx, channel_name in enumerate(["c0", "c1", "c2"]):
        lesion_channel = lesion_pixels[:, idx]
        background_channel = background_pixels[:, idx]
        mean_diff = lesion_channel.mean() - background_channel.mean()
        std_diff = lesion_channel.std() - background_channel.std()
        features[f"{prefix}_{channel_name}_mean_diff"] = float(mean_diff)
        features[f"{prefix}_{channel_name}_abs_mean_diff"] = float(abs(mean_diff))
        features[f"{prefix}_{channel_name}_std_diff"] = float(std_diff)
    return features


def extract_contrast_features(image, mask):
    lesion = mask if mask.any() else np.ones_like(mask, dtype=bool)
    background = _background_ring(lesion)

    features = {
        "contrast_lesion_area_ratio": float(lesion.mean()),
        "contrast_background_ring_ratio": float(background.mean()),
    }

    rgb_lesion = _safe_pixels(image, lesion)
    rgb_background = _safe_pixels(image, background)
    features.update(_channel_contrast("contrast_rgb", rgb_lesion, rgb_background))

    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hsv_lesion = _safe_pixels(hsv, lesion)
    hsv_background = _safe_pixels(hsv, background)
    features.update(_channel_contrast("contrast_hsv", hsv_lesion, hsv_background))

    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    lab_lesion = _safe_pixels(lab, lesion)
    lab_background = _safe_pixels(lab, background)
    features.update(_channel_contrast("contrast_lab", lab_lesion, lab_background))

    gray = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)
    gray_lesion = _safe_values(gray, lesion)
    gray_background = _safe_values(gray, background)
    features["contrast_gray_mean_diff"] = float(gray_lesion.mean() - gray_background.mean())
    features["contrast_gray_std_diff"] = float(gray_lesion.std() - gray_background.std())
    features["contrast_gray_entropy_diff"] = float(_entropy(gray_lesion) - _entropy(gray_background))

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_lesion = _safe_values(grad, lesion)
    grad_background = _safe_values(grad, background)
    features["contrast_grad_mean_diff"] = float(grad_lesion.mean() - grad_background.mean())
    features["contrast_grad_std_diff"] = float(grad_lesion.std() - grad_background.std())
    return features

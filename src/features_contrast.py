import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def _safe_pixels(image, region):
    pixels = image[region]
    if pixels.size == 0:
        pixels = image.reshape(-1, image.shape[-1])
    return pixels


def _gray_image(image):
    return np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)


def _entropy(values, bins=32, value_range=(0, 255)):
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=value_range, density=False)
    total = hist.sum()
    if total == 0:
        return 0.0
    prob = hist[hist > 0] / total
    return float(-(prob * np.log2(prob)).sum())


def _channel_mean_std(prefix, lesion_pixels, background_pixels):
    features = {}
    for idx, channel_name in enumerate(["0", "1", "2"]):
        lesion_channel = lesion_pixels[:, idx].astype(np.float32)
        background_channel = background_pixels[:, idx].astype(np.float32)
        lesion_mean = lesion_channel.mean()
        background_mean = background_channel.mean()
        lesion_std = lesion_channel.std()
        background_std = background_channel.std()

        features[f"{prefix}_mean_diff_{channel_name}"] = float(lesion_mean - background_mean)
        features[f"{prefix}_std_diff_{channel_name}"] = float(lesion_std - background_std)
        features[f"{prefix}_abs_mean_diff_{channel_name}"] = float(abs(lesion_mean - background_mean))
    return features


def _background_ring(mask, iterations=8):
    dilated = ndi.binary_dilation(mask, iterations=iterations)
    ring = np.logical_and(dilated, ~mask)
    if ring.any():
        return ring
    return ~mask


def extract_contrast_features(image, mask):
    """Extract lesion-vs-surrounding-background contrast features.

    The mask defines the lesion. The background is a narrow ring around the
    lesion, which approximates nearby normal skin better than the full image.
    """
    lesion = mask
    background = _background_ring(mask)

    features = {}
    features["contrast_lesion_area_ratio"] = float(lesion.mean())
    features["contrast_background_ring_ratio"] = float(background.mean())

    rgb_lesion = _safe_pixels(image, lesion)
    rgb_background = _safe_pixels(image, background)
    features.update(_channel_mean_std("contrast_rgb", rgb_lesion, rgb_background))

    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hsv_lesion = _safe_pixels(hsv, lesion)
    hsv_background = _safe_pixels(hsv, background)
    features.update(_channel_mean_std("contrast_hsv", hsv_lesion, hsv_background))

    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    lab_lesion = _safe_pixels(lab, lesion)
    lab_background = _safe_pixels(lab, background)
    features.update(_channel_mean_std("contrast_lab", lab_lesion, lab_background))

    gray = _gray_image(image)
    gray_lesion = gray[lesion] if lesion.any() else gray.reshape(-1)
    gray_background = gray[background] if background.any() else gray.reshape(-1)
    features["contrast_gray_mean_diff"] = float(gray_lesion.mean() - gray_background.mean())
    features["contrast_gray_std_diff"] = float(gray_lesion.std() - gray_background.std())
    features["contrast_gray_entropy_diff"] = float(_entropy(gray_lesion) - _entropy(gray_background))

    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_lesion = grad[lesion] if lesion.any() else grad.reshape(-1)
    grad_background = grad[background] if background.any() else grad.reshape(-1)
    features["contrast_grad_mean_diff"] = float(grad_lesion.mean() - grad_background.mean())
    features["contrast_grad_std_diff"] = float(grad_lesion.std() - grad_background.std())

    return features

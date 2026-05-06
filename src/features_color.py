import numpy as np
from PIL import Image


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


def extract_color_features(image, mask):
    features = {}
    rgb_pixels = _region_pixels(image, mask)
    features.update(_channel_stats("rgb", rgb_pixels))

    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hsv_pixels = _region_pixels(hsv, mask)
    features.update(_channel_stats("hsv", hsv_pixels))

    lab = np.asarray(Image.fromarray(image).convert("LAB"))
    lab_pixels = _region_pixels(lab, mask)
    features.update(_channel_stats("lab", lab_pixels))
    return features

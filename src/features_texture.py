import numpy as np
from PIL import Image


def extract_texture_features(image, mask):
    gray = np.asarray(Image.fromarray(image).convert("L")).astype(np.float32)
    pixels = gray[mask] if mask.any() else gray.reshape(-1)
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    grad_pixels = grad[mask] if mask.any() else grad.reshape(-1)
    hist, _ = np.histogram(pixels, bins=16, range=(0, 255), density=True)

    features = {f"gray_hist_{idx}": float(value) for idx, value in enumerate(hist)}
    features["gray_mean"] = float(pixels.mean())
    features["gray_std"] = float(pixels.std())
    features["gray_p10"] = float(np.percentile(pixels, 10))
    features["gray_p90"] = float(np.percentile(pixels, 90))
    features["grad_mean"] = float(grad_pixels.mean())
    features["grad_std"] = float(grad_pixels.std())
    return features

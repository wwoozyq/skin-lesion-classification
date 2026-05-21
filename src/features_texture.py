import numpy as np
from PIL import Image
from scipy import stats
from skimage import feature as skfeature, filters, exposure


def extract_texture_features(image, mask):
    gray = np.asarray(Image.fromarray(image).convert("L"))
    features = {}
    features.update(extract_gray_histogram(gray, mask))
    features.update(extract_lbp(gray, mask))
    features.update(extract_glcm(gray, mask))
    features.update(extract_gradient_stats(gray, mask))
    return features


def _safe_mask_pixels(arr, mask):
    if mask.any():
        return arr[mask]
    return arr.reshape(-1)


# --- Gray-level histogram (original baseline feature) ---
def extract_gray_histogram(gray, mask):
    features = {}
    pixels = _safe_mask_pixels(gray, mask)
    hist, _ = np.histogram(pixels, bins=16, range=(0, 255), density=True)
    for idx, value in enumerate(hist):
        features[f"gray_hist_{idx}"] = float(value)
    features["gray_mean"] = float(pixels.mean())
    features["gray_std"] = float(pixels.std())
    features["gray_p10"] = float(np.percentile(pixels, 10))
    features["gray_p90"] = float(np.percentile(pixels, 90))
    return features


# --- LBP: single scale (R=2) ---

def extract_lbp(gray, mask):
    features = {}
    radius = 2
    n_points = 8 * radius
    lbp = skfeature.local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_pixels = _safe_mask_pixels(lbp, mask)
    n_bins = int(n_points + 2)
    hist, _ = np.histogram(lbp_pixels, bins=n_bins, range=(0, n_bins), density=True)
    for idx, value in enumerate(hist):
        features[f"lbp_h{idx}"] = float(value)
    return features


# --- GLCM (spatial co-occurrence statistics) ---

def extract_glcm(gray, mask):
    features = {}
    roi = gray.copy().astype(np.float32)
    roi[~mask] = 0
    roi[mask] = roi[mask] + 1
    roi = roi.astype(np.uint8)

    distances = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    n_levels = 32

    roi_q = exposure.rescale_intensity(roi, out_range=(0, n_levels - 1)).astype(np.uint8)

    glcm = skfeature.graycomatrix(
        roi_q,
        distances=distances,
        angles=angles,
        levels=n_levels,
        symmetric=True,
        normed=True,
    )

    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    for prop in props:
        values = skfeature.graycoprops(glcm, prop)
        mean_per_dist = values.mean(axis=1)
        for d_idx, d_val in enumerate(distances):
            features[f"glcm_{prop}_d{d_val}"] = float(mean_per_dist[d_idx])
    return features


# --- Gradient Statistics (enhanced) ---

def extract_gradient_stats(gray, mask):
    features = {}
    grad_mag = filters.sobel(gray.astype(np.float32))
    grad_pixels = _safe_mask_pixels(grad_mag, mask)

    features["grad_mean"] = float(grad_pixels.mean())
    features["grad_std"] = float(grad_pixels.std())

    if len(grad_pixels) > 30:
        features["grad_skew"] = float(stats.skew(grad_pixels))
        features["grad_kurtosis"] = float(stats.kurtosis(grad_pixels))
    else:
        features["grad_skew"] = 0.0
        features["grad_kurtosis"] = 0.0
    return features
